import os, glob, json, math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from skimage.transform import resize
from transformers import AutoTokenizer
from functools import partial
from scipy.spatial.transform import Rotation as R

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def generate_text(config):
    """
    根据配置信息生成描述文本。

    Args:
        config (dict): 配置信息。

    Returns:
        str: 生成的描述文本。
        "city topview from camera, camera conf as following. camera_height:{camera_height},tilt_deg:{tilt_deg:.1f},lens:{lens}"
    """
    camera_height = config['location'][2]  # Z坐标
    tilt_rad = math.pi / 2 - config['rotation_euler'][0]
    tilt_deg = math.degrees(tilt_rad)
    lens = config['data']['lens']
    conf_str = f"{{camera_height:{camera_height},tilt_deg:{tilt_deg:.1f},lens:{lens}}}"
    return f"city topview from camera, camera conf as following. {conf_str}"

class HSIControlNetDataset(Dataset):
    """
    每次返回一个 4-in-1 样本：
        image            : (3,512,512)
        conditioning     : (48,512,512)
        text / input_ids : 与原来一致
    """
    def __init__(self,
                 root_dir: str,
                 image_size: int = 512,
                 tokenizer_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",):
        self.root_dir   = root_dir
        self.image_size = image_size
        self.tokenizer  = AutoTokenizer.from_pretrained(
            tokenizer_name, subfolder="tokenizer", use_fast=False)
        print("dataset name:",root_dir)
        # 1. 收集所有 json（即所有单视角目录）
        jsons = sorted(glob.glob(f"{root_dir}/**/*.json", recursive=True))
        folders = [os.path.dirname(j) for j in jsons]
        n = len(folders)
        print("folders len",n)
        # 2. 4 个一组
        self.groups = [folders[i:i+4] for i in range(0, len(folders), 4)
                       if i+4<=n]   # 去掉尾部不足 4 的
        
        print("groups len",len(self.groups))
        # 3. 全局 HSI 数据（仅加载一次）
        self.hsi_path = '/data/try_small/result_new/Houston18.mat'
        with h5py.File(self.hsi_path, 'r') as f:
            self.hsi = np.array(f['ori_data'])# (C,H,W)
        self.hsi_norm = self.hsi / (self.hsi.max(axis=(1,2), keepdims=True) + 1e-8)#每个通道里，整张 H×W 图除以该通道自己的最大值
        self.hsi_norm = self.hsi_norm.transpose(1, 2, 0)#CHW -> HWC
        # match得到的固定透视矩阵
        self.P = np.array([
            [4.80769231e-02, 2.55899318e+00, 4.44000000e+02],
            [2.55528846e+00, -4.82433141e-02, 1.21000000e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])

        self.warped_hsi = cv2.warpPerspective(self.hsi_norm, self.P, (4096, 1024),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)#对齐到topview
        out_size_per_subimage = image_size // 2
        scale = out_size_per_subimage / 512 # 512 is the canonical size
        self.scale_mat = np.array([
            [scale, 0., 0],
            [0., scale, 0.],
            [0., 0., 1.]
        ])

    def __len__(self):
        return len(self.groups)

    # ---------- 下面 3 个工具函数 -------------
    def _warp_hsi(self, H_mat, out_size):
        """把 (C,H,W) warp 成 (C,out_size,out_size)"""
        M = self.scale_mat @ H_mat
        warped = cv2.warpPerspective(self.warped_hsi, M, (out_size, out_size),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        res = warped.transpose(2,0,1)
        return res       # (C,H,W)

    def _process_depth_normal(self, folder, size):
        """读取 depth 和 normal 并拼接, resize, transpose"""
        d_path = os.path.join(folder, 'camconf_depth.npy')
        n_path = os.path.join(folder, 'camconf_normal_cam.npy')
        
        # 加载 npy
        depth = np.load(d_path)   # (H, W)
        normal = np.load(n_path)  # (H, W, 3)

        # 确保通道维度正确
        if depth.ndim == 2:
            depth = depth[..., None]
        
        # Resize 到 size x size
        if depth.shape[0] != size or depth.shape[1] != size:
            depth = cv2.resize(depth, (size, size), interpolation=cv2.INTER_NEAREST)
            # Resize 后如果通道丢失需加回
            if depth.ndim == 2: depth = depth[..., None]

        if normal.shape[0] != size or normal.shape[1] != size:
            normal = cv2.resize(normal, (size, size), interpolation=cv2.INTER_LINEAR)
        
        # 拼接: Normal(3) + Depth(1)   -> (H, W, 4)
        merged = np.concatenate([normal, depth], axis=-1)
        
        # Transpose -> (C, H, W) -> (4, size, size)
        return merged.transpose(2, 0, 1).astype(np.float32)
    
    def _get_camera_matrix(self, config):
        """
        计算外参矩阵。
        输入 config 包含 location (x,y,z) 和 rotation_euler (x,y,z radians)。
        假设物体在世界原点 (0,0,0)。
        """
        location = np.array(config['location'])
        rotation_euler = np.array(config['rotation_euler'])

        # 1. 计算旋转矩阵 (Camera to World 旋转)
        # Blender Euler 是 XYZ 顺序
        r = R.from_euler('xyz', rotation_euler, degrees=False)
        R_c2w = r.as_matrix()

        # 2. 构建 RT 矩阵 (4x4)
        # 这是一个 Camera-to-World (c2w) 矩阵，描述相机在世界坐标系中的位姿
        # c2w = [ R | T ]
        #       [ 0 | 1 ]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = location

        # 如果你需要 World-to-Camera (w2c)，即物体在相机坐标系下的坐标，请取消下面这行的注释
        # w2c = np.linalg.inv(c2w)
        # return w2c.flatten()

        # 默认返回 c2w (相机在世界坐标系下的位置和朝向)
        return c2w


    def __getitem__(self, idx):
        folders = self.groups[idx]           # 长度为 4 的列表
        # ---- 1. 读取 Depth+Normal 并堆叠 (Stack) ----
        # 结果维度: (4, 4, 256, 256) -> [Frames, Channels, H, W]
        imgs_list = [self._process_depth_normal(f, 128) for f in folders]
        image = np.stack(imgs_list, axis=0) 
        # ---- 2. 处理 HSI 并堆叠 (Stack) ----
        # 之前是拼成 2x2，现在改为在 Batch 维度堆叠以匹配 Image
        Hs = [np.load(os.path.join(f, 'homography_matrix.npy')) for f in folders]
        warp_hsi_parital_func = partial(self._warp_hsi, out_size=256)
        cubes = list(map(warp_hsi_parital_func, Hs)) # List of (48, 256, 256)
        
        cond = np.stack(cubes, axis=0) # (4, 48, 256, 256)
        # ---- 3. 文本（取第一个文件夹的即可）----
        cfg0  = load_config(os.path.join(folders[0], 'camconf.json'))
        text = generate_text(cfg0)
        input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).input_ids.squeeze(0)
        # ---- 4. 处理相机外参 ----
        configs = [load_config(os.path.join(f, 'camconf.json')) for f in folders]
        # 计算每个 view 的 4x4 矩阵并展平为 16
        locations = np.array([cfg['location'] for cfg in configs])
        centroid = locations.mean(axis=0)          # 形状 (3,)
        centroid[2] = 0  # 只在 XY 平面上居中
        for cfg in configs:
            cfg['location'] = ((np.array(cfg['location']) - centroid) * 0.01).tolist()
        camera_matrices = [self._get_camera_matrix(cfg).flatten() for cfg in configs]
        camera_matrix = np.stack(camera_matrices, axis=0).astype(np.float32) # (4, 16)
        return dict(text=text,
                    input_ids=input_ids,
                    image=image,
                    conditioning_image=cond,
                    camera_matrix=camera_matrix)

