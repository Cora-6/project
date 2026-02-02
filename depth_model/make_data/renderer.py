"""
渲染器实现
"""
import torch
import nvdiffrast.torch as dr
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from camera_utils import get_camera_matrices

class SimpleRenderer:
    """基础三维渲染器，加载 obj 并提供法线+深度输出"""
    def __init__(self, device):
        self.device = device
        self.ctx = dr.RasterizeGLContext(device=self.device)
        self.mesh = None

    def load_mesh(self, obj_path: str):
        """加载 OBJ 文件"""
        import os
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"找不到模型: {obj_path}")
        print(f"[Info] Loading mesh: {obj_path} ...")
        verts, faces, _ = load_obj(obj_path, device=self.device)
        mesh_pytorch3d = Meshes(verts=[verts], faces=[faces.verts_idx]).to(self.device)
        self.mesh = {
            "v_pos": mesh_pytorch3d.verts_padded(),
            "t_pos_idx": mesh_pytorch3d.faces_padded(),
            "v_nrm": mesh_pytorch3d.verts_normals_padded()
        }
        print(f"[Info] Loaded. V: {self.mesh['v_pos'].shape[1]}, F: {self.mesh['t_pos_idx'].shape[1]}")

    def forward(self, c2w, fovy_rad, height, width):
        """执行渲染，输出深度和法线图"""
        if self.mesh is None: raise RuntimeError("Mesh not loaded.")
        batch_size = c2w.shape[0]
        w2c, mvp = get_camera_matrices(c2w, fovy_rad, height, width)
        v_pos_clip = torch.cat([self.mesh['v_pos'], torch.ones_like(self.mesh['v_pos'][..., :1])], dim=-1)
        v_pos_clip = torch.bmm(v_pos_clip, mvp.permute(0, 2, 1))
        rast, _ = dr.rasterize(self.ctx, v_pos_clip, self.mesh['t_pos_idx'][0].int(), (height, width))
        mask = (rast[..., 3:] > 0).float()
        v_pos_cam = torch.bmm(
            torch.cat([self.mesh['v_pos'], torch.ones_like(self.mesh['v_pos'][..., :1])], dim=-1),
            w2c.permute(0, 2, 1)
        )
        v_depth = -v_pos_cam[..., 2:].contiguous()
        interpolated_depth, _ = dr.interpolate(v_depth, rast, self.mesh['t_pos_idx'][0].int())
        depth = interpolated_depth[..., :1]
        interpolated_normal, _ = dr.interpolate(self.mesh['v_nrm'], rast, self.mesh['t_pos_idx'][0].int())
        normal_world = torch.nn.functional.normalize(interpolated_normal[..., :3], dim=-1)
        rot_matrix = w2c[:, :3, :3]
        normal_cam = torch.bmm(normal_world.view(batch_size, -1, 3), rot_matrix.permute(0, 2, 1))
        normal_cam = normal_cam.view(batch_size, height, width, 3)
        normal_cam = torch.nn.functional.normalize(normal_cam, dim=-1)
        depth[mask == 0] = 0.0
        bg_normal = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 1, 1, 3)
        normal_cam = torch.lerp(bg_normal, normal_cam, mask)
        normal_cam = normal_cam * torch.tensor([-1.0, -1.0, 1.0], device=self.device)[None,None,:]
        normal_cam_vis = (normal_cam + 1.0) / 2.0
        return {
            "depth": depth.squeeze(),
            "mask": mask.squeeze(),
            "normal_camera": normal_cam_vis.squeeze()
        }