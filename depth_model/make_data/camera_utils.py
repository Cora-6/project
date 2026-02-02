"""
相机参数与变换工具
"""
import torch
import math
import json

def c2wtow2c(c2w):
    """相机 c2w 矩阵转 w2c"""
    return torch.inverse(c2w)

def get_camera_matrices(c2w, fovy_rad, height, width):
    """获取相机变换的 w2c 和 MVP 投影矩阵"""
    w2c = c2wtow2c(c2w)
    aspect_ratio = width / height
    focal_length_y = 1.0 / math.tan(fovy_rad / 2.0)
    proj = torch.zeros(4, 4, device=c2w.device)
    proj[0, 0] = focal_length_y / aspect_ratio
    proj[1, 1] = focal_length_y
    proj[2, 2] = (1000.1) / (0.1 - 1000.0)
    proj[2, 3] = (2 * 1000.0 * 0.1) / (0.1 - 1000.0)
    proj[3, 2] = -1.0
    mvp = torch.bmm(proj.unsqueeze(0), w2c)
    return w2c, mvp

def parse_json_camera(json_path, device):
    """解析单帧 json 文件中的相机参数，返回 c2w 及 fov"""
    with open(json_path, 'r') as f: data = json.load(f)
    loc = torch.tensor(data["location"], device=device)
    x, y, z = data["rotation_euler"]
    # Blender Euler -> Matrix (Z Y X)
    cx, sx = math.cos(x), math.sin(x); cy, sy = math.cos(y), math.sin(y); cz, sz = math.cos(z), math.sin(z)
    Rx = torch.tensor([[1,0,0],[0,cx,-sx],[0,sx,cx]], device=device)
    Ry = torch.tensor([[cy,0,sy],[0,1,0],[-sy,0,cy]], device=device)
    Rz = torch.tensor([[cz,-sz,0],[sz,cz,0],[0,0,1]], device=device)
    R = torch.mm(Rz, torch.mm(Ry, Rx))

    c2w = torch.eye(4, device=device)
    c2w[:3, :3] = R; c2w[:3, 3] = loc
    # Flip Y 轴
    c2w = torch.mm(c2w, torch.tensor([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], device=device, dtype=c2w.dtype))
    lens = data["data"]["lens"]
    sensor_w = data["data"]["sensor_width"]
    fovx = 2 * math.atan(sensor_w / (2 * lens))
    return c2w.unsqueeze(0), fovx