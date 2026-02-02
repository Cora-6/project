# The function to parse JSON camera parameters into extrinsic and intrinsic matrices
import json, math, torch

def parse_json_camera_extrinsic_intrinsic(json_path, device, H, W, dtype=torch.float32, flip_y=True,center=[0.0,0.0,0.0]):
    """
    返回:
      extrinsic: (1, 4, 4) world2cam
      intrinsic: (1, 3, 3) K
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # 位置与欧拉角
    loc = torch.tensor(data["location"], device=device, dtype=dtype)
    loc -= torch.tensor(center, device=device, dtype=dtype)  # 平移到以 center 为中心
    x, y, z = data["rotation_euler"]  # 假设 Blender 的 XYZ，矩阵按 ZYX 乘法

    # 旋转矩阵 R = Rz * Ry * Rx
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    Rx = torch.tensor([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx,  cx]], device=device, dtype=dtype)
    Ry = torch.tensor([[ cy, 0, sy],
                       [  0, 1,  0],
                       [-sy, 0, cy]], device=device, dtype=dtype)
    Rz = torch.tensor([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [ 0,   0, 1]], device=device, dtype=dtype)
    R = Rz @ (Ry @ Rx)  # ZYX

    # c2w
    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, :3] = R
    c2w[:3, 3] = loc

    # 如需翻转 Y 轴（看你的坐标系/渲染约定）
    if flip_y:
        flip = torch.tensor([[1,  0, 0, 0],
                             [0, -1, 0, 0],
                             [0,  0, 1, 0],
                             [0,  0, 0, 1]], device=device, dtype=dtype)
        c2w = c2w @ flip

    # world2cam (extrinsic)
    extrinsic = torch.linalg.inv(c2w)  # (1,4,4) 之前可 unsqueeze
    # extrinsic 传 camera2world
    # extrinsic = c2w
    extrinsic = extrinsic.unsqueeze(0)  # 保持批次维度

    # 内参 K
    lens = data["data"]["lens"]
    sensor_w = data["data"].get("sensor_width", 36.0)
    fovx = 2 * math.atan(sensor_w / (2 * lens))
    fx = W / (2 * math.tan(fovx / 2))

    # 若无 sensor_height，则按宽高比推 fovy
    fovy = 2 * math.atan(math.tan(fovx / 2) * (H / W))
    fy = H / (2 * math.tan(fovy / 2))

    cx_ = W * 0.5
    cy_ = H * 0.5

    intrinsic = torch.tensor([[fx, 0,  cx_],
                              [0,  fy, cy_],
                              [0,  0,   1]], device=device, dtype=dtype)
    intrinsic = intrinsic.unsqueeze(0)

    return extrinsic, intrinsic