import torch
from omegaconf import OmegaConf
from convert_from_ckpt import convert_ldm_unet_checkpoint, create_unet_diffusers_config

# ================= 配置区域 =================

BASE_DIR = "/home/viplab/liukejia/normal-depth-diffusion"
OUTPUT_DIR = "/home/viplab/liukejia/project/depth_model/nd_ckpt_diffuser"
CONFIG_PATH = f"{BASE_DIR}/configs/inference/nd/txtcond_mvsd-4-objaverse_finetune_wovae.yaml"
CKPT_PATH = f"{BASE_DIR}/pretrained_models/Damo_XR_Lab/Normal-Depth-Diffusion-Model/nd_mv_ema.ckpt"

# 模型参数配置
IMAGE_SIZE = 256
CAMERA_DIM = 16
CONTROLNET = False

# ================= 主转换逻辑 =================

# 1. 加载配置和权重
print(f"Loading config from {CONFIG_PATH}")
original_config = OmegaConf.load(CONFIG_PATH)

print(f"Loading checkpoint from {CKPT_PATH}")
checkpoint_full = torch.load(CKPT_PATH, map_location="cpu")
checkpoint = checkpoint_full['state_dict']

# 2. 提取 Camera Embedding 权重
print("Converting camera embedding weights...")
camera_ckpt = {}
camera_ckpt["camera_embedding.linear_1.weight"] = checkpoint["model.diffusion_model.camera_embed.0.weight"]
camera_ckpt["camera_embedding.linear_1.bias"] = checkpoint["model.diffusion_model.camera_embed.0.bias"]
camera_ckpt["camera_embedding.linear_2.weight"] = checkpoint["model.diffusion_model.camera_embed.2.weight"]
camera_ckpt["camera_embedding.linear_2.bias"] = checkpoint["model.diffusion_model.camera_embed.2.bias"]

# 3. 转换 UNet 主体权重,并转换 Camera Embedding 权重
print("Converting standard UNet weights...")
unet_config = create_unet_diffusers_config(original_config, image_size=IMAGE_SIZE, controlnet=CONTROLNET)
weights_ldm = convert_ldm_unet_checkpoint(
    checkpoint, unet_config, path=None, extract_ema=False, controlnet=CONTROLNET, skip_extract_state_dict=False
)

weights_ldm.update(camera_ckpt)

# 4. 保存转换后的权重
save_path = f"{OUTPUT_DIR}/nd_mv_ema_diffuser.pth"
print(f"Saving weights only to {save_path}")
torch.save(weights_ldm, save_path)
print("Done.")



