from pathlib import Path
import sys

import numpy as np
import torch
from diffusers import StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler

PROJECT_ROOT = Path("/home/viplab/liukejia/project")
MV_ROOT = PROJECT_ROOT / "mv_model"
sys.path.insert(0, str(MV_ROOT))
from warp_and_controlnet_model import WarpAndControlNetModel

INPUT = PROJECT_ROOT / "muufl_target001_hsi" / "target001_hsi48_fourviews_256_fp16_lens100_bandnorm.npz"
OUTPUT = PROJECT_ROOT / "muufl_target001_sideview_outputs_lens100_bandnorm"
CKPT = Path("/data/mv_model_0_with_drop/checkpoint-32500/controlnet")
BASE_MODEL = "hotshotco/SDXL-512"
PROMPT = (
    "city topview from camera, camera conf as following. "
    "{camera_height:173.205081,tilt_deg:60.0,lens:100.0}"
)

OUTPUT.mkdir(parents=True, exist_ok=True)
data = np.load(INPUT)
cubes = np.asarray(data["cubes"], dtype=np.float32)
if cubes.shape != (4, 256, 256, 48):
    raise RuntimeError(f"Unexpected cubes shape: {cubes.shape}")

# Same ordering and stitching as mv_model/test_grid.ipynb:
# A000=top-left, A090=top-right, A180=bottom-left, A270=bottom-right.
chw = [cube.transpose(2, 0, 1) for cube in cubes]
top = np.concatenate((chw[0], chw[1]), axis=2)
bottom = np.concatenate((chw[2], chw[3]), axis=2)
cond = np.concatenate((top, bottom), axis=1)
if cond.shape != (48, 512, 512):
    raise RuntimeError(f"Unexpected condition shape: {cond.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print("device:", device, "dtype:", dtype, flush=True)
print("prompt:", PROMPT, flush=True)

controlnet = WarpAndControlNetModel.from_pretrained(
    CKPT, torch_dtype=dtype, use_safetensors=True, safety_checker=None
).to(device=device, dtype=dtype)
if hasattr(controlnet, "warper") and controlnet.warper is not None:
    controlnet.warper = controlnet.warper.to(device=device, dtype=dtype)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=dtype,
    use_safetensors=True,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
if device == "cuda":
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)

cond_tensor = torch.from_numpy(cond).unsqueeze(0).to(device=device, dtype=dtype)
with torch.inference_mode():
    grid = pipe(prompt=PROMPT, image=cond_tensor, guidance_scale=3.0).images[0]

grid_path = OUTPUT / "target001_fourviews_grid.png"
grid.save(grid_path)
labels = ("A000", "A090", "A180", "A270")
boxes = ((0, 0, 256, 256), (256, 0, 512, 256),
         (0, 256, 256, 512), (256, 256, 512, 512))
for label, box in zip(labels, boxes):
    path = OUTPUT / f"target001_{label}_sideview.png"
    grid.crop(box).save(path)
    print("saved:", path, flush=True)
print("saved grid:", grid_path, flush=True)
