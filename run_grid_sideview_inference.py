from pathlib import Path
import sys

import numpy as np
import torch
from diffusers import StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler

PROJECT = Path("/home/viplab/liukejia/project")
sys.path.insert(0, str(PROJECT / "mv_model"))
from warp_and_controlnet_model import WarpAndControlNetModel

ROOT = Path("/data/muufl_grid_50m_pipeline")
INPUT = ROOT / "inputs_hsi"
OUTPUT = ROOT / "generated_sideviews"
CKPT = Path("/data/mv_model_0_with_drop/checkpoint-32500/controlnet")
PROMPT = "city topview from camera, camera conf as following. {camera_height:173.205081,tilt_deg:60.0,lens:100.0}"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
controlnet = WarpAndControlNetModel.from_pretrained(
    CKPT, torch_dtype=dtype, use_safetensors=True, safety_checker=None
).to(device=device, dtype=dtype)
if hasattr(controlnet, "warper") and controlnet.warper is not None:
    controlnet.warper = controlnet.warper.to(device=device, dtype=dtype)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "hotshotco/SDXL-512", controlnet=controlnet,
    torch_dtype=dtype, use_safetensors=True,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

files = sorted(INPUT.glob("T*/T*_fourviews_hsi48.npz"))
for index, path in enumerate(files, 1):
    tid = path.parent.name
    cubes = np.asarray(np.load(path)["cubes"], dtype=np.float32)
    chw = [x.transpose(2, 0, 1) for x in cubes]
    cond = np.concatenate((
        np.concatenate((chw[0], chw[1]), axis=2),
        np.concatenate((chw[2], chw[3]), axis=2),
    ), axis=1)
    tensor = torch.from_numpy(cond).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.inference_mode():
        grid = pipe(prompt=PROMPT, image=tensor, guidance_scale=3.0).images[0]
    out = OUTPUT / tid
    out.mkdir(parents=True, exist_ok=True)
    grid.save(out / f"{tid}_generated_grid.png")
    labels = ("A000", "A090", "A180", "A270")
    boxes = ((0, 0, 256, 256), (256, 0, 512, 256),
             (0, 256, 256, 512), (256, 256, 512, 512))
    image_dir = out / "images"
    image_dir.mkdir(exist_ok=True)
    for label, box in zip(labels, boxes):
        grid.crop(box).save(image_dir / f"{tid}_{label}_generated.png")
    print(f"SIDEVIEW_PROGRESS {index}/{len(files)} {tid}", flush=True)
