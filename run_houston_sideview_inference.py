from pathlib import Path
import sys
import numpy as np
import torch
from diffusers import StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler

PROJECT_ROOT = Path('/home/viplab/liukejia/project')
sys.path.insert(0, str(PROJECT_ROOT / 'mv_model'))
from warp_and_controlnet_model import WarpAndControlNetModel

INPUT = PROJECT_ROOT / 'houston/Houston_fourviews_hsi48.npz'
OUTPUT = PROJECT_ROOT / 'houston/inference'
CKPT = Path('/data/mv_model_0_with_drop/checkpoint-32500/controlnet')
PROMPT = 'city topview from camera, camera conf as following. {camera_height:173.205081,tilt_deg:60.0,lens:100.0}'
OUTPUT.mkdir(parents=True, exist_ok=True)
data = np.load(INPUT)
cubes = np.asarray(data['cubes'], dtype=np.float32)
if cubes.shape != (4, 256, 256, 48): raise RuntimeError(cubes.shape)
chw = [cube.transpose(2,0,1) for cube in cubes]
top = np.concatenate((chw[0], chw[1]), axis=2)
bottom = np.concatenate((chw[2], chw[3]), axis=2)
cond = np.concatenate((top, bottom), axis=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
controlnet = WarpAndControlNetModel.from_pretrained(CKPT, torch_dtype=dtype, use_safetensors=True, safety_checker=None).to(device=device,dtype=dtype)
if getattr(controlnet,'warper',None) is not None: controlnet.warper=controlnet.warper.to(device=device,dtype=dtype)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained('hotshotco/SDXL-512',controlnet=controlnet,torch_dtype=dtype,use_safetensors=True)
pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config)
if device=='cuda': pipe.enable_model_cpu_offload()
else: pipe=pipe.to(device)
with torch.inference_mode():
    grid=pipe(prompt=PROMPT,image=torch.from_numpy(cond).unsqueeze(0).to(device=device,dtype=dtype),guidance_scale=3.0).images[0]
grid.save(OUTPUT/'Houston_generated_grid.png')
for label,box in zip(('A000','A090','A180','A270'),((0,0,256,256),(256,0,512,256),(0,256,256,512),(256,256,512,512))):
    grid.crop(box).save(OUTPUT/f'Houston_{label}_generated.png')
print('HOUSTON_INFERENCE_DONE', OUTPUT, flush=True)
