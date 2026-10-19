import gc
import glob
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/viplab/liukejia/project/vggt")
INPUT = PROJECT / "inputs_muufl_target001_3dtile"
OUTPUT = PROJECT / "outputs_muufl_target001_3dtile"
sys.path[:0] = [str(PROJECT), str(PROJECT / "vggt")]

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

names = sorted(glob.glob(str(INPUT / "images" / "*.png")))
if len(names) != 4:
    raise RuntimeError(f"Expected exactly 4 PNG inputs, got {len(names)}: {names}")
OUTPUT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("inputs:", names, flush=True)
print("device:", device, flush=True)

model = VGGT()
weights = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
    map_location="cpu",
)
model.load_state_dict(weights)
model.eval().to(device)

images = load_and_preprocess_images(names).to(device)
amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
    predictions = model(images)

extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"], images.shape[-2:]
)
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic
for key in list(predictions):
    if isinstance(predictions[key], torch.Tensor):
        predictions[key] = predictions[key].detach().cpu().numpy().squeeze(0)
predictions["pose_enc_list"] = None
predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
    predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
)

npz_path = OUTPUT / "target001_3dtile_vggt.npz"
glb_path = OUTPUT / "target001_3dtile_vggt.glb"
np.savez(npz_path, **predictions)
scene = predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="All",
    show_cam=True,
    mask_sky=False,
    mask_black_bg=False,
    mask_white_bg=False,
    target_dir=str(INPUT),
    prediction_mode="Depthmap and Camera Branch",
)
scene.export(file_obj=str(glb_path))

print("npz:", npz_path, flush=True)
print("glb:", glb_path, flush=True)
print("prediction shapes:", {k: getattr(v, "shape", None) for k, v in predictions.items()}, flush=True)
gc.collect()
torch.cuda.empty_cache()
