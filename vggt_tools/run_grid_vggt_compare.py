import gc
import glob
import sys
from pathlib import Path

import torch

PROJECT = Path("/home/viplab/liukejia/project/vggt")
ROOT = Path("/data/muufl_grid_50m_pipeline")
sys.path[:0] = [str(PROJECT), str(PROJECT / "vggt")]
from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda"
model = VGGT()
weights = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
    map_location="cpu",
)
model.load_state_dict(weights)
model.eval().to(device)
amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def reconstruct(image_dir, output_path):
    names = sorted(glob.glob(str(image_dir / "*")))
    if len(names) != 4:
        raise RuntimeError(f"Expected 4 images in {image_dir}, got {len(names)}")
    images = load_and_preprocess_images(names).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
        pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    pred["extrinsic"] = extrinsic
    pred["intrinsic"] = intrinsic
    for key in list(pred):
        if isinstance(pred[key], torch.Tensor):
            pred[key] = pred[key].detach().cpu().numpy().squeeze(0)
    pred["pose_enc_list"] = None
    pred["world_points_from_depth"] = unproject_depth_map_to_point_map(
        pred["depth"], pred["extrinsic"], pred["intrinsic"]
    )
    scene = predictions_to_glb(
        pred, conf_thres=50.0, filter_by_frames="All", show_cam=True,
        mask_sky=False, mask_black_bg=False, mask_white_bg=False,
        target_dir=str(image_dir.parent), prediction_mode="Depthmap and Camera Branch",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(file_obj=str(output_path))
    del pred, images, scene
    gc.collect()
    torch.cuda.empty_cache()

targets = [f"T{i:03d}" for i in range(1, 16)]
for i, tid in enumerate(targets, 1):
    reconstruct(
        ROOT / "generated_sideviews" / tid / "images",
        ROOT / "vggt_generated" / f"{tid}_generated.glb",
    )
    reconstruct(
        ROOT / "inputs_3dtile" / tid / "images",
        ROOT / "vggt_3dtile" / f"{tid}_3dtile.glb",
    )
    print(f"VGGT_PROGRESS {i}/{len(targets)} {tid}", flush=True)
