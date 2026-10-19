import gc
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/viplab/liukejia/project/vggt")
ROOT = Path("/data/muufl_grid_50m_pipeline")
sys.path[:0] = [str(PROJECT), str(PROJECT / "vggt")]
from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def to_ext4(ext):
    ext = np.asarray(ext)
    if ext.shape[-2:] == (4, 4):
        return ext
    out = np.tile(np.eye(4), (len(ext), 1, 1))
    out[:, :3, :4] = ext
    return out


def centers(ext):
    return np.linalg.inv(to_ext4(ext))[:, :3, 3]


def baseline_scale(pred_ext, exact_ext):
    pc, ec = centers(pred_ext), centers(exact_ext)
    pd, ed = [], []
    for i in range(len(pc)):
        for j in range(i + 1, len(pc)):
            pd.append(np.linalg.norm(pc[i] - pc[j]))
            ed.append(np.linalg.norm(ec[i] - ec[j]))
    ratios = np.asarray(ed) / np.maximum(np.asarray(pd), 1e-12)
    return float(np.median(ratios)), ratios


def exact_unproject(depth, exact_ext, exact_k, scale):
    depth = np.asarray(depth)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    n, h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    result = np.empty((n, h, w, 3), dtype=np.float32)
    c2w = np.linalg.inv(to_ext4(exact_ext))
    for i in range(n):
        z = depth[i].astype(np.float64) * scale
        x = (u - exact_k[i, 0, 2]) / exact_k[i, 0, 0] * z
        y = (v - exact_k[i, 1, 2]) / exact_k[i, 1, 1] * z
        cam = np.stack((x, y, z, np.ones_like(z)), axis=-1)
        result[i] = (cam @ c2w[i].T)[..., :3].astype(np.float32)
    return result


cam = np.load(ROOT / "grid_camera_parameters.npz")
camera_by_target = {}
for tid in sorted(set(cam["targets"].tolist())):
    idx = np.flatnonzero(cam["targets"] == tid)
    idx = idx[np.argsort(cam["angles"][idx])]
    camera_by_target[tid] = (cam["extrinsics"][idx], cam["intrinsics"][idx], cam["angles"][idx])

device = "cuda"
model = VGGT()
model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt", map_location="cpu"))
model.eval().to(device)
amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def reconstruct(image_dir, out_path, tid, branch):
    names = sorted(glob.glob(str(image_dir / "*")))
    images = load_and_preprocess_images(names).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
        pred = model(images)
    pred_ext, pred_k = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    pred["extrinsic"], pred["intrinsic"] = pred_ext, pred_k
    for key in list(pred):
        if isinstance(pred[key], torch.Tensor):
            pred[key] = pred[key].detach().cpu().numpy().squeeze(0)
    pred["pose_enc_list"] = None

    exact_ext44, k512, angles = camera_by_target[tid]
    exact_ext = exact_ext44[:, :3, :4].astype(np.float32)
    h, w = pred["depth"].shape[1:3]
    exact_k = k512.copy()
    exact_k[:, 0, :] *= w / 512.0
    exact_k[:, 1, :] *= h / 512.0
    scale, ratios = baseline_scale(pred["extrinsic"], exact_ext)
    pred["world_points_from_depth"] = exact_unproject(
        pred["depth"], exact_ext, exact_k, scale)
    pred["extrinsic"] = exact_ext
    pred["intrinsic"] = exact_k.astype(np.float32)
    pred["camera_source"] = "blender_exact_reprojection"

    scene = predictions_to_glb(
        pred, conf_thres=50.0, filter_by_frames="All", show_cam=True,
        mask_sky=False, mask_black_bg=False, mask_white_bg=False,
        target_dir=str(image_dir.parent), prediction_mode="Depthmap and Camera Branch")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(file_obj=str(out_path))
    meta = {"target": tid, "branch": branch, "camera_source": pred["camera_source"],
            "depth_scale": scale, "pairwise_baseline_scale_ratios": ratios.tolist(),
            "angles": angles.tolist(), "method": "scaled VGGT z-depth + exact Blender K,w2c"}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    del pred, images, scene
    gc.collect(); torch.cuda.empty_cache()
    return scale


for i in range(1, 16):
    tid = f"T{i:03d}"
    sg = reconstruct(ROOT / "generated_sideviews" / tid / "images",
                     ROOT / "vggt_generated_exact_reproject" / f"{tid}_generated_exact.glb", tid, "generated")
    st = reconstruct(ROOT / "inputs_3dtile" / tid / "images",
                     ROOT / "vggt_3dtile_exact_reproject" / f"{tid}_3dtile_exact.glb", tid, "3dtile")
    print(f"REPROJECT_PROGRESS {i}/15 {tid} scales={sg:.5f},{st:.5f}", flush=True)
