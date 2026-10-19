import gc
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/viplab/liukejia/project/vggt")
ROOT = Path("/data/muufl_grid_50m_pipeline")
CAMERA_FILE = ROOT / "grid_camera_parameters.npz"
sys.path[:0] = [str(PROJECT), str(PROJECT / "vggt")]
from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def centers_from_w2c(ext):
    ext4 = np.tile(np.eye(4, dtype=np.float64), (len(ext), 1, 1))
    ext4[:, :3, :4] = np.asarray(ext)[:, :3, :4]
    return np.linalg.inv(ext4)[:, :3, 3]


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    r = u @ np.diag(sign) @ vt
    scale = float(np.sum(d * sign) / max(np.mean(np.sum(xs * xs, axis=1)), 1e-12))
    return scale, r, md - scale * (r @ ms)


camera_npz = np.load(CAMERA_FILE)
camera_by_target = {}
for tid in sorted(set(camera_npz["targets"].tolist())):
    indices = np.flatnonzero(camera_npz["targets"] == tid)
    indices = indices[np.argsort(camera_npz["angles"][indices])]
    camera_by_target[tid] = (
        camera_npz["extrinsics"][indices], camera_npz["intrinsics"][indices],
        camera_npz["angles"][indices]
    )

device = "cuda"
model = VGGT()
weights = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt", map_location="cpu"
)
model.load_state_dict(weights)
model.eval().to(device)
amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def reconstruct(image_dir, output_path, tid, branch):
    names = sorted(glob.glob(str(image_dir / "*")))
    if len(names) != 4:
        raise RuntimeError(f"Expected 4 images in {image_dir}, got {len(names)}")
    images = load_and_preprocess_images(names).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
        pred = model(images)
    pred_ext, pred_intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    pred["extrinsic"], pred["intrinsic"] = pred_ext, pred_intr
    for key in list(pred):
        if isinstance(pred[key], torch.Tensor):
            pred[key] = pred[key].detach().cpu().numpy().squeeze(0)
    pred["pose_enc_list"] = None
    pred["world_points_from_depth"] = unproject_depth_map_to_point_map(
        pred["depth"], pred["extrinsic"], pred["intrinsic"]
    )

    exact_ext44, exact_k512, angles = camera_by_target[tid]
    exact_ext = exact_ext44[:, :3, :4].astype(np.float32)
    pred_centers = centers_from_w2c(pred["extrinsic"])
    exact_centers = centers_from_w2c(exact_ext)
    scale, rotation, translation = umeyama(pred_centers, exact_centers)
    points = pred["world_points_from_depth"]
    pred["world_points_from_depth"] = (
        scale * (points.reshape(-1, 3) @ rotation.T) + translation
    ).reshape(points.shape).astype(np.float32)

    h, w = pred["depth"].shape[1:3]
    exact_k = exact_k512.copy()
    exact_k[:, 0, :] *= w / 512.0
    exact_k[:, 1, :] *= h / 512.0
    pred["extrinsic"] = exact_ext
    pred["intrinsic"] = exact_k.astype(np.float32)
    pred["camera_source"] = "blender_exact_after_similarity"

    scene = predictions_to_glb(
        pred, conf_thres=50.0, filter_by_frames="All", show_cam=True,
        mask_sky=False, mask_black_bg=False, mask_white_bg=False,
        target_dir=str(image_dir.parent), prediction_mode="Depthmap and Camera Branch",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(file_obj=str(output_path))
    meta = {
        "target": tid, "branch": branch, "angles": angles.tolist(),
        "camera_source": pred["camera_source"], "similarity_scale": scale,
        "rotation": rotation.tolist(), "translation": translation.tolist(),
        "predicted_camera_rms_before": float(np.sqrt(np.mean(np.sum((pred_centers-exact_centers)**2, axis=1)))),
        "exact_camera_rms_after": 0.0,
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    del pred, images, scene
    gc.collect()
    torch.cuda.empty_cache()
    return scale


for i in range(1, 16):
    tid = f"T{i:03d}"
    sg = reconstruct(ROOT / "generated_sideviews" / tid / "images",
                     ROOT / "vggt_generated_camconf" / f"{tid}_generated_camconf.glb", tid, "generated")
    st = reconstruct(ROOT / "inputs_3dtile" / tid / "images",
                     ROOT / "vggt_3dtile_camconf" / f"{tid}_3dtile_camconf.glb", tid, "3dtile")
    print(f"CAMCONF_PROGRESS {i}/15 {tid} scales={sg:.5f},{st:.5f}", flush=True)
