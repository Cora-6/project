import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


PROJECT = Path("/home/viplab/liukejia/project/vggt")
HOUSTON = Path("/home/viplab/liukejia/project/houston")
IMAGE_DIR = HOUSTON / "inference"
CAMERA_FILE = HOUSTON / "Houston_T126_camera_parameters.npz"
OUTPUT = HOUSTON / "vggt_output"
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


def exact_unproject(depth, exact_ext, exact_k, scale):
    depth = np.asarray(depth)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    count, height, width = depth.shape
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    c2w = np.linalg.inv(to_ext4(exact_ext))
    result = np.empty((count, height, width, 3), dtype=np.float32)
    for index in range(count):
        z = depth[index].astype(np.float64) * scale
        x = (u - exact_k[index, 0, 2]) / exact_k[index, 0, 0] * z
        y = (v - exact_k[index, 1, 2]) / exact_k[index, 1, 1] * z
        camera_points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
        result[index] = (camera_points @ c2w[index].T)[..., :3].astype(np.float32)
    return result


camera = np.load(CAMERA_FILE)
order = np.argsort(camera["angles"])
exact_ext44 = camera["extrinsics"][order]
exact_ext = exact_ext44[:, :3, :4].astype(np.float32)
exact_k512 = camera["intrinsics"][order]
angles = camera["angles"][order]
image_names = [str(IMAGE_DIR / f"Houston_A{int(angle):03d}_generated.png") for angle in angles]
if not all(Path(name).exists() for name in image_names):
    raise RuntimeError(f"Missing generated views: {image_names}")

device = "cuda"
model = VGGT()
model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt", map_location="cpu"
))
model.eval().to(device)
amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
images = load_and_preprocess_images(image_names).to(device)
with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
    prediction = model(images)
pred_ext, pred_k = pose_encoding_to_extri_intri(prediction["pose_enc"], images.shape[-2:])
prediction["extrinsic"], prediction["intrinsic"] = pred_ext, pred_k
for key in list(prediction):
    if isinstance(prediction[key], torch.Tensor):
        prediction[key] = prediction[key].detach().cpu().numpy().squeeze(0)
prediction["pose_enc_list"] = None

pred_centers = centers(prediction["extrinsic"])
exact_centers = centers(exact_ext)
ratios = []
for i in range(4):
    for j in range(i + 1, 4):
        ratios.append(np.linalg.norm(exact_centers[i] - exact_centers[j]) /
                      max(np.linalg.norm(pred_centers[i] - pred_centers[j]), 1e-12))
scale = float(np.median(ratios))
height, width = prediction["depth"].shape[1:3]
exact_k = exact_k512.copy()
exact_k[:, 0, :] *= width / 512.0
exact_k[:, 1, :] *= height / 512.0
prediction["world_points_from_depth"] = exact_unproject(
    prediction["depth"], exact_ext, exact_k, scale
)
prediction["extrinsic"] = exact_ext
prediction["intrinsic"] = exact_k.astype(np.float32)
prediction["camera_source"] = "houston_blender_exact_reprojection"

OUTPUT.mkdir(parents=True, exist_ok=True)
raw_path = OUTPUT / "T126_generated_exact.glb"
scene = predictions_to_glb(
    prediction, conf_thres=50.0, filter_by_frames="All", show_cam=True,
    mask_sky=False, mask_black_bg=False, mask_white_bg=False,
    target_dir=str(IMAGE_DIR), prediction_mode="Depthmap and Camera Branch",
)
scene.export(file_obj=str(raw_path))

# Undo visual_util's first-camera display alignment, then pre-encode coordinates
# so Blender's glTF importer recovers the original Houston Blender world axes.
opengl = np.diag([1.0, -1.0, -1.0, 1.0])
align_y180 = np.diag([-1.0, 1.0, -1.0, 1.0])
vggt_alignment = np.linalg.inv(exact_ext44[0]) @ opengl @ align_y180
blender_to_gltf = np.array([
    [1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1],
], dtype=np.float64)
corrected = trimesh.load(raw_path, force="scene")
corrected.apply_transform(np.linalg.inv(vggt_alignment))
corrected.apply_transform(blender_to_gltf)
corrected_path = OUTPUT / "T126_generated_blenderworld.glb"
corrected.export(corrected_path)
(OUTPUT / "T126_generated_blenderworld.json").write_text(json.dumps({
    "target": "T126", "angles": angles.tolist(), "depth_scale": scale,
    "pairwise_baseline_scale_ratios": ratios,
    "camera_source": prediction["camera_source"],
    "target_coordinate_system": "Houston Blender world coordinates",
}, indent=2))
print(f"HOUSTON_VGGT_DONE scale={scale:.6f} output={corrected_path}", flush=True)
