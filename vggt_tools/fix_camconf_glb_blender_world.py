#!/usr/bin/env python3
from pathlib import Path
import json

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path("/data/muufl_grid_50m_pipeline")
cam = np.load(ROOT / "grid_camera_parameters.npz")

# glTF is Y-up. Blender's glTF importer maps (x,y,z)_gltf -> (x,-z,y)_blender.
# Pre-encode Blender coordinates so importing the GLB recovers the original values.
blender_to_gltf = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)
opengl = np.diag([1.0, -1.0, -1.0, 1.0])
align_y180 = np.eye(4)
align_y180[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

for branch in ("generated", "3dtile"):
    src_dir = ROOT / f"vggt_{branch}_camconf"
    dst_dir = ROOT / f"vggt_{branch}_camconf_blenderworld"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 16):
        tid = f"T{index:03d}"
        source = src_dir / f"{tid}_{branch}_camconf.glb"
        target = dst_dir / f"{tid}_{branch}_camconf_blenderworld.glb"
        ids = np.flatnonzero(cam["targets"] == tid)
        ids = ids[np.argsort(cam["angles"][ids])]
        ext0 = cam["extrinsics"][ids[0]]
        vggt_alignment = np.linalg.inv(ext0) @ opengl @ align_y180

        scene = trimesh.load(source, force="scene")
        scene.apply_transform(np.linalg.inv(vggt_alignment))
        scene.apply_transform(blender_to_gltf)
        scene.export(target)
        target.with_suffix(".json").write_text(json.dumps({
            "source": str(source),
            "removed_vggt_first_camera_alignment": True,
            "preencoded_for_blender_gltf_import": True,
            "target_coordinate_system": "muufl_grid_50m_15targets.blend world coordinates",
        }, indent=2))
        print(f"FIXED {branch} {tid} -> {target}", flush=True)
