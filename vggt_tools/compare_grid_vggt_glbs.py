#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


ROOT = Path("/data/muufl_grid_50m_pipeline")
OUT = ROOT / "vggt_comparison"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260901)


def pointcloud(path, n=60000):
    scene = trimesh.load(path, force="scene")
    geom = max(scene.geometry.values(), key=lambda g: len(g.vertices))
    pts = np.asarray(geom.vertices, dtype=np.float64)
    if len(pts) > n:
        pts = pts[RNG.choice(len(pts), n, replace=False)]
    return pts


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    r = u @ np.diag(sign) @ vt
    var = np.mean(np.sum(xs * xs, axis=1))
    scale = float(np.sum(d * sign) / max(var, 1e-12))
    t = md - scale * (r @ ms)
    return scale, r, t


def apply(pts, s, r, t):
    return s * (pts @ r.T) + t


def align_similarity(src, dst, iterations=15):
    # Same ordered views give a common orientation; initialize center and robust scale.
    cs, cd = np.median(src, 0), np.median(dst, 0)
    rs = np.median(np.linalg.norm(src - cs, axis=1))
    rd = np.median(np.linalg.norm(dst - cd, axis=1))
    s_total = rd / max(rs, 1e-12)
    r_total = np.eye(3)
    t_total = cd - s_total * cs
    tree = cKDTree(dst)
    for _ in range(iterations):
        moved = apply(src, s_total, r_total, t_total)
        dist, idx = tree.query(moved, workers=-1)
        keep = dist <= np.quantile(dist, 0.70)
        ds, dr, dt = umeyama(moved[keep], dst[idx[keep]])
        s_total = ds * s_total
        r_total = dr @ r_total
        t_total = ds * (dr @ t_total) + dt
    return s_total, r_total, t_total


def distances(a, b):
    return cKDTree(b).query(a, workers=-1)[0]


rows = []
for i in range(1, 16):
    tid = f"T{i:03d}"
    gen = pointcloud(ROOT / "vggt_generated" / f"{tid}_generated.glb")
    ref = pointcloud(ROOT / "vggt_3dtile" / f"{tid}_3dtile.glb")
    scale, rot, trans = align_similarity(gen, ref)
    aligned = apply(gen, scale, rot, trans)
    d_gr = distances(aligned, ref)
    d_rg = distances(ref, aligned)
    lo, hi = np.quantile(ref, [0.01, 0.99], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    nd_gr, nd_rg = d_gr / diag, d_rg / diag
    row = {
        "target": tid,
        "scale_generated_to_3dtile": scale,
        "robust_reference_diagonal": diag,
        "accuracy_mean_norm": float(nd_gr.mean()),
        "completeness_mean_norm": float(nd_rg.mean()),
        "chamfer_l1_norm": float((nd_gr.mean() + nd_rg.mean()) / 2),
        "chamfer_l2_norm": float((np.mean(nd_gr**2) + np.mean(nd_rg**2)) / 2),
        "f1_at_1pct": float(2 * np.mean(nd_gr < .01) * np.mean(nd_rg < .01) /
                            max(np.mean(nd_gr < .01) + np.mean(nd_rg < .01), 1e-12)),
        "f1_at_2pct": float(2 * np.mean(nd_gr < .02) * np.mean(nd_rg < .02) /
                            max(np.mean(nd_gr < .02) + np.mean(nd_rg < .02), 1e-12)),
    }
    rows.append(row)
    # Compact overlay: cyan reference, red generated reconstruction.
    take = min(30000, len(ref), len(aligned))
    rr = ref[RNG.choice(len(ref), take, replace=False)]
    gg = aligned[RNG.choice(len(aligned), take, replace=False)]
    vertices = np.vstack([rr, gg])
    colors = np.vstack([
        np.tile([0, 220, 255, 255], (take, 1)),
        np.tile([255, 70, 50, 255], (take, 1)),
    ])
    trimesh.PointCloud(vertices, colors=colors).export(OUT / f"{tid}_aligned_overlay.ply")
    np.savez_compressed(OUT / f"{tid}_similarity_transform.npz",
                        scale=scale, rotation=rot, translation=trans)
    print(f"COMPARE_PROGRESS {i}/15 {tid} chamfer={row['chamfer_l1_norm']:.6f}", flush=True)

with open(OUT / "metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

aggregate = {}
for key in rows[0]:
    if key == "target":
        continue
    values = np.array([r[key] for r in rows])
    aggregate[key] = {"mean": float(values.mean()), "median": float(np.median(values)),
                      "min": float(values.min()), "max": float(values.max())}
with open(OUT / "summary.json", "w") as f:
    json.dump(aggregate, f, indent=2)
print(json.dumps(aggregate, indent=2), flush=True)
