import sys
from pathlib import Path
import trimesh

ROOT = Path("/data/muufl_grid_50m_pipeline")


def merge(source, output):
    merged = trimesh.Scene()
    files = sorted(source.glob("*.glb"))
    for path in files:
        scene = trimesh.load(path, force="scene")
        for node in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node]
            geom = scene.geometry[geom_name].copy()
            geom.apply_transform(transform)
            merged.add_geometry(geom, node_name=f"{path.stem}_{node}",
                                geom_name=f"{path.stem}_{geom_name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.export(output)
    print(f"MERGED {len(files)} files -> {output}")


pairs = [
    ("vggt_generated_camconf", "merged_generated_camconf.glb"),
    ("vggt_3dtile_camconf", "merged_3dtile_camconf.glb"),
    ("vggt_generated_exact_reproject", "merged_generated_exact_reproject.glb"),
    ("vggt_3dtile_exact_reproject", "merged_3dtile_exact_reproject.glb"),
]
which = set(sys.argv[1:])
for directory, filename in pairs:
    if which and directory not in which:
        continue
    merge(ROOT / directory, ROOT / "merged_glbs" / filename)
