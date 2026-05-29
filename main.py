import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class PipelineConfig:
    project_root: Path

    # 1) 推理生图
    test_grid_notebook: Path
    test_grid_overrides: Dict[str, Any]

    # 2) 重建（模型输出图 -> 点云）
    reconstruct_notebook: Path
    reconstruct_pred_overrides: Dict[str, Any]

    # 3) 先拼接 render_3d，再重建（render_3d -> 点云）
    render3d_dir: Path
    stitched_render3d_dir: Path
    reconstruct_gt_overrides: Dict[str, Any]

    executed_notebook_dir: Path


def _py_literal(v: Any) -> str:
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, (int, float, bool)) or v is None:
        return repr(v)
    if isinstance(v, (list, tuple, dict)):
        return repr(v)
    raise TypeError(f"unsupported override type: {type(v)}")


def patch_notebook_assignments(nb_path: Path, overrides: Dict[str, Any], out_path: Path) -> Path:
    try:
        import nbformat
    except Exception as e:
        raise RuntimeError("需要 nbformat，请先安装: pip install nbformat") from e

    nb = nbformat.read(str(nb_path), as_version=4)
    keys = list(overrides.keys())
    patterns = {
        k: re.compile(rf"^(\s*){re.escape(k)}\s*=.*$", flags=re.MULTILINE) for k in keys
    }

    patched_any = {k: False for k in keys}

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        for k in keys:
            pat = patterns[k]
            repl = rf"\1{k} = {_py_literal(overrides[k])}"
            new_src, n = pat.subn(repl, src)
            if n > 0:
                src = new_src
                patched_any[k] = True
        cell["source"] = src

    # 对于未命中的变量，追加到第一个 code cell 末尾
    for k in keys:
        if not patched_any[k]:
            for cell in nb.cells:
                if cell.get("cell_type") == "code":
                    cell["source"] = (cell.get("source", "") + f"\n\n{k} = {_py_literal(overrides[k])}\n")
                    patched_any[k] = True
                    break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(out_path))
    return out_path


def execute_notebook(nb_path: Path, cwd: Path, timeout: int = -1) -> None:
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except Exception as e:
        raise RuntimeError("需要 nbconvert/nbformat，请先安装: pip install nbconvert nbformat") from e

    nb = nbformat.read(str(nb_path), as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(cwd)}})
    nbformat.write(nb, str(nb_path))


def _parse_num_tokens(stem: str):
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    try:
        nums = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]
    except ValueError:
        return None
    return f"{nums[0]}_{nums[1]}_{nums[2]}", nums[3]


def stitch_render3d_to_2x2(
    render3d_dir: Path,
    out_dir: Path,
    target_angles: Tuple[int, int, int, int] = (0, 90, 180, 270),
    size: int = 512,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(sorted(render3d_dir.glob(ext)))

    groups: Dict[str, Dict[int, Path]] = {}
    for p in files:
        parsed = _parse_num_tokens(p.stem)
        if parsed is None:
            continue
        num123, a = parsed
        if a in target_angles and num123 not in groups:
            groups[num123] = {}
        if a in target_angles:
            groups[num123][a] = p

    written: List[Path] = []
    idx = 1
    sub = size // 2
    for num123, d in sorted(groups.items(), key=lambda x: tuple(int(t) for t in x[0].split("_"))):
        if not all(a in d for a in target_angles):
            continue
        imgs = []
        ok = True
        for a in target_angles:
            bgr = cv2.imread(str(d[a]), cv2.IMREAD_COLOR)
            if bgr is None:
                ok = False
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (sub, sub), interpolation=cv2.INTER_AREA)
            imgs.append(rgb)
        if not ok:
            continue

        top = np.concatenate([imgs[0], imgs[1]], axis=1)
        bottom = np.concatenate([imgs[2], imgs[3]], axis=1)
        grid = np.concatenate([top, bottom], axis=0)

        out_name = f"num123_{num123}_{idx:04d}.png"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        written.append(out_path)
        idx += 1

    return written


def run_full_test_pipeline(cfg: PipelineConfig) -> None:
    cfg.executed_notebook_dir.mkdir(parents=True, exist_ok=True)

    # A) test_grid 推理生图
    test_nb_exec = cfg.executed_notebook_dir / "test_grid.executed.ipynb"
    patch_notebook_assignments(cfg.test_grid_notebook, cfg.test_grid_overrides, test_nb_exec)
    print(f"[1/4] 执行推理 notebook: {test_nb_exec}")
    execute_notebook(test_nb_exec, cwd=cfg.project_root)

    # B) 重建模型输出图点云
    recon_pred_exec = cfg.executed_notebook_dir / "reconstruct_pred.executed.ipynb"
    patch_notebook_assignments(cfg.reconstruct_notebook, cfg.reconstruct_pred_overrides, recon_pred_exec)
    print(f"[2/4] 执行重建(模型输出图): {recon_pred_exec}")
    execute_notebook(recon_pred_exec, cwd=cfg.project_root)

    # C) 拼接 render_3d
    print(f"[3/4] 拼接 render_3d -> 2x2: {cfg.render3d_dir} -> {cfg.stitched_render3d_dir}")
    stitched = stitch_render3d_to_2x2(cfg.render3d_dir, cfg.stitched_render3d_dir)
    print(f"    已生成拼接图: {len(stitched)} 张")

    # D) 重建 render_3d 点云
    recon_gt_exec = cfg.executed_notebook_dir / "reconstruct_gt.executed.ipynb"
    patch_notebook_assignments(cfg.reconstruct_notebook, cfg.reconstruct_gt_overrides, recon_gt_exec)
    print(f"[4/4] 执行重建(render_3d 拼接图): {recon_gt_exec}")
    execute_notebook(recon_gt_exec, cwd=cfg.project_root)

    print("\n✅ 全流程完成")
    print(json.dumps(
        {
            "test_grid_notebook_executed": str(test_nb_exec),
            "reconstruct_pred_notebook_executed": str(recon_pred_exec),
            "stitched_render3d_dir": str(cfg.stitched_render3d_dir),
            "reconstruct_gt_notebook_executed": str(recon_gt_exec),
        },
        ensure_ascii=False,
        indent=2,
    ))


def build_default_config(project_root: Path) -> PipelineConfig:
    # 你可以按需改这些默认路径
    return PipelineConfig(
        project_root=project_root,
        test_grid_notebook=project_root / "mv_model" / "test_grid.ipynb",
        test_grid_overrides={
            "OUT_DIR": str(project_root / "mv_model" / "results_grid4"),
            # 如需覆盖可继续加："NPY_DIR": "...", "CAMCONF_DIR": "...", "CKPT_DIR": "..."
        },
        reconstruct_notebook=project_root / "reconstruct" / "single_512_to_glb.ipynb",
        reconstruct_pred_overrides={
            "input_path": str(project_root / "mv_model" / "results_grid4"),
            "output_path": str(project_root / "reconstruct" / "reconstruct_glb_grid4_pos"),
            "use_camconf_camera": True,
            "camconf_dir": "/data/test_grid4_from_test_grid4/camconf",
            "strict_camconf": True,
            "post_align_mode": "camera_fit",
            "save_npz": False,
        },
        render3d_dir=Path("/data/test_grid4_from_test_grid4/render_3d"),
        stitched_render3d_dir=project_root / "reconstruct" / "render_3d_stitched_grid4",
        reconstruct_gt_overrides={
            "input_path": str(project_root / "reconstruct" / "render_3d_stitched_grid4"),
            "output_path": str(project_root / "reconstruct" / "reconstruct_glb_render3d_grid4"),
            "use_camconf_camera": True,
            "camconf_dir": "/data/test_grid4_from_test_grid4/camconf",
            "strict_camconf": True,
            "post_align_mode": "camera_fit",
            "save_npz": False,
        },
        executed_notebook_dir=project_root / "runs" / "executed_notebooks",
    )


def main():
    parser = argparse.ArgumentParser(description="运行 test 全流程：推理生图 + 两路重建点云")
    parser.add_argument(
        "--project-root",
        type=str,
        default="/home/viplab/liukejia/project",
        help="项目根目录",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cfg = build_default_config(root)
    run_full_test_pipeline(cfg)


if __name__ == "__main__":
    main()
