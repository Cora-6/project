# Multi-view HSI/RGB 3D reconstruction

This repository contains research code for generating multi-view side images from
hyperspectral or RGB inputs, reconstructing 3D geometry, and evaluating image and
point-cloud quality.

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Notebook-driven end-to-end inference and reconstruction pipeline |
| `mv_model/` | Main SDXL ControlNet model, training, and inference code |
| `mv_model_no_deformable/` | Ablation without the deformable component |
| `mv_model_no_hsi/` | Ablation without hyperspectral input |
| `depth_model/` | Depth model preparation and ControlNet training |
| `reconstruct/` | VGGT-based reconstruction notebooks and geometry utilities |
| `metric/` | Image, depth, and point-cloud evaluation notebooks |
| `houston/` | Houston-specific orchestration code |
| `run_*_inference.py` | Dataset-specific inference entry points |
| `full_pipeline*.ipynb` | Experimental end-to-end notebooks |

## Environment

The depth-model environment is recorded in
`depth_model/train_nd_with_controlnet/environment.yml`. The other pipelines also
require PyTorch, Diffusers, Transformers, NumPy, OpenCV, Jupyter, nbformat, and
nbconvert. CUDA and package versions should match the model checkpoints in use.

Training scripts expect the W&B credential to be supplied by the shell:

```bash
export WANDB_API_KEY="<your-key>"
```

Never commit credentials, model weights, datasets, or generated artifacts.

## Running

The default pipeline contains machine-specific dataset and checkpoint paths.
Review the paths before execution, then run:

```bash
python main.py --help
python main.py --project-root /path/to/project
```

Dataset-specific scripts currently keep their input, output, checkpoint, and
prompt configuration near the top of each file:

```bash
python run_grid_sideview_inference.py
python run_houston_sideview_inference.py
python run_muufl_sideview_pipeline.py
```

Most experiments are notebook-based. Start Jupyter from the repository root so
relative imports resolve consistently.

## Data and artifacts

Large datasets, checkpoints, NumPy arrays, meshes, generated images, W&B runs,
and reconstruction outputs are intentionally excluded from Git. Keep these in
external storage and document their paths locally.

## Notes

This is research code. Several entry points contain absolute paths inherited from
the original experiment environment; update them for a different machine. The
`backup/pre-cleanup-20261019` branch preserves the pre-cleanup code snapshot.

## Shared multi-view training architecture

The three experiment directories now use one implementation in
`mv_model_common/`:

- `conditioning_mode=hsi`, `controlnet_type=warp`: main model.
- `conditioning_mode=hsi`, `controlnet_type=standard`: no-deformable ablation.
- `conditioning_mode=rgb`, `controlnet_type=warp`: no-HSI ablation.

The original Python and shell entry-point paths remain as compatibility wrappers,
so existing commands and notebooks do not need to change. New shared behavior
should be implemented once in `mv_model_common/`; experiment directories should
only contain variant-specific notebooks and launcher defaults.
