"""Compatibility wrapper for the shared multi-view implementation."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mv_model_common.dataset import HSIControlNetDataset as _SharedDataset


class HSIControlNetDataset(_SharedDataset):
    """Backward-compatible RGB-conditioning dataset."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("conditioning_mode", "rgb")
        super().__init__(*args, **kwargs)
