"""Compatibility wrapper for the shared multi-view implementation."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mv_model_common.dataset import *  # noqa: F401,F403
