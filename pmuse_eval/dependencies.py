"""Path checks for externally downloaded metric repositories."""

from __future__ import annotations

from pathlib import Path
from typing import Any


INSTRUMENT_CONFIG_REL = Path("config/resnet34_sinc_nsynth.py")
INSTRUMENT_CHECKPOINT_REL = Path("log/nsynth_sinc_midi_aug_l8_asm_s100/8_77.h5")
YOURMT3_CHECKPOINT_REL = Path(
    "amt/logs/2024/ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100/"
    "checkpoints/model.ckpt"
)


def _required_directory(path: Path, name: str) -> Path:
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _required_file(path: Path, name: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def check_instrument_paths(root: Path) -> dict[str, Any]:
    root = _required_directory(root, "instrument embedding repo")
    checkpoint = _required_file(
        root / INSTRUMENT_CHECKPOINT_REL,
        "instrument checkpoint",
    )
    config_file = _required_file(
        root / INSTRUMENT_CONFIG_REL,
        "instrument config",
    )
    _required_directory(root / "musyn", "instrument embedding Python package")
    return {
        "root": str(root),
        "checkpoint": str(checkpoint),
        "config_file": str(config_file),
    }


def check_yourmt3_paths(root: Path) -> dict[str, Any]:
    root = _required_directory(root, "YourMT3 repo")
    checkpoint = _required_file(
        root / YOURMT3_CHECKPOINT_REL,
        "YourMT3 checkpoint",
    )
    _required_file(root / "model_helper.py", "YourMT3 model_helper.py")
    _required_directory(root / "amt" / "src", "YourMT3 Python source")
    return {
        "root": str(root),
        "checkpoint": str(checkpoint),
    }
