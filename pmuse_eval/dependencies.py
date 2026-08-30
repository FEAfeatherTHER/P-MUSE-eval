"""Path checks for externally downloaded metric repositories."""

from __future__ import annotations

from pathlib import Path
from typing import Any


INSTRUMENT_CONFIG_REL = Path("config/resnet34_sinc_nsynth.py")
INSTRUMENT_CHECKPOINT_REL = Path("log/nsynth_sinc_midi_aug_l8_asm_s100/8_77.h5")


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
