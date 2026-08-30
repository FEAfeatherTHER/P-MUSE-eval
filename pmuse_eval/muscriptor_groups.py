"""MuScriptor group selection for the fixed P-MUSE benchmark instruments."""

from __future__ import annotations


def instrument_group_for_record(
    family: str,
    dataset_name: str,
    instrument_name: str,
) -> str:
    """Map benchmark metadata to one MuScriptor instrument group."""
    family_key = str(family).strip().lower()
    dataset_key = str(dataset_name).strip().lower()
    instrument_key = str(instrument_name).strip().lower().replace("-", "_")

    if family_key == "drum":
        return "drums"
    if family_key == "piano":
        if instrument_key.startswith("keyboard_electronic_"):
            return "electric_piano"
        if instrument_key.startswith("keyboard_acoustic_"):
            return "acoustic_piano"
        # MuScriptor groups GM program 7 (Clavinet) with acoustic piano.
        if dataset_key == "slakh" and instrument_key == "clavinet":
            return "acoustic_piano"
    elif family_key == "guitar":
        if "distort" in instrument_key:
            return "distorted_electric_guitar"
        if dataset_key == "synthtab" or instrument_key.startswith(
            "guitar_electronic_"
        ):
            return "clean_electric_guitar"
        if dataset_key == "guitarset" or instrument_key.startswith(
            "guitar_acoustic_"
        ):
            return "acoustic_guitar"
    elif family_key == "bass":
        # MuScriptor groups GM program 35 (Fretless Bass) with acoustic bass.
        if "acoustic" in instrument_key or instrument_key == "fretless bass":
            return "acoustic_bass"
        if (
            instrument_key.startswith("bass_electronic_")
            or dataset_key == "slakh"
        ):
            return "electric_bass"
    raise ValueError(
        "No MuScriptor group mapping for "
        f"family={family!r}, dataset={dataset_name!r}, instrument={instrument_name!r}"
    )
