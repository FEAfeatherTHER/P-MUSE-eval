"""Read-only validation for the public P-MUSE test-set records."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = ("benchmark_paired", "benchmark_style", "benchmark_mixed")
TASKS = ("gen", "edit")
EDIT_VARIANTS = ("add", "delete", "pitch_shift", "velocity_scale", "timing")
FAMILIES = ("bass", "drum", "guitar", "piano")

SCHEMAS = {
    ("benchmark_paired", "gen"): (
        "record_id", "family", "audio_path", "midi_path",
        "prefix_start_sec", "prefix_end_sec", "target_start_sec",
        "target_end_sec", "target_ref_audio_path",
    ),
    ("benchmark_paired", "edit"): (
        "record_id", "family", "audio_path", "prefix", "target", "suffix",
        "edits", "target_ref_audio_path",
    ),
    ("benchmark_style", "gen"): (
        "record_id", "family", "prompt_audio_path", "target_midi_path",
        "target_duration", "target_ref_audio_path",
    ),
    ("benchmark_style", "edit"): (
        "record_id", "family", "prefix", "target", "suffix", "edits",
        "target_ref_audio_path",
    ),
    ("benchmark_mixed", "gen"): (
        "record_id", "family", "audio_path", "midi_path",
        "style_prompt_audio_path", "style_prompt_start_sec",
        "style_prompt_end_sec", "prefix_start_sec", "prefix_end_sec",
        "target_start_sec", "target_end_sec", "target_ref_audio_path",
    ),
    ("benchmark_mixed", "edit"): (
        "record_id", "family", "audio_path", "style_prompt_audio_path",
        "style_prompt_start_sec", "style_prompt_end_sec", "prefix", "target",
        "suffix", "edits", "target_ref_audio_path",
    ),
}

RECORD_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def checked_record_id(value: Any) -> str:
    if not isinstance(value, str) or RECORD_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "record_id must be a non-empty ASCII filename stem containing only "
            "letters, digits, dot, underscore, or hyphen"
        )
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSON value must be an object")
            records.append(value)
    return records


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Overwrite a JSONL file without exposing a partially written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_relative_file(root: Path, value: Any, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: path must be a non-empty string")
        return
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{label}: path must stay inside the benchmark: {value}")
        return
    if not (root / path).is_file():
        errors.append(f"{label}: missing file: {value}")


def _path_values(task: str, record: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    if task == "gen":
        for key in (
            "audio_path", "midi_path", "prompt_audio_path", "target_midi_path",
            "style_prompt_audio_path", "target_ref_audio_path",
        ):
            if key in record:
                yield key, record[key]
        return

    for key in ("audio_path", "style_prompt_audio_path", "target_ref_audio_path"):
        if key in record:
            yield key, record[key]
    for name in ("prefix", "target", "suffix"):
        segment = record.get(name)
        if isinstance(segment, dict):
            for key in ("audio_clip_path", "midi_clip_path"):
                if key in segment:
                    yield f"{name}.{key}", segment[key]
    edits = record.get("edits")
    if isinstance(edits, dict):
        for variant in EDIT_VARIANTS:
            if variant in edits:
                yield f"edits.{variant}", edits[variant]


def _validate_times(
    benchmark: str,
    task: str,
    record: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    pairs: list[tuple[float, float, str]] = []
    if task == "gen" and benchmark in {"benchmark_paired", "benchmark_mixed"}:
        pairs.extend([
            (float(record["prefix_start_sec"]), float(record["prefix_end_sec"]), "prefix"),
            (float(record["target_start_sec"]), float(record["target_end_sec"]), "target"),
        ])
    if benchmark == "benchmark_mixed":
        pairs.append((
            float(record["style_prompt_start_sec"]),
            float(record["style_prompt_end_sec"]),
            "style_prompt",
        ))
    if task == "edit" and benchmark in {"benchmark_paired", "benchmark_mixed"}:
        for name in ("prefix", "target", "suffix"):
            segment = record[name]
            pairs.append((float(segment["start_sec"]), float(segment["end_sec"]), name))
    for start, end, name in pairs:
        if start < 0 or end <= start:
            errors.append(f"{label}: invalid {name} interval [{start}, {end}]")
    if task == "edit" and benchmark == "benchmark_style":
        for name in ("prefix", "target", "suffix"):
            if float(record[name]["duration"]) <= 0:
                errors.append(f"{label}: {name}.duration must be positive")
    if task == "gen" and benchmark == "benchmark_style":
        if float(record["target_duration"]) <= 0:
            errors.append(f"{label}: target_duration must be positive")


def _validate_edit_shape(
    benchmark: str,
    record: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    if benchmark == "benchmark_style":
        expected_segments = {
            "prefix": {"audio_clip_path", "duration"},
            "target": {"midi_clip_path", "duration"},
            "suffix": {"audio_clip_path", "duration"},
        }
    else:
        expected_segments = {
            name: {"start_sec", "end_sec", "midi_clip_path"}
            for name in ("prefix", "target", "suffix")
        }
    for name, expected_keys in expected_segments.items():
        value = record[name]
        if not isinstance(value, dict):
            errors.append(f"{label}.{name}: must be an object")
            continue
        actual_keys = set(value)
        if actual_keys != expected_keys:
            errors.append(
                f"{label}.{name}: keys differ; missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
    edits = record["edits"]
    if not isinstance(edits, dict):
        errors.append(f"{label}.edits: must be an object")
    elif set(edits) != set(EDIT_VARIANTS):
        errors.append(f"{label}.edits: variants must be {EDIT_VARIANTS}")


def validate_testset(testset_root: Path, expected_rows: int = 400) -> dict[str, Any]:
    errors: list[str] = []
    summary: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        benchmark_summary: dict[str, Any] = {}
        for task in TASKS:
            path = testset_root / benchmark / f"benchmark_{task}.jsonl"
            records = read_jsonl(path)
            ids = [
                record.get("record_id")
                for record in records
                if isinstance(record.get("record_id"), str)
            ]
            if len(records) != expected_rows:
                errors.append(f"{path}: expected {expected_rows} rows, found {len(records)}")
            if len(ids) != len(set(ids)):
                errors.append(f"{path}: duplicate record_id")
            expected_keys = set(SCHEMAS[(benchmark, task)])
            family_counts: Counter[str] = Counter()
            for index, record in enumerate(records, start=1):
                label = f"{path}:{index}"
                actual_keys = set(record)
                if actual_keys != expected_keys:
                    errors.append(
                        f"{label}: keys differ; missing={sorted(expected_keys - actual_keys)}, "
                        f"extra={sorted(actual_keys - expected_keys)}"
                    )
                    continue
                try:
                    checked_record_id(record["record_id"])
                except ValueError as exc:
                    errors.append(f"{label}: {exc}")
                family = str(record["family"])
                family_counts[family] += 1
                if family not in FAMILIES:
                    errors.append(f"{label}: unknown family: {family}")
                if task == "edit":
                    _validate_edit_shape(benchmark, record, label, errors)
                for field, value in _path_values(task, record):
                    _validate_relative_file(
                        testset_root / benchmark, value, f"{label}.{field}", errors
                    )
                try:
                    _validate_times(benchmark, task, record, label, errors)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{label}: invalid time or duration value: {exc}")
            benchmark_summary[task] = {
                "rows": len(records),
                "families": dict(sorted(family_counts.items())),
            }
        summary[benchmark] = benchmark_summary
    if errors:
        raise ValueError("test-set validation failed:\n" + "\n".join(errors[:100]))
    return summary


def validate_main() -> None:
    parser = argparse.ArgumentParser(description="Validate P-MUSE test-set records and assets")
    parser.add_argument("testset_root", type=Path)
    parser.add_argument("--expected-rows", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(validate_testset(args.testset_root, args.expected_rows), indent=2))


if __name__ == "__main__":
    validate_main()
