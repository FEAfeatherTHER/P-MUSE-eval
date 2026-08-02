"""Fixed P-MUSE submission layout and read-only validation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .manifests import checked_record_id


BENCHMARKS = ("benchmark_paired", "benchmark_style", "benchmark_mixed")
TASKS = ("gen", "edit")
EDIT_VARIANTS = ("add", "delete", "pitch_shift", "velocity_scale", "timing")


@dataclass(frozen=True)
class ExpectedSample:
    benchmark: str
    task: str
    record_id: str
    family: str
    variant: str | None
    benchmark_root: Path
    generated_audio_path: Path
    record: dict[str, Any]

    @property
    def sample_id(self) -> str:
        suffix = self.variant or "gen"
        return f"{self.benchmark}/{self.task}/{self.record_id}__{suffix}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def generated_filename(record_id: str, task: str, variant: str | None = None) -> str:
    record_id = checked_record_id(record_id)
    if task == "gen":
        if variant is not None:
            raise ValueError("generation samples do not have an edit variant")
        return f"{record_id}.wav"
    if task == "edit":
        if variant not in EDIT_VARIANTS:
            raise ValueError(f"unknown edit variant: {variant}")
        return f"{record_id}__{variant}.wav"
    raise ValueError(f"unknown task: {task}")


def iter_expected_samples(
    testset_root: Path,
    submission_root: Path,
    benchmark: str,
    task: str,
) -> Iterator[ExpectedSample]:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {benchmark}")
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")

    benchmark_root = Path(testset_root).resolve() / benchmark
    record_file = benchmark_root / f"benchmark_{task}.jsonl"
    output_dir = Path(submission_root).resolve() / benchmark / task
    for record in read_jsonl(record_file):
        record_id = checked_record_id(record["record_id"])
        family = str(record["family"])
        variants: tuple[str | None, ...] = (None,) if task == "gen" else EDIT_VARIANTS
        for variant in variants:
            yield ExpectedSample(
                benchmark=benchmark,
                task=task,
                record_id=record_id,
                family=family,
                variant=variant,
                benchmark_root=benchmark_root,
                generated_audio_path=output_dir / generated_filename(record_id, task, variant),
                record=record,
            )


def _audio_status(path: Path, check_audio: bool) -> tuple[str, str | None]:
    if not path.is_file():
        return "missing", None
    if not check_audio:
        return "present", None
    try:
        try:
            import soundfile as sf

            info = sf.info(str(path))
            frames, sample_rate = info.frames, info.samplerate
        except ImportError:
            import wave

            with wave.open(str(path), "rb") as handle:
                frames, sample_rate = handle.getnframes(), handle.getframerate()
        if frames <= 0 or sample_rate <= 0:
            return "invalid", "empty audio"
    except Exception as exc:
        return "invalid", str(exc)
    return "present", None


def validate_submission(
    testset_root: Path,
    submission_root: Path,
    benchmark: str,
    task: str,
    check_audio: bool = True,
) -> dict[str, Any]:
    expected = list(iter_expected_samples(testset_root, submission_root, benchmark, task))
    rows = []
    status_counts: Counter[str] = Counter()
    expected_paths = {sample.generated_audio_path.resolve() for sample in expected}
    for sample in expected:
        status, error = _audio_status(sample.generated_audio_path, check_audio)
        status_counts[status] += 1
        rows.append({
            "sample_id": sample.sample_id,
            "record_id": sample.record_id,
            "family": sample.family,
            "variant": sample.variant,
            "generated_audio_path": str(sample.generated_audio_path),
            "status": status,
            "error": error,
        })

    output_dir = Path(submission_root).resolve() / benchmark / task
    actual_paths = {path.resolve() for path in output_dir.glob("*.wav")} if output_dir.is_dir() else set()
    unexpected = sorted(str(path) for path in actual_paths - expected_paths)
    return {
        "benchmark": benchmark,
        "task": task,
        "expected": len(expected),
        "present": status_counts["present"],
        "missing": status_counts["missing"],
        "invalid": status_counts["invalid"],
        "unexpected": unexpected,
        "samples": rows,
    }
