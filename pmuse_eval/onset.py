"""Pitch-aware, 50 ms onset F1 for P-MUSE benchmarks."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .cache import sha256_file
from .manifests import (
    BENCHMARKS,
    EDIT_VARIANTS,
    checked_record_id,
    read_jsonl,
    write_jsonl_atomic,
)
from .note_metrics import (
    DEFAULT_ONSET_TOLERANCE,
    MidiNotes,
    load_midi_notes,
    score_note_onsets,
)


REGION_POLICY = "edited_target_v1"


@dataclass(frozen=True)
class MidiIntervals:
    """Legacy pitch-agnostic MIDI interval result."""

    status: str
    intervals: tuple[tuple[float, float], ...] = ()
    error: str | None = None


def load_midi_intervals(
    midi_path: Path,
    start_sec: float = 0.0,
    end_sec: float = math.inf,
    shift_sec: float = 0.0,
) -> MidiIntervals:
    """Load pitch-agnostic intervals with the original public contract."""
    notes = load_midi_notes(midi_path, start_sec, end_sec, shift_sec)
    return MidiIntervals(notes.status, notes.intervals, notes.error)


def score_interval_onsets(
    reference: Sequence[tuple[float, float]],
    estimated: Sequence[tuple[float, float]],
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
) -> dict[str, float]:
    """Score pitch-agnostic intervals with the original public signature."""
    return score_note_onsets(
        reference,
        [0] * len(reference),
        estimated,
        [0] * len(estimated),
        onset_tolerance,
        match_pitch=False,
    )


def _supports_pitched_scorer(interval_scorer: Callable[..., dict[str, float]]) -> bool:
    """Return whether a scorer accepts the six-argument pitched interface."""
    if interval_scorer is score_note_onsets:
        return True
    try:
        inspect.signature(interval_scorer).bind(
            (), (), (), (), DEFAULT_ONSET_TOLERANCE, True,
        )
    except TypeError:
        return False
    except ValueError:
        return True
    return True


def segment_duration(segment: dict[str, Any]) -> float:
    if "duration" in segment:
        return float(segment["duration"])
    return float(segment["end_sec"]) - float(segment.get("start_sec", 0.0))


def edit_target_region(record: dict[str, Any]) -> tuple[float, float]:
    target_start = segment_duration(record["prefix"])
    return target_start, target_start + segment_duration(record["target"])


def generation_reference(
    record: dict[str, Any], benchmark: str, benchmark_root: Path,
) -> tuple[Path, tuple[float, float, float]]:
    if benchmark == "benchmark_style":
        return (
            benchmark_root / record["target_midi_path"],
            (0.0, float(record["target_duration"]), 0.0),
        )
    target_start = float(record["target_start_sec"])
    target_end = float(record["target_end_sec"])
    return (
        benchmark_root / record["midi_path"],
        (target_start, target_end, -target_start),
    )


def _score_pair(
    ref_path: Path,
    gen_path: Path,
    ref_window: tuple[float, float, float],
    gen_window: tuple[float, float, float],
    onset_tolerance: float,
    match_pitch: bool,
    interval_loader: Callable[[Path, float, float, float], MidiNotes | MidiIntervals],
    interval_scorer: Callable[..., dict[str, float]],
) -> tuple[str, dict[str, float] | None, str | None]:
    if not ref_path.is_file():
        return "missing_reference_midi", None, None
    if not gen_path.is_file():
        return "missing_generated_midi", None, None
    reference = interval_loader(ref_path, *ref_window)
    if reference.status != "ok":
        return f"reference_{reference.status}", None, reference.error
    generated = interval_loader(gen_path, *gen_window)
    if generated.status != "ok":
        return f"generated_{generated.status}", None, generated.error
    try:
        legacy_loader = isinstance(reference, MidiIntervals) or isinstance(
            generated, MidiIntervals,
        )
        if legacy_loader and _supports_pitched_scorer(interval_scorer):
            return "ok", score_interval_onsets(
                reference.intervals, generated.intervals, onset_tolerance,
            ), None
        if legacy_loader or not _supports_pitched_scorer(interval_scorer):
            return "ok", interval_scorer(
                reference.intervals, generated.intervals, onset_tolerance,
            ), None
        return "ok", interval_scorer(
            reference.intervals,
            reference.pitches,
            generated.intervals,
            generated.pitches,
            onset_tolerance,
            match_pitch,
        ), None
    except Exception as exc:
        return "score_error", None, str(exc)


def _mean_scores(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    sample_list = list(samples)
    valid = [item for item in sample_list if item["status"] == "ok"]
    return {
        "expected": len(sample_list),
        "scored": len(valid),
        "skipped": len(sample_list) - len(valid),
        "mean_precision": (
            sum(item["precision"] for item in valid) / len(valid) if valid else None
        ),
        "mean_recall": (
            sum(item["recall"] for item in valid) / len(valid) if valid else None
        ),
        "mean_f1": sum(item["f1"] for item in valid) / len(valid) if valid else None,
    }


def _group_summary(samples: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        value = sample.get(key)
        if value is not None:
            groups[str(value)].append(sample)
    return {name: _mean_scores(group) for name, group in sorted(groups.items())}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _transcription_statuses(path: Path) -> dict[tuple[str, str | None], str] | None:
    if not path.is_file():
        return None
    return {
        (str(item["record_id"]), item.get("variant")): str(item["status"])
        for item in read_jsonl(path)
    }


def _transcription_cache(path: Path) -> dict[str, dict[str, Any]] | None:
    if not path.is_file():
        return None
    return {
        str(item["midi_path"]): item
        for item in read_jsonl(path)
        if item.get("midi_path")
    }


def _transcription_evidence_status(
    record_id: str,
    variant: str | None,
    midi_path: Path,
    statuses: dict[tuple[str, str | None], str] | None,
    cache: dict[str, dict[str, Any]] | None,
) -> str | None:
    if statuses is None:
        return "missing_transcription_status"
    transcription_status = statuses.get((record_id, variant), "missing_status_entry")
    if transcription_status not in {"ok", "cached"}:
        return f"transcription_{transcription_status}"
    if cache is None:
        return "missing_transcription_cache"
    entry = cache.get(str(midi_path))
    if entry is None:
        return "missing_transcription_cache_entry"
    if not midi_path.is_file():
        return "missing_transcribed_midi"
    try:
        if entry.get("midi_sha256") != sha256_file(midi_path):
            return "changed_transcribed_midi"
        audio_path = Path(str(entry["audio_path"]))
        if not audio_path.is_file():
            return "missing_transcription_audio"
        if entry.get("audio_sha256") != sha256_file(audio_path):
            return "changed_transcription_audio"
    except (KeyError, OSError):
        return "invalid_transcription_cache_entry"
    return None


def evaluate_onset(
    *,
    testset_root: Path,
    benchmark: str,
    task: str,
    generated_midi_root: Path,
    results_dir: Path,
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
    match_pitch: bool = True,
    verify_transcription_cache: bool = True,
    interval_loader: Callable[[Path, float, float, float], MidiNotes | MidiIntervals] = load_midi_notes,
    interval_scorer: Callable[..., dict[str, float]] = score_note_onsets,
) -> dict[str, Any]:
    """Score all expected records and overwrite per-sample and summary outputs."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if task not in ("gen", "edit"):
        raise ValueError(f"Unknown task: {task}")

    benchmark_root = Path(testset_root).resolve() / benchmark
    generated_root = Path(generated_midi_root).resolve() / benchmark / task
    records = read_jsonl(benchmark_root / f"benchmark_{task}.jsonl")
    samples: list[dict[str, Any]] = []
    generated_statuses = _transcription_statuses(
        generated_root / "transcription_status.jsonl"
    )
    generated_cache = _transcription_cache(generated_root / "cache_index.jsonl")

    for record in records:
        record_id = checked_record_id(record["record_id"])
        family = str(record.get("family", "?"))
        if task == "gen":
            reference_path, reference_window = generation_reference(
                record, benchmark, benchmark_root,
            )
            entries = ((
                None,
                reference_path,
                generated_root / f"{record_id}.mid",
                reference_window,
                (0.0, math.inf, 0.0),
            ),)
        else:
            target_start, target_end = edit_target_region(record)
            entries = tuple(
                (
                    variant,
                    benchmark_root / record["edits"][variant],
                    generated_root / f"{record_id}__{variant}.mid",
                    (0.0, target_end - target_start, 0.0),
                    (target_start, target_end, -target_start),
                )
                for variant in EDIT_VARIANTS
            )

        for variant, ref_path, gen_path, ref_window, gen_window in entries:
            transcription_error = None
            if verify_transcription_cache:
                evidence = _transcription_evidence_status(
                    record_id, variant, gen_path, generated_statuses, generated_cache,
                )
                if evidence is not None:
                    transcription_error = f"generated_{evidence}"
            if transcription_error is None:
                status, score, error = _score_pair(
                    ref_path, gen_path, ref_window, gen_window, onset_tolerance, match_pitch,
                    interval_loader, interval_scorer,
                )
            else:
                status, score, error = transcription_error, None, None
            item: dict[str, Any] = {
                "record_id": record_id,
                "family": family,
                "variant": variant,
                "reference_midi_path": str(ref_path),
                "generated_midi_path": str(gen_path),
                "status": status,
            }
            if task == "edit":
                item["target_start_sec"] = gen_window[0]
                item["target_end_sec"] = gen_window[1]
            if score is not None:
                item.update(score)
            if error:
                item["error"] = error
            samples.append(item)

    summary = {
        "benchmark": benchmark,
        "task": task,
        "metric": "onset_f1",
        "match_pitch": bool(match_pitch),
        "onset_tolerance_sec": float(onset_tolerance),
        "region_policy": "full_audio" if task == "gen" else REGION_POLICY,
        "reference_policy": (
            "benchmark_target_midi" if task == "gen" else "edited_target_midi"
        ),
        "overall": _mean_scores(samples),
        "per_family": _group_summary(samples, "family"),
        "per_variant": _group_summary(samples, "variant") if task == "edit" else {},
        "status_counts": dict(sorted(
            (status, sum(item["status"] == status for item in samples))
            for status in {item["status"] for item in samples}
        )),
    }
    output_dir = Path(results_dir) / "onset"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "samples.jsonl", samples)
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate P-MUSE onset F1")
    parser.add_argument("--testset-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--task", choices=("gen", "edit"), required=True)
    parser.add_argument("--generated-midi-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--match-pitch", dest="match_pitch", action="store_true")
    parser.add_argument("--no-match-pitch", dest="match_pitch", action="store_false")
    parser.set_defaults(match_pitch=True)
    args = parser.parse_args(argv)
    summary = evaluate_onset(
        testset_root=args.testset_root,
        benchmark=args.benchmark,
        task=args.task,
        generated_midi_root=args.generated_midi_dir,
        results_dir=args.results_dir / args.benchmark / args.task,
        match_pitch=args.match_pitch,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
