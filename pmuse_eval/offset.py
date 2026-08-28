"""Pitch-aware onset-and-offset F1 for P-MUSE benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from .manifests import BENCHMARKS, EDIT_VARIANTS, checked_record_id, read_jsonl, write_jsonl_atomic
from .note_metrics import (
    DEFAULT_OFFSET_MIN_TOLERANCE,
    DEFAULT_OFFSET_RATIO,
    DEFAULT_ONSET_TOLERANCE,
    MidiNotes,
    load_midi_notes,
    score_note_offsets,
)
from .onset import (
    REGION_POLICY,
    _atomic_json,
    _group_summary,
    _mean_scores,
    _score_pair,
    _transcription_cache,
    _transcription_evidence_status,
    _transcription_statuses,
    edit_target_region,
    generation_reference,
)


def evaluate_offset(
    *,
    testset_root: Path,
    benchmark: str,
    task: str,
    generated_midi_root: Path,
    results_dir: Path,
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
    match_pitch: bool = True,
    verify_transcription_cache: bool = True,
    interval_loader: Callable[[Path, float, float, float], MidiNotes] = load_midi_notes,
    interval_scorer: Callable[..., dict[str, float]] = score_note_offsets,
) -> dict[str, Any]:
    """Score all expected records with the YourMT3 onset-and-offset definition."""
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
            item: dict[str, Any] = {
                "record_id": record_id,
                "family": family,
                "variant": variant,
                "reference_midi_path": str(ref_path),
                "generated_midi_path": str(gen_path),
            }
            if task == "edit":
                item["target_start_sec"] = gen_window[0]
                item["target_end_sec"] = gen_window[1]

            if family == "drum":
                item["status"] = "not_applicable"
                samples.append(item)
                continue

            transcription_error = None
            if verify_transcription_cache:
                evidence = _transcription_evidence_status(
                    record_id, variant, gen_path, generated_statuses, generated_cache,
                )
                if evidence is not None:
                    transcription_error = f"generated_{evidence}"
            if transcription_error is None:
                status, score, error, _ = _score_pair(
                    ref_path,
                    gen_path,
                    ref_window,
                    gen_window,
                    onset_tolerance,
                    match_pitch,
                    interval_loader,
                    interval_scorer,
                )
            else:
                status, score, error = transcription_error, None, None
            item["status"] = status
            if score is not None:
                item.update(score)
            if error:
                item["error"] = error
            samples.append(item)

    applicable_samples = [
        item for item in samples if item["status"] != "not_applicable"
    ]
    not_applicable = [
        item for item in samples if item["status"] == "not_applicable"
    ]
    summary = {
        "benchmark": benchmark,
        "task": task,
        "metric": "offset_f1",
        "match_pitch": bool(match_pitch),
        "onset_tolerance_sec": float(onset_tolerance),
        "offset_tolerance_ratio": DEFAULT_OFFSET_RATIO,
        "offset_min_tolerance_sec": DEFAULT_OFFSET_MIN_TOLERANCE,
        "region_policy": "full_audio" if task == "gen" else REGION_POLICY,
        "reference_policy": (
            "benchmark_target_midi" if task == "gen" else "edited_target_midi"
        ),
        "overall": _mean_scores(applicable_samples),
        "per_family": _group_summary(applicable_samples, "family"),
        "per_variant": (
            _group_summary(applicable_samples, "variant") if task == "edit" else {}
        ),
        "not_applicable": {
            "count": len(not_applicable),
            "families": {"drum": len(not_applicable)},
            "excluded_from": ["overall", "per_family", "per_variant"],
        },
        "status_counts": dict(sorted(
            (status, sum(item["status"] == status for item in samples))
            for status in {item["status"] for item in samples}
        )),
    }
    output_dir = Path(results_dir) / "offset"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "samples.jsonl", samples)
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate P-MUSE offset F1")
    parser.add_argument("--testset-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--task", choices=("gen", "edit"), required=True)
    parser.add_argument("--generated-midi-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--match-pitch", dest="match_pitch", action="store_true")
    parser.add_argument("--no-match-pitch", dest="match_pitch", action="store_false")
    parser.set_defaults(match_pitch=True)
    args = parser.parse_args(argv)
    summary = evaluate_offset(
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
