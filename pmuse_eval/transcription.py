"""Instrument-constrained MuScriptor transcription and MIDI caching."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .cache import file_info
from .manifests import (
    BENCHMARKS,
    EDIT_VARIANTS,
    checked_record_id,
    read_jsonl,
    write_jsonl_atomic,
)
from .muscriptor_groups import instrument_group_for_record


DEFAULT_MUSCRIPTOR_MODEL = "large"
MUSCRIPTOR_MODEL_SIZES = ("small", "medium", "large")
MUSCRIPTOR_REPO_TEMPLATE = "MuScriptor/muscriptor-{size}"
TRANSCRIPTION_CONFIG = {
    "metric_version": 3,
    "backend": "muscriptor",
    "package_version": "0.3.0",
    "mono": True,
    "target_sample_rate": 16000,
    "segment_duration_sec": 5.0,
    "decoding": "greedy",
    "temperature": 1.0,
    "cfg_coef": 1.0,
    "batch_size": 1,
    "prelude_forcing": True,
    "detect_tempo": "best-effort",
}


@dataclass(frozen=True)
class TranscriptionJob:
    kind: str
    record_id: str
    family: str
    dataset_name: str
    instrument_name: str
    instrument_group: str
    audio_path: Path
    midi_path: Path
    variant: str | None = None


def transcription_config_snapshot(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a JSON-compatible configuration value for cache comparison."""
    payload = config if config is not None else TRANSCRIPTION_CONFIG
    snapshot = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    if not isinstance(snapshot, dict):
        raise ValueError("transcription configuration must be a JSON object")
    return snapshot


def collect_transcription_jobs(
    testset_root: Path,
    submission_root: Path,
    benchmark: str,
    task: str,
    generated_midi_root: Path,
) -> list[TranscriptionJob]:
    """Collect every expected job, including jobs whose WAV is missing."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if task not in ("gen", "edit"):
        raise ValueError(f"Unknown task: {task}")

    benchmark_root = Path(testset_root).resolve() / benchmark
    output_root = Path(generated_midi_root).resolve() / benchmark / task
    submission_dir = Path(submission_root).resolve() / benchmark / task
    records = read_jsonl(benchmark_root / f"benchmark_{task}.jsonl")
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for metadata in read_jsonl(benchmark_root / "metadata.jsonl"):
        metadata_id = checked_record_id(metadata["record_id"])
        if metadata_id in metadata_by_id:
            raise ValueError(f"Duplicate metadata record_id: {metadata_id}")
        metadata_by_id[metadata_id] = metadata
    jobs: list[TranscriptionJob] = []

    for record in records:
        record_id = checked_record_id(record["record_id"])
        family = str(record.get("family", "?"))
        try:
            metadata = metadata_by_id[record_id]
        except KeyError as exc:
            raise ValueError(f"Missing metadata for record_id: {record_id}") from exc
        if str(metadata.get("family")) != family:
            raise ValueError(f"Family mismatch in metadata for record_id: {record_id}")
        dataset_name = str(metadata.get("dataset_name", ""))
        instrument_name = str(metadata.get("instrument_name", ""))
        instrument_group = instrument_group_for_record(
            family, dataset_name, instrument_name,
        )
        if task == "gen":
            jobs.append(TranscriptionJob(
                kind="generated",
                record_id=record_id,
                family=family,
                dataset_name=dataset_name,
                instrument_name=instrument_name,
                instrument_group=instrument_group,
                audio_path=submission_dir / f"{record_id}.wav",
                midi_path=output_root / f"{record_id}.mid",
            ))
        else:
            for variant in EDIT_VARIANTS:
                jobs.append(TranscriptionJob(
                    kind="generated",
                    record_id=record_id,
                    family=family,
                    dataset_name=dataset_name,
                    instrument_name=instrument_name,
                    instrument_group=instrument_group,
                    variant=variant,
                    audio_path=submission_dir / f"{record_id}__{variant}.wav",
                    midi_path=output_root / f"{record_id}__{variant}.mid",
                ))
    return jobs


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {
        str(item["midi_path"]): item
        for item in read_jsonl(path)
        if item.get("midi_path")
    }


def _matches_cache(
    midi_path: Path,
    cached: dict[str, Any] | None,
    identity: dict[str, Any],
) -> bool:
    if cached is None or cached.get("status") != "ok" or not midi_path.is_file():
        return False
    try:
        return (
            all(cached.get(key) == value for key, value in identity.items())
            and cached.get("midi_file_info") == file_info(midi_path)
        )
    except OSError:
        return False


def resolve_device(device: str) -> str:
    """Resolve a requested device without importing torch until transcription is used."""
    requested = str(device or "auto").lower()
    if requested == "cpu":
        return "cpu"
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        return requested
    raise ValueError("device must be one of: auto, cpu, cuda")


def validate_device_name(device: str) -> str:
    requested = str(device or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return requested


def resolve_muscriptor_checkpoint(model_source: str) -> Path:
    """Resolve a MuScriptor size keyword or local checkpoint to one file."""
    source = str(model_source)
    if source in MUSCRIPTOR_MODEL_SIZES:
        from huggingface_hub import hf_hub_download

        repo_id = MUSCRIPTOR_REPO_TEMPLATE.format(size=source)
        hf_hub_download(repo_id=repo_id, filename="config.json")
        return Path(hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
        )).resolve()
    checkpoint = Path(source).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MuScriptor checkpoint not found: {checkpoint}")
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"MuScriptor config.json not found next to checkpoint: {config_path}"
        )
    return checkpoint


def load_muscriptor_model(checkpoint: Path, device: str = "auto"):
    """Load one resolved MuScriptor checkpoint on the requested device."""
    from muscriptor import TranscriptionModel

    return TranscriptionModel.load_model(
        Path(checkpoint),
        device=resolve_device(device),
    )


def transcribe_audio(
    model: Any,
    audio_path: Path,
    midi_path: Path,
    instrument_group: str,
) -> None:
    """Write one instrument-constrained MuScriptor transcription atomically."""
    midi_bytes = model.transcribe_to_midi(
        audio_path,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=1.0,
        instruments=[instrument_group],
        batch_size=1,
        no_eos_is_ok=True,
        beam_size=1,
        prelude_forcing=True,
        detect_tempo="best-effort",
    )
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_midi = midi_path.with_name(f".{midi_path.name}.tmp")
    with temporary_midi.open("wb") as handle:
        handle.write(midi_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_midi, midi_path)


def transcribe_benchmark(
    *,
    testset_root: Path,
    submission_root: Path,
    benchmark: str,
    task: str,
    generated_midi_root: Path,
    model_source: str = DEFAULT_MUSCRIPTOR_MODEL,
    device: str = "auto",
    checkpoint_resolver: Callable[[str], Path] = resolve_muscriptor_checkpoint,
    model_loader: Callable[[Path, str], Any] = load_muscriptor_model,
    transcriber: Callable[[Any, Path, Path, str], None] = transcribe_audio,
) -> list[dict[str, Any]]:
    """Transcribe submitted WAVs and atomically refresh the generated cache."""
    device = validate_device_name(device)
    checkpoint = Path(checkpoint_resolver(str(model_source))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MuScriptor checkpoint not found: {checkpoint}")
    checkpoint_info = file_info(checkpoint)
    model_config_path = checkpoint.parent / "config.json"
    model_config_info = (
        file_info(model_config_path) if model_config_path.is_file() else None
    )
    config_snapshot = transcription_config_snapshot()
    jobs = collect_transcription_jobs(
        testset_root, submission_root, benchmark, task, generated_midi_root,
    )

    output_root = Path(generated_midi_root) / benchmark / task
    cache_path = output_root / "cache_index.jsonl"
    cache = _read_cache(cache_path)
    statuses: list[dict[str, Any]] = []
    model = None
    model_load_attempted = False
    model_load_error: Exception | None = None

    for job in jobs:
        instrument_groups = [job.instrument_group]
        base = {
            "kind": job.kind,
            "record_id": job.record_id,
            "family": job.family,
            "dataset_name": job.dataset_name,
            "instrument_name": job.instrument_name,
            "variant": job.variant,
            "audio_path": str(job.audio_path),
            "midi_path": str(job.midi_path),
            "model_source": str(model_source),
            "checkpoint_path": str(checkpoint),
            "model_config_path": str(model_config_path),
            "instrument_groups": instrument_groups,
        }
        if not job.audio_path.is_file():
            statuses.append({**base, "status": "missing_audio"})
            continue
        try:
            audio_info = file_info(job.audio_path)
        except OSError as exc:
            statuses.append({**base, "status": "audio_read_error", "error": str(exc)})
            continue

        cache_key = str(job.midi_path)
        cached = cache.get(cache_key)
        identity = {
            **base,
            "audio_file_info": audio_info,
            "checkpoint_file_info": checkpoint_info,
            "model_config_file_info": model_config_info,
            "transcription_config": config_snapshot,
        }
        if _matches_cache(job.midi_path, cached, identity):
            statuses.append({**identity, "status": "cached"})
            continue

        try:
            if not model_load_attempted:
                model_load_attempted = True
                try:
                    model = model_loader(checkpoint, device)
                except Exception as exc:
                    model_load_error = exc
            if model_load_error is not None:
                raise model_load_error
            transcriber(model, job.audio_path, job.midi_path, job.instrument_group)
            if not job.midi_path.is_file():
                raise RuntimeError("transcriber returned without creating the MIDI file")
            cache_record = {
                **identity,
                "midi_file_info": file_info(job.midi_path),
                "status": "ok",
            }
            cache[cache_key] = cache_record
            statuses.append(cache_record)
        except Exception as exc:  # Per-sample failures are part of the score coverage.
            statuses.append({**identity, "status": "failed", "error": str(exc)})

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(
        cache_path, sorted(cache.values(), key=lambda item: item["midi_path"])
    )
    write_jsonl_atomic(output_root / "transcription_status.jsonl", statuses)
    return statuses


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Transcribe a P-MUSE submission with MuScriptor")
    parser.add_argument("--testset-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--task", choices=("gen", "edit"), required=True)
    parser.add_argument("--generated-midi-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MUSCRIPTOR_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    statuses = transcribe_benchmark(
        testset_root=args.testset_root,
        submission_root=args.submission_root,
        benchmark=args.benchmark,
        task=args.task,
        generated_midi_root=args.generated_midi_dir,
        model_source=args.model,
        device=args.device,
    )
    print(json.dumps({
        "benchmark": args.benchmark,
        "task": args.task,
        "model": args.model,
        "expected_jobs": len(statuses),
        "status_counts": dict(sorted(Counter(
            item["status"] for item in statuses
        ).items())),
    }, indent=2))


if __name__ == "__main__":
    main()
