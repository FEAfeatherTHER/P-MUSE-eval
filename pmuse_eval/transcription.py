"""YourMT3 transcription and MIDI caching."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .cache import file_info
from .dependencies import YOURMT3_CHECKPOINT_REL, check_yourmt3_paths
from .manifests import (
    BENCHMARKS,
    EDIT_VARIANTS,
    checked_record_id,
    read_jsonl,
    write_jsonl_atomic,
)


YOURMT3_MODEL_NAME = "YPTF+Single (noPS)"
YOURMT3_EXPERIMENT = (
    "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
)
YOURMT3_MODEL_ARGS = (
    YOURMT3_EXPERIMENT,
    "-p", "2024",
    "-enc", "perceiver-tf",
    "-ac", "spec",
    "-hop", "300",
    "-atc", "1",
    "-pr", "16",
)
TRANSCRIPTION_CONFIG = {
    "metric_version": 1,
    "model": YOURMT3_MODEL_NAME,
    "model_args": YOURMT3_MODEL_ARGS,
    "mono": True,
    "target_sample_rate": 16000,
    "segment_frames": 32767,
    "segment_hop_frames": 32767,
    "inference_batch_size": 8,
}


@dataclass(frozen=True)
class TranscriptionJob:
    kind: str
    record_id: str
    family: str
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


def expected_checkpoint_path(yourmt3_root: Path) -> Path:
    return Path(yourmt3_root) / YOURMT3_CHECKPOINT_REL


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
    jobs: list[TranscriptionJob] = []

    for record in records:
        record_id = checked_record_id(record["record_id"])
        family = str(record.get("family", "?"))
        if task == "gen":
            jobs.append(TranscriptionJob(
                kind="generated",
                record_id=record_id,
                family=family,
                audio_path=submission_dir / f"{record_id}.wav",
                midi_path=output_root / f"{record_id}.mid",
            ))
        else:
            for variant in EDIT_VARIANTS:
                jobs.append(TranscriptionJob(
                    kind="generated",
                    record_id=record_id,
                    family=family,
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


def load_yourmt3_model(yourmt3_root: Path, device: str = "auto"):
    """Load the fixed public YourMT3 model once."""
    root = Path(yourmt3_root).resolve()
    for value in (root, root / "amt" / "src"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    resolved_device = resolve_device(device)
    import torch

    original_cuda_available = torch.cuda.is_available
    original_cuda_device_count = torch.cuda.device_count
    previous_directory = Path.cwd()
    try:
        if resolved_device == "cpu":
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
        from model_helper import load_model_checkpoint

        os.chdir(root)
        model = load_model_checkpoint(args=list(YOURMT3_MODEL_ARGS), device="cpu")
    finally:
        torch.cuda.is_available = original_cuda_available
        torch.cuda.device_count = original_cuda_device_count
        os.chdir(previous_directory)
    model.to(resolved_device)
    model.eval()
    return model


def transcribe_audio(model: Any, audio_path: Path, midi_path: Path) -> None:
    """Use YourMT3 while hiding its internal ``model_output`` directory."""
    from model_helper import transcribe
    import torch
    import torchaudio

    audio, source_sr = torchaudio.load(str(audio_path))
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    target_sr = int(model.audio_cfg["sample_rate"])
    audio = torchaudio.functional.resample(audio, source_sr, target_sr)
    audio_info = {
        "filepath": str(audio_path),
        "track_name": audio_path.stem,
        "sample_rate": int(source_sr),
        "num_channels": 1,
        "num_frames": int(audio.shape[-1]),
        "duration": float(audio.shape[-1] / target_sr),
        "encoding": "wav",
        "bits_per_sample": 16,
    }
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pmuse_yourmt3_") as temporary:
        previous_directory = Path.cwd()
        original_cuda_available = torch.cuda.is_available
        model_device = next(model.parameters()).device
        try:
            os.chdir(temporary)
            if model_device.type == "cpu":
                torch.cuda.is_available = lambda: False
            produced = Path(transcribe(model, audio_info)).resolve()
        finally:
            torch.cuda.is_available = original_cuda_available
            os.chdir(previous_directory)
        if not produced.is_file():
            raise RuntimeError(f"YourMT3 did not create MIDI for {audio_path}")
        temporary_midi = midi_path.with_name(f".{midi_path.name}.tmp")
        shutil.copyfile(produced, temporary_midi)
        os.replace(temporary_midi, midi_path)


def transcribe_benchmark(
    *,
    testset_root: Path,
    submission_root: Path,
    benchmark: str,
    task: str,
    generated_midi_root: Path,
    yourmt3_root: Path,
    device: str = "auto",
    model_loader: Callable[[Path, str], Any] = load_yourmt3_model,
    transcriber: Callable[[Any, Path, Path], None] = transcribe_audio,
) -> list[dict[str, Any]]:
    """Transcribe submitted WAVs and atomically refresh the generated cache."""
    device = validate_device_name(device)
    if model_loader is load_yourmt3_model:
        check_yourmt3_paths(yourmt3_root)
    checkpoint = expected_checkpoint_path(yourmt3_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YourMT3 checkpoint not found: {checkpoint}")
    checkpoint_info = file_info(checkpoint)
    config_snapshot = transcription_config_snapshot()
    jobs = collect_transcription_jobs(
        testset_root, submission_root, benchmark, task, generated_midi_root,
    )

    output_root = Path(generated_midi_root) / benchmark / task
    cache_path = output_root / "cache_index.jsonl"
    cache = _read_cache(cache_path)
    statuses: list[dict[str, Any]] = []
    model = None

    for job in jobs:
        base = {
            "kind": job.kind,
            "record_id": job.record_id,
            "family": job.family,
            "variant": job.variant,
            "audio_path": str(job.audio_path),
            "midi_path": str(job.midi_path),
            "checkpoint_path": str(checkpoint),
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
            "transcription_config": config_snapshot,
        }
        if _matches_cache(job.midi_path, cached, identity):
            statuses.append({**identity, "status": "cached"})
            continue

        try:
            if model is None:
                model = model_loader(Path(yourmt3_root), device)
            transcriber(model, job.audio_path, job.midi_path)
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
    parser = argparse.ArgumentParser(description="Transcribe a P-MUSE submission with YourMT3")
    parser.add_argument("--testset-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--task", choices=("gen", "edit"), required=True)
    parser.add_argument("--generated-midi-dir", type=Path, required=True)
    parser.add_argument("--yourmt3-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    statuses = transcribe_benchmark(
        testset_root=args.testset_root,
        submission_root=args.submission_root,
        benchmark=args.benchmark,
        task=args.task,
        generated_midi_root=args.generated_midi_dir,
        yourmt3_root=args.yourmt3_root,
        device=args.device,
    )
    print(json.dumps({
        "benchmark": args.benchmark,
        "task": args.task,
        "expected_jobs": len(statuses),
        "status_counts": dict(sorted(Counter(
            item["status"] for item in statuses
        ).items())),
    }, indent=2))


if __name__ == "__main__":
    main()
