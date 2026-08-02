"""Instrument-embedding cosine similarity for P-MUSE benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .audio import (
    SAMPLE_RATE,
    AudioSlice,
    edit_target_interval,
    editing_prompt_slices,
    generation_prompt_slices,
    load_audio_slices,
)
from .cache import write_json_atomic, write_jsonl_atomic
from .dependencies import check_instrument_paths
from .manifests import BENCHMARKS, EDIT_VARIANTS, checked_record_id, read_jsonl


@dataclass(frozen=True)
class InstrumentItem:
    record_id: str
    family: str
    variant: str | None
    prompt_slices: tuple[AudioSlice, ...]
    generated_path: Path
    generated_slice: AudioSlice

def collect_instrument_items(
    testset_root: Path,
    submission_root: Path,
    benchmark: str,
    task: str,
) -> list[InstrumentItem]:
    """Collect all expected samples without dropping missing generated files."""

    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {benchmark}")
    if task not in {"gen", "edit"}:
        raise ValueError(f"unknown task: {task}")
    benchmark_root = Path(testset_root) / benchmark
    records = read_jsonl(benchmark_root / f"benchmark_{task}.jsonl")
    generated_root = Path(submission_root) / benchmark / task
    items: list[InstrumentItem] = []
    for record in records:
        record_id = checked_record_id(record["record_id"])
        family = str(record["family"])
        if task == "gen":
            generated = generated_root / f"{record_id}.wav"
            items.append(
                InstrumentItem(
                    record_id=record_id,
                    family=family,
                    variant=None,
                    prompt_slices=generation_prompt_slices(
                        benchmark, record, benchmark_root
                    ),
                    generated_path=generated,
                    generated_slice=AudioSlice(generated),
                )
            )
            continue

        e_start, e_end = edit_target_interval(record)
        prompt = editing_prompt_slices(benchmark, record, benchmark_root)
        for variant in EDIT_VARIANTS:
            generated = generated_root / f"{record_id}__{variant}.wav"
            items.append(
                InstrumentItem(
                    record_id=record_id,
                    family=family,
                    variant=variant,
                    prompt_slices=prompt,
                    generated_path=generated,
                    generated_slice=AudioSlice(generated, e_start, e_end),
                )
            )
    return items


def cosine_similarity(first, second) -> float:
    first_values = [float(value) for value in first]
    second_values = [float(value) for value in second]
    if len(first_values) != len(second_values):
        raise ValueError("embedding dimensions differ")
    dot = sum(a * b for a, b in zip(first_values, second_values))
    first_norm = math.sqrt(sum(value * value for value in first_values))
    second_norm = math.sqrt(sum(value * value for value in second_values))
    denominator = first_norm * second_norm
    score = 0.0 if denominator == 0.0 else dot / denominator
    if not math.isfinite(score):
        raise ValueError("embedding cosine similarity is not finite")
    return score


class InstrumentEmbeddingExtractor:
    """Lazy loader for the upstream musical instrument embedding model."""

    def __init__(
        self,
        instrument_embedding_root: Path,
        config_file: Path,
        checkpoint: Path,
        device: str = "auto",
    ) -> None:
        self.root = Path(instrument_embedding_root).resolve()
        self.config_file = Path(config_file).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device_name = device
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        if not self.config_file.is_file():
            raise FileNotFoundError(f"instrument config not found: {self.config_file}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"instrument checkpoint not found: {self.checkpoint}")
        device = self.device_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from musyn.models import create_model
        from musyn.utils import Config

        config = Config.fromfile(str(self.config_file))
        model = create_model(config.model)
        load_args = {
            "map_location": lambda storage, location: storage,
            "weights_only": False,
        }
        try:
            checkpoint = torch.load(str(self.checkpoint), **load_args)
        except TypeError as exc:
            if "weights_only" not in str(exc):
                raise
            load_args.pop("weights_only")
            checkpoint = torch.load(str(self.checkpoint), **load_args)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()

    def embed(self, audio) -> Any:
        import numpy as np
        import torch

        self._load()
        waveform = torch.from_numpy(
            np.ascontiguousarray(audio, dtype=np.float32)
        ).unsqueeze(0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feature = self._model.predict(waveform)
        embedding = feature.squeeze(0).detach().cpu().numpy().astype(np.float32)
        embedding = embedding.reshape(-1)
        if embedding.size != 512:
            raise ValueError(f"expected 512-dimensional embedding, got {embedding.size}")
        return embedding


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_instrument_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        values = [
            float(sample["cosine_similarity"])
            for sample in group
            if sample["status"] == "scored"
        ]
        return {
            "expected": len(group),
            "scored": len(values),
            "skipped": len(group) - len(values),
            "mean": _mean(values),
        }

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_family[str(sample["family"])].append(sample)
        if sample.get("variant") is not None:
            by_variant[str(sample["variant"])].append(sample)
    return {
        "metric": "instrument_embedding_cosine_similarity",
        "overall": summarize(samples),
        "per_family": {
            key: summarize(group) for key, group in sorted(by_family.items())
        },
        "per_variant": {
            key: summarize(group) for key, group in sorted(by_variant.items())
        },
        "status_counts": dict(sorted(
            (status, sum(sample["status"] == status for sample in samples))
            for status in {sample["status"] for sample in samples}
        )),
    }


def evaluate_instrument(
    testset_root: Path,
    submission_root: Path,
    benchmark: str,
    task: str,
    results_dir: Path,
    instrument_embedding_root: Path,
    device: str = "auto",
    extractor: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one benchmark/task and atomically replace its result files."""

    paths = check_instrument_paths(instrument_embedding_root)
    instrument_embedding_root = Path(paths["root"])
    checkpoint = Path(paths["checkpoint"])
    config_file = Path(paths["config_file"])
    if extractor is None:
        extractor = InstrumentEmbeddingExtractor(
            instrument_embedding_root, config_file, checkpoint, device
        )
    items = collect_instrument_items(testset_root, submission_root, benchmark, task)
    samples: list[dict[str, Any]] = []

    for item in items:
        result: dict[str, Any] = {
            "record_id": item.record_id,
            "family": item.family,
            "variant": item.variant,
            "generated_audio": str(item.generated_path),
            "status": "scored",
            "cosine_similarity": None,
        }
        if not item.generated_path.is_file():
            result["status"] = "missing_generated"
            samples.append(result)
            continue
        try:
            prompt_embedding = extractor.embed(
                load_audio_slices(item.prompt_slices)
            )
            generated_embedding = extractor.embed(
                load_audio_slices((item.generated_slice,))
            )
            result["cosine_similarity"] = cosine_similarity(
                prompt_embedding, generated_embedding
            )
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
        samples.append(result)

    summary = summarize_instrument_samples(samples)
    summary.update(
        {
            "benchmark": benchmark,
            "task": task,
            "sample_rate": SAMPLE_RATE,
        }
    )
    output = Path(results_dir) / "instrument"
    write_jsonl_atomic(output / "samples.jsonl", samples)
    write_json_atomic(output / "summary.json", summary)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate P-MUSE instrument similarity"
    )
    parser.add_argument("--testset-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--task", choices=("gen", "edit"), required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--instrument-embedding-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    summary = evaluate_instrument(
        testset_root=args.testset_root,
        submission_root=args.submission_root,
        benchmark=args.benchmark,
        task=args.task,
        results_dir=args.results_dir / args.benchmark / args.task,
        instrument_embedding_root=args.instrument_embedding_root,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
