"""Audio slicing helpers shared by P-MUSE metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioSlice:
    """A half-open time interval from one audio file."""

    path: Path
    start_sec: float = 0.0
    end_sec: float | None = None

    def __post_init__(self) -> None:
        if self.start_sec < 0:
            raise ValueError("audio slice start must be non-negative")
        if self.end_sec is not None and self.end_sec <= self.start_sec:
            raise ValueError("audio slice end must be greater than start")


def segment_duration(segment: dict[str, Any]) -> float:
    """Return a segment duration from either public record representation."""

    if "duration" in segment:
        duration = float(segment["duration"])
    else:
        duration = float(segment["end_sec"]) - float(segment.get("start_sec", 0.0))
    if duration <= 0:
        raise ValueError(f"segment duration must be positive, got {duration}")
    return duration


def edit_target_interval(record: dict[str, Any]) -> tuple[float, float]:
    """Return the edited target interval inside the complete generated WAV."""

    start = segment_duration(record["prefix"])
    return start, start + segment_duration(record["target"])


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _record_segment_slice(
    record: dict[str, Any], benchmark_root: Path, name: str
) -> AudioSlice:
    segment = record[name]
    if "audio_clip_path" in segment:
        return AudioSlice(
            _resolve(benchmark_root, segment["audio_clip_path"]),
            0.0,
            segment_duration(segment),
        )
    return AudioSlice(
        _resolve(benchmark_root, segment.get("audio_path", record["audio_path"])),
        float(segment["start_sec"]),
        float(segment["end_sec"]),
    )


def generation_prompt_slices(
    benchmark: str, record: dict[str, Any], benchmark_root: Path
) -> tuple[AudioSlice, ...]:
    """Build the prompt reference for one generation example."""

    benchmark_root = Path(benchmark_root)
    if benchmark == "benchmark_paired":
        return (
            AudioSlice(
                _resolve(benchmark_root, record["audio_path"]),
                float(record["prefix_start_sec"]),
                float(record["prefix_end_sec"]),
            ),
        )
    if benchmark == "benchmark_style":
        return (AudioSlice(_resolve(benchmark_root, record["prompt_audio_path"])),)
    if benchmark == "benchmark_mixed":
        return (
            AudioSlice(
                _resolve(benchmark_root, record["style_prompt_audio_path"]),
                float(record["style_prompt_start_sec"]),
                float(record["style_prompt_end_sec"]),
            ),
            AudioSlice(
                _resolve(benchmark_root, record["audio_path"]),
                float(record["prefix_start_sec"]),
                float(record["prefix_end_sec"]),
            ),
        )
    raise ValueError(f"unknown benchmark: {benchmark}")


def editing_prompt_slices(
    benchmark: str, record: dict[str, Any], benchmark_root: Path
) -> tuple[AudioSlice, ...]:
    """Build the pre-edit timbre prompt in the established concatenation order."""

    benchmark_root = Path(benchmark_root)
    prompt = [
        _record_segment_slice(record, benchmark_root, "prefix"),
        _record_segment_slice(record, benchmark_root, "suffix"),
    ]
    if benchmark == "benchmark_mixed":
        prompt.append(
            AudioSlice(
                _resolve(benchmark_root, record["style_prompt_audio_path"]),
                float(record["style_prompt_start_sec"]),
                float(record["style_prompt_end_sec"]),
            )
        )
    elif benchmark not in {"benchmark_paired", "benchmark_style"}:
        raise ValueError(f"unknown benchmark: {benchmark}")
    return tuple(prompt)


def load_audio_slices(
    slices: Iterable[AudioSlice],
    sample_rate: int = SAMPLE_RATE,
    loader: Callable[..., Any] | None = None,
):
    """Load, mono-mix, resample, and concatenate slices as float32 audio."""

    import numpy as np

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"instrument embedding requires {SAMPLE_RATE} Hz audio")
    if loader is None:
        import librosa

        loader = librosa.load

    chunks = []
    for audio_slice in slices:
        duration = (
            None
            if audio_slice.end_sec is None
            else audio_slice.end_sec - audio_slice.start_sec
        )
        audio, _ = loader(
            str(audio_slice.path),
            sr=sample_rate,
            mono=True,
            offset=audio_slice.start_sec,
            duration=duration,
        )
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            raise ValueError(f"empty audio slice: {audio_slice.path}")
        chunks.append(chunk)
    if not chunks:
        raise ValueError("at least one audio slice is required")
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
