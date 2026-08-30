"""Shared MIDI note loading and note-level metric helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_ONSET_TOLERANCE = 0.05


@dataclass(frozen=True)
class MidiNotes:
    status: str
    intervals: tuple[tuple[float, float], ...] = ()
    pitches: tuple[int, ...] = ()
    error: str | None = None


def load_midi_notes(
    midi_path: Path,
    start_sec: float = 0.0,
    end_sec: float = math.inf,
    shift_sec: float = 0.0,
) -> MidiNotes:
    """Load note intervals and integer MIDI pitches in a time window."""
    try:
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        return MidiNotes("parse_error", error=str(exc))

    intervals: list[tuple[float, float]] = []
    pitches: list[int] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            onset = float(note.start)
            if onset < start_sec or onset >= end_sec:
                continue
            shifted_start = max(start_sec, onset) + shift_sec
            shifted_end = min(end_sec, float(note.end)) + shift_sec
            if shifted_end <= shifted_start:
                shifted_end = shifted_start + 1e-3
            intervals.append((shifted_start, shifted_end))
            pitches.append(int(note.pitch))
    if not intervals:
        return MidiNotes("empty")
    return MidiNotes("ok", tuple(intervals), tuple(pitches))


def _midi_to_hz(pitches: Sequence[int]):
    import numpy as np

    values = np.asarray(pitches, dtype=np.float64)
    return 440.0 * np.power(2.0, (values - 69.0) / 12.0)


def _interval_array(intervals: Sequence[tuple[float, float]]):
    """Return intervals with the ``(n, 2)`` shape expected by mir_eval."""
    import numpy as np

    return np.asarray(intervals, dtype=np.float64).reshape(-1, 2)


def score_note_onsets(
    reference_intervals: Sequence[tuple[float, float]],
    reference_pitches: Sequence[int],
    estimated_intervals: Sequence[tuple[float, float]],
    estimated_pitches: Sequence[int],
    onset_tolerance: float = DEFAULT_ONSET_TOLERANCE,
    match_pitch: bool = True,
) -> dict[str, float]:
    """Compute one-to-one onset precision, recall, and F1."""
    from mir_eval.transcription import (
        onset_precision_recall_f1,
        precision_recall_f1_overlap,
    )

    reference = _interval_array(reference_intervals)
    estimated = _interval_array(estimated_intervals)
    if len(estimated) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if match_pitch:
        precision, recall, f1, _ = precision_recall_f1_overlap(
            reference,
            _midi_to_hz(reference_pitches),
            estimated,
            _midi_to_hz(estimated_pitches),
            onset_tolerance=float(onset_tolerance),
            offset_ratio=None,
        )
    else:
        precision, recall, f1 = onset_precision_recall_f1(
            reference,
            estimated,
            onset_tolerance=float(onset_tolerance),
        )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
