import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from pmuse_eval.note_metrics import (
    load_midi_notes,
    score_note_offsets,
    score_note_onsets,
)


class NoteOnsetScoringTest(unittest.TestCase):
    def test_wrong_pitch_respects_match_pitch_switch(self) -> None:
        reference_intervals = [(0.0, 1.0)]
        estimated_intervals = [(0.0, 1.0)]

        pitched = score_note_onsets(
            reference_intervals,
            [60],
            estimated_intervals,
            [61],
            match_pitch=True,
        )
        pitch_agnostic = score_note_onsets(
            reference_intervals,
            [60],
            estimated_intervals,
            [61],
            match_pitch=False,
        )

        self.assertEqual(pitched["f1"], 0.0)
        self.assertEqual(pitch_agnostic["f1"], 1.0)

    def test_accepts_onset_exactly_at_50_millisecond_tolerance(self) -> None:
        scores = score_note_onsets(
            [(0.0, 1.0)],
            [60],
            [(0.05, 1.0)],
            [60],
        )

        self.assertEqual(scores["f1"], 1.0)


class NoteOffsetScoringTest(unittest.TestCase):
    def test_wrong_pitch_respects_match_pitch_switch(self) -> None:
        reference_intervals = [(0.0, 1.0)]
        estimated_intervals = [(0.0, 1.0)]

        pitched = score_note_offsets(
            reference_intervals,
            [60],
            estimated_intervals,
            [61],
            match_pitch=True,
        )
        pitch_agnostic = score_note_offsets(
            reference_intervals,
            [60],
            estimated_intervals,
            [61],
            match_pitch=False,
        )

        self.assertEqual(pitched["f1"], 0.0)
        self.assertEqual(pitch_agnostic["f1"], 1.0)

    def test_offset_match_requires_onset_match(self) -> None:
        scores = score_note_offsets(
            [(0.0, 1.0)],
            [60],
            [(0.06, 1.0)],
            [60],
        )

        self.assertEqual(scores["f1"], 0.0)

    def test_uses_twenty_percent_duration_offset_tolerance(self) -> None:
        within_tolerance = score_note_offsets(
            [(0.0, 1.0)],
            [60],
            [(0.0, 1.19)],
            [60],
        )
        outside_tolerance = score_note_offsets(
            [(0.0, 1.0)],
            [60],
            [(0.0, 1.21)],
            [60],
        )

        self.assertEqual(within_tolerance["f1"], 1.0)
        self.assertEqual(outside_tolerance["f1"], 0.0)


class MidiNoteLoadingTest(unittest.TestCase):
    def test_loads_windowed_intervals_with_integer_midi_pitches(self) -> None:
        with TemporaryDirectory() as directory:
            midi_path = Path(directory) / "notes.mid"
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)
            instrument.notes.append(pretty_midi.Note(100, 64, 0.1, 0.4))
            instrument.notes.append(pretty_midi.Note(100, 67, 0.7, 0.9))
            midi.instruments.append(instrument)
            midi.write(str(midi_path))

            notes = load_midi_notes(midi_path, start_sec=0.0, end_sec=0.5)

        self.assertEqual(notes.status, "ok")
        self.assertEqual(notes.pitches, (64,))
        self.assertAlmostEqual(notes.intervals[0][0], 0.1)
        self.assertAlmostEqual(notes.intervals[0][1], 0.4)
