import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from pmuse_eval.note_metrics import MidiNotes
from pmuse_eval.onset import (
    MidiIntervals,
    evaluate_onset,
    load_midi_intervals,
    score_interval_onsets,
)


class OnsetEvaluationTest(unittest.TestCase):
    def test_legacy_loader_uses_legacy_scoring_when_scorer_is_default(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root = root / "testset" / "benchmark_style"
            benchmark_root.mkdir(parents=True)
            (benchmark_root / "reference.mid").touch()
            (benchmark_root / "benchmark_gen.jsonl").write_text(
                json.dumps({
                    "record_id": "legacy-loader",
                    "family": "piano",
                    "target_midi_path": "reference.mid",
                    "target_duration": 1.0,
                }) + "\n",
                encoding="utf-8",
            )
            generated_root = root / "generated" / "benchmark_style" / "gen"
            generated_root.mkdir(parents=True)
            (generated_root / "legacy-loader.mid").touch()

            def load_intervals(_: Path, *__: float) -> MidiIntervals:
                return MidiIntervals("ok", ((0.0, 1.0),))

            summary = evaluate_onset(
                testset_root=root / "testset",
                benchmark="benchmark_style",
                task="gen",
                generated_midi_root=root / "generated",
                results_dir=root / "results",
                verify_transcription_cache=False,
                interval_loader=load_intervals,
            )

        self.assertEqual(summary["overall"]["scored"], 1)
        self.assertEqual(summary["overall"]["mean_f1"], 1.0)

    def test_legacy_scorer_receives_intervals_when_loader_is_default(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root = root / "testset" / "benchmark_style"
            benchmark_root.mkdir(parents=True)
            reference_path = benchmark_root / "reference.mid"
            generated_root = root / "generated" / "benchmark_style" / "gen"
            generated_root.mkdir(parents=True)
            generated_path = generated_root / "legacy-scorer.mid"
            for path in (reference_path, generated_path):
                midi = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(program=0)
                instrument.notes.append(pretty_midi.Note(100, 60, 0.0, 1.0))
                midi.instruments.append(instrument)
                midi.write(str(path))
            (benchmark_root / "benchmark_gen.jsonl").write_text(
                json.dumps({
                    "record_id": "legacy-scorer",
                    "family": "piano",
                    "target_midi_path": "reference.mid",
                    "target_duration": 1.0,
                }) + "\n",
                encoding="utf-8",
            )

            def score_intervals(
                reference: tuple[tuple[float, float], ...],
                estimated: tuple[tuple[float, float], ...],
                tolerance: float,
            ) -> dict[str, float]:
                self.assertEqual(len(reference), 1)
                self.assertEqual(len(estimated), 1)
                self.assertEqual(tolerance, 0.05)
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

            summary = evaluate_onset(
                testset_root=root / "testset",
                benchmark="benchmark_style",
                task="gen",
                generated_midi_root=root / "generated",
                results_dir=root / "results",
                verify_transcription_cache=False,
                interval_scorer=score_intervals,
            )

        self.assertEqual(summary["overall"]["scored"], 1)
        self.assertEqual(summary["overall"]["mean_f1"], 1.0)

    def test_legacy_loader_returns_legacy_interval_result(self) -> None:
        result = load_midi_intervals(Path("missing.mid"))

        self.assertIsInstance(result, MidiIntervals)
        self.assertEqual(result.status, "parse_error")
        self.assertIsNotNone(result.error)

    def test_legacy_interval_helpers_and_injected_callbacks_remain_compatible(self) -> None:
        legacy_parse_error = MidiIntervals("parse_error", (), "legacy error")
        self.assertEqual(legacy_parse_error.error, "legacy error")
        self.assertEqual(
            score_interval_onsets([(0.0, 1.0)], [(0.0, 1.0)])["f1"], 1.0,
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root = root / "testset" / "benchmark_style"
            benchmark_root.mkdir(parents=True)
            (benchmark_root / "reference.mid").touch()
            (benchmark_root / "benchmark_gen.jsonl").write_text(
                json.dumps({
                    "record_id": "legacy-sample",
                    "family": "piano",
                    "target_midi_path": "reference.mid",
                    "target_duration": 1.0,
                }) + "\n",
                encoding="utf-8",
            )
            generated_root = root / "generated" / "benchmark_style" / "gen"
            generated_root.mkdir(parents=True)
            (generated_root / "legacy-sample.mid").touch()

            def load_intervals(_: Path, *__: float) -> MidiIntervals:
                return MidiIntervals("ok", ((0.0, 1.0),))

            def score_intervals(
                reference: list[tuple[float, float]],
                estimated: list[tuple[float, float]],
                tolerance: float,
            ) -> dict[str, float]:
                self.assertEqual(reference, ((0.0, 1.0),))
                self.assertEqual(estimated, ((0.0, 1.0),))
                self.assertEqual(tolerance, 0.05)
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

            summary = evaluate_onset(
                testset_root=root / "testset",
                benchmark="benchmark_style",
                task="gen",
                generated_midi_root=root / "generated",
                results_dir=root / "results",
                verify_transcription_cache=False,
                interval_loader=load_intervals,
                interval_scorer=score_intervals,
            )

        self.assertEqual(summary["overall"]["scored"], 1)
        self.assertEqual(summary["overall"]["mean_f1"], 1.0)

    def test_pitch_agnostic_switch_is_recorded_and_changes_scores(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root = root / "testset" / "benchmark_style"
            benchmark_root.mkdir(parents=True)
            (benchmark_root / "reference.mid").touch()
            (benchmark_root / "benchmark_gen.jsonl").write_text(
                json.dumps({
                    "record_id": "sample",
                    "family": "piano",
                    "target_midi_path": "reference.mid",
                    "target_duration": 1.0,
                }) + "\n",
                encoding="utf-8",
            )
            generated_root = root / "generated" / "benchmark_style" / "gen"
            generated_root.mkdir(parents=True)
            (generated_root / "sample.mid").touch()

            def load_notes(path: Path, *_: float) -> MidiNotes:
                pitch = 60 if path.name == "reference.mid" else 61
                return MidiNotes("ok", ((0.0, 1.0),), (pitch,))

            summary = evaluate_onset(
                testset_root=root / "testset",
                benchmark="benchmark_style",
                task="gen",
                generated_midi_root=root / "generated",
                results_dir=root / "results",
                match_pitch=False,
                verify_transcription_cache=False,
                interval_loader=load_notes,
            )

            samples = (root / "results" / "onset" / "samples.jsonl").read_text(
                encoding="utf-8",
            )

        self.assertEqual(summary["metric"], "onset_f1")
        self.assertFalse(summary["match_pitch"])
        self.assertEqual(summary["overall"]["mean_f1"], 1.0)
        self.assertIn('"f1": 1.0', samples)
