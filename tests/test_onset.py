import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pmuse_eval.note_metrics import MidiNotes
from pmuse_eval.onset import evaluate_onset


class OnsetEvaluationTest(unittest.TestCase):
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
