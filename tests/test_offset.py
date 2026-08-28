import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pmuse_eval.note_metrics import MidiNotes
from pmuse_eval.offset import evaluate_offset


class OffsetEvaluationTest(unittest.TestCase):
    def test_drum_rows_are_preserved_but_excluded_from_means(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root = root / "testset" / "benchmark_style"
            benchmark_root.mkdir(parents=True)
            records = [
                {
                    "record_id": "drum-sample",
                    "family": "drum",
                    "target_midi_path": "drum.mid",
                    "target_duration": 1.0,
                },
                {
                    "record_id": "piano-sample",
                    "family": "piano",
                    "target_midi_path": "piano.mid",
                    "target_duration": 1.0,
                },
            ]
            (benchmark_root / "benchmark_gen.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            for name in ("drum.mid", "piano.mid"):
                (benchmark_root / name).touch()
            generated_root = root / "generated" / "benchmark_style" / "gen"
            generated_root.mkdir(parents=True)
            for name in ("drum-sample.mid", "piano-sample.mid"):
                (generated_root / name).touch()

            def load_notes(_: Path, *__: float) -> MidiNotes:
                return MidiNotes("ok", ((0.0, 1.0),), (60,))

            summary = evaluate_offset(
                testset_root=root / "testset",
                benchmark="benchmark_style",
                task="gen",
                generated_midi_root=root / "generated",
                results_dir=root / "results",
                verify_transcription_cache=False,
                interval_loader=load_notes,
            )
            samples = [
                json.loads(line)
                for line in (root / "results" / "offset" / "samples.jsonl").read_text(
                    encoding="utf-8",
                ).splitlines()
            ]

        drum_sample = next(item for item in samples if item["family"] == "drum")
        self.assertEqual(drum_sample["status"], "not_applicable")
        self.assertEqual(summary["overall"]["expected"], 1)
        self.assertEqual(summary["overall"]["mean_f1"], 1.0)
        self.assertNotIn("drum", summary["per_family"])
        self.assertEqual(summary["not_applicable"]["count"], 1)
        self.assertEqual(summary["not_applicable"]["excluded_from"], ["overall", "per_variant"])
