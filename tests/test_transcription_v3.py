import unittest
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pmuse_eval import transcription


class InstrumentConstraintTest(unittest.TestCase):
    def test_family_maps_to_all_relevant_muscriptor_groups(self) -> None:
        mapper = getattr(transcription, "instrument_groups_for_family", None)
        self.assertIsNotNone(mapper, "MuScriptor family mapping is missing")
        self.assertEqual(
            mapper("piano"),
            ("acoustic_piano", "electric_piano"),
        )
        self.assertEqual(
            mapper("guitar"),
            (
                "acoustic_guitar",
                "clean_electric_guitar",
                "distorted_electric_guitar",
            ),
        )
        self.assertEqual(
            mapper("bass"),
            ("acoustic_bass", "electric_bass"),
        )
        self.assertEqual(mapper("drum"), ("drums",))


class LocalCheckpointTest(unittest.TestCase):
    def test_local_checkpoint_requires_adjacent_config(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.write_bytes(b"weights")

            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                transcription.resolve_muscriptor_checkpoint(str(checkpoint))

            (checkpoint.parent / "config.json").write_text(
                '{"dim": 1536, "num_heads": 24, "num_layers": 48, "card": 1395}',
                encoding="utf-8",
            )

            self.assertEqual(
                transcription.resolve_muscriptor_checkpoint(str(checkpoint)),
                checkpoint,
            )


class MuScriptorMidiWriterTest(unittest.TestCase):
    def test_writes_postprocessed_midi_with_family_constraint(self) -> None:
        transcriber = transcription.transcribe_audio
        if "family" not in inspect.signature(transcriber).parameters:
            self.fail("MuScriptor transcriber must accept the benchmark family")

        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = None

            def transcribe_to_midi(self, audio_path, **kwargs):
                self.kwargs = {"audio_path": audio_path, **kwargs}
                return b"MThd-muscriptor"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "sample.wav"
            audio_path.write_bytes(b"audio")
            midi_path = root / "nested" / "sample.mid"
            model = FakeModel()

            transcriber(model, audio_path, midi_path, "guitar")

            self.assertEqual(midi_path.read_bytes(), b"MThd-muscriptor")
            self.assertFalse(midi_path.with_name(".sample.mid.tmp").exists())

        self.assertEqual(
            model.kwargs,
            {
                "audio_path": audio_path,
                "use_sampling": False,
                "temperature": 1.0,
                "cfg_coef": 1.0,
                "instruments": [
                    "acoustic_guitar",
                    "clean_electric_guitar",
                    "distorted_electric_guitar",
                ],
                "batch_size": 1,
                "no_eos_is_ok": True,
                "beam_size": 1,
                "prelude_forcing": True,
                "detect_tempo": "best-effort",
            },
        )


class BenchmarkTranscriptionTest(unittest.TestCase):
    def test_loads_model_once_and_reuses_family_aware_cache(self) -> None:
        api = transcription.transcribe_benchmark
        parameters = inspect.signature(api).parameters
        if "model_source" not in parameters:
            self.fail("transcribe_benchmark must expose the MuScriptor model source")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root = root / "testset" / "benchmark_style"
            benchmark_root.mkdir(parents=True)
            records = [
                {"record_id": "piano_sample", "family": "piano"},
                {"record_id": "drum_sample", "family": "drum"},
            ]
            (benchmark_root / "benchmark_gen.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            submission = root / "submission" / "benchmark_style" / "gen"
            submission.mkdir(parents=True)
            for record in records:
                (submission / f"{record['record_id']}.wav").write_bytes(b"audio")
            checkpoint = root / "weights" / "model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"large checkpoint")
            config = checkpoint.parent / "config.json"
            config.write_text('{"variant": "large"}', encoding="utf-8")
            generated = root / "generated"
            loader_calls: list[tuple[Path, str]] = []
            transcriber_calls: list[str] = []

            def resolve_checkpoint(model_source: str) -> Path:
                self.assertEqual(model_source, "large")
                return checkpoint

            def load_model(path: Path, device: str):
                loader_calls.append((path, device))
                return object()

            def write_midi(model, audio_path: Path, midi_path: Path, family: str) -> None:
                self.assertIsNotNone(model)
                transcriber_calls.append(family)
                midi_path.parent.mkdir(parents=True, exist_ok=True)
                midi_path.write_bytes(family.encode())

            first = api(
                testset_root=root / "testset",
                submission_root=root / "submission",
                benchmark="benchmark_style",
                task="gen",
                generated_midi_root=generated,
                model_source="large",
                device="cpu",
                checkpoint_resolver=resolve_checkpoint,
                model_loader=load_model,
                transcriber=write_midi,
            )
            second = api(
                testset_root=root / "testset",
                submission_root=root / "submission",
                benchmark="benchmark_style",
                task="gen",
                generated_midi_root=generated,
                model_source="large",
                device="cpu",
                checkpoint_resolver=resolve_checkpoint,
                model_loader=load_model,
                transcriber=write_midi,
            )
            cache = [
                json.loads(line)
                for line in (
                    generated / "benchmark_style" / "gen" / "cache_index.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(loader_calls, [(checkpoint, "cpu")])
        self.assertEqual(transcriber_calls, ["piano", "drum"])
        self.assertEqual([item["status"] for item in first], ["ok", "ok"])
        self.assertEqual([item["status"] for item in second], ["cached", "cached"])
        self.assertEqual(
            [item["instrument_groups"] for item in cache],
            [
                ["drums"],
                ["acoustic_piano", "electric_piano"],
            ],
        )
        self.assertTrue(all(item["model_source"] == "large" for item in cache))

    def test_config_change_invalidates_cached_transcription(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_root = root / "testset" / "benchmark_paired"
            benchmark_root.mkdir(parents=True)
            (benchmark_root / "benchmark_gen.jsonl").write_text(
                json.dumps({"record_id": "sample", "family": "piano"}) + "\n",
                encoding="utf-8",
            )
            submission = root / "submission" / "benchmark_paired" / "gen"
            submission.mkdir(parents=True)
            (submission / "sample.wav").write_bytes(b"audio")
            checkpoint = root / "weights" / "model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"weights")
            config = checkpoint.parent / "config.json"
            config.write_text('{"revision": 1}', encoding="utf-8")
            transcriber_calls: list[str] = []

            def write_midi(model, audio_path, midi_path, family):
                transcriber_calls.append(family)
                midi_path.parent.mkdir(parents=True, exist_ok=True)
                midi_path.write_bytes(b"midi")

            arguments = {
                "testset_root": root / "testset",
                "submission_root": root / "submission",
                "benchmark": "benchmark_paired",
                "task": "gen",
                "generated_midi_root": root / "generated",
                "model_source": str(checkpoint),
                "device": "cpu",
                "model_loader": lambda path, device: object(),
                "transcriber": write_midi,
            }
            transcription.transcribe_benchmark(**arguments)
            config.write_text('{"revision": 2}', encoding="utf-8")
            transcription.transcribe_benchmark(**arguments)

        self.assertEqual(transcriber_calls, ["piano", "piano"])


if __name__ == "__main__":
    unittest.main()
