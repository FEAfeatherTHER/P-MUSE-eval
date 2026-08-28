import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pmuse_eval.dependencies import YOURMT3_CHECKPOINT_REL
from pmuse_eval.onset import _transcription_evidence_status
from pmuse_eval.transcription import TRANSCRIPTION_CONFIG, transcribe_benchmark


class TranscriptionMetadataCacheTest(unittest.TestCase):
    """Cache decisions must depend on current paths, metadata, and config values."""

    def _fixture(self, root: Path) -> dict[str, Path]:
        benchmark_root = root / "testset" / "benchmark_style"
        benchmark_root.mkdir(parents=True)
        (benchmark_root / "benchmark_gen.jsonl").write_text(
            json.dumps({
                "record_id": "sample",
                "family": "piano",
                "target_midi_path": "reference.mid",
                "target_duration": 1.0,
            }) + "\n",
            encoding="utf-8",
        )
        submission_root = root / "submission"
        audio_path = submission_root / "benchmark_style" / "gen" / "sample.wav"
        audio_path.parent.mkdir(parents=True)
        audio_path.write_bytes(b"first audio")
        yourmt3_root = root / "yourmt3"
        checkpoint_path = yourmt3_root / YOURMT3_CHECKPOINT_REL
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.write_bytes(b"first checkpoint")
        return {
            "testset_root": root / "testset",
            "submission_root": submission_root,
            "generated_root": root / "generated",
            "yourmt3_root": yourmt3_root,
            "audio_path": audio_path,
            "checkpoint_path": checkpoint_path,
            "midi_path": root / "generated" / "benchmark_style" / "gen" / "sample.mid",
        }

    def _transcribe(self, paths: dict[str, Path], calls: list[str]) -> list[dict[str, object]]:
        def load_model(_: Path, __: str) -> object:
            return object()

        def write_midi(_: object, audio_path: Path, midi_path: Path) -> None:
            calls.append(str(audio_path))
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            midi_path.write_bytes(b"generated midi")

        return transcribe_benchmark(
            testset_root=paths["testset_root"],
            submission_root=paths["submission_root"],
            benchmark="benchmark_style",
            task="gen",
            generated_midi_root=paths["generated_root"],
            yourmt3_root=paths["yourmt3_root"],
            device="cpu",
            model_loader=load_model,
            transcriber=write_midi,
        )

    def test_matching_metadata_and_equal_config_values_reuses_midi(self) -> None:
        """Removing metadata/value comparisons would transcribe twice."""
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            calls: list[str] = []
            first_config = dict(TRANSCRIPTION_CONFIG)
            second_config = dict(TRANSCRIPTION_CONFIG)
            with patch("pmuse_eval.transcription.TRANSCRIPTION_CONFIG", first_config):
                self._transcribe(paths, calls)
            with patch("pmuse_eval.transcription.TRANSCRIPTION_CONFIG", second_config):
                second = self._transcribe(paths, calls)

        self.assertEqual(calls, [str(paths["audio_path"])])
        self.assertEqual(second[0]["status"], "cached")

    def test_changed_wav_metadata_retranscribes(self) -> None:
        """Ignoring WAV metadata would incorrectly reuse an old MIDI result."""
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            calls: list[str] = []
            self._transcribe(paths, calls)
            paths["audio_path"].write_bytes(b"changed audio data")
            second = self._transcribe(paths, calls)

        self.assertEqual(len(calls), 2)
        self.assertEqual(second[0]["status"], "ok")

    def test_changed_generated_midi_metadata_retranscribes(self) -> None:
        """Ignoring generated MIDI metadata would accept a replaced output file."""
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            calls: list[str] = []
            self._transcribe(paths, calls)
            paths["midi_path"].write_bytes(b"replaced generated midi")
            second = self._transcribe(paths, calls)

        self.assertEqual(len(calls), 2)
        self.assertEqual(second[0]["status"], "ok")

    def test_changed_checkpoint_metadata_retranscribes(self) -> None:
        """Ignoring checkpoint metadata would mix results from different models."""
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            calls: list[str] = []
            self._transcribe(paths, calls)
            paths["checkpoint_path"].write_bytes(b"changed checkpoint data")
            second = self._transcribe(paths, calls)

        self.assertEqual(len(calls), 2)
        self.assertEqual(second[0]["status"], "ok")

    def test_changed_transcription_config_retranscribes(self) -> None:
        """Comparing configuration identity instead of values would miss this change."""
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            calls: list[str] = []
            first_config = dict(TRANSCRIPTION_CONFIG)
            second_config = {**TRANSCRIPTION_CONFIG, "inference_batch_size": 4}
            with patch("pmuse_eval.transcription.TRANSCRIPTION_CONFIG", first_config):
                self._transcribe(paths, calls)
            with patch("pmuse_eval.transcription.TRANSCRIPTION_CONFIG", second_config):
                second = self._transcribe(paths, calls)

        self.assertEqual(len(calls), 2)
        self.assertEqual(second[0]["status"], "ok")

    def test_legacy_cache_record_is_replaced_by_metadata_record(self) -> None:
        """Treating legacy records as hits would preserve incomplete cache evidence."""
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            cache_path = paths["midi_path"].parent / "cache_index.jsonl"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text(json.dumps({
                "midi_path": str(paths["midi_path"]),
                "legacy_cache_marker": "legacy",
            }) + "\n", encoding="utf-8")
            calls: list[str] = []
            statuses = self._transcribe(paths, calls)
            record = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(statuses[0]["status"], "ok")
        self.assertEqual(len(calls), 1)
        self.assertEqual(record["midi_path"], str(paths["midi_path"]))
        self.assertIn("midi_file_info", record)
        self.assertNotIn("legacy_cache_marker", record)

    def test_metric_evidence_rejects_current_wav_or_midi_metadata_changes(self) -> None:
        """Skipping current-file checks would score stale transcription evidence."""
        with TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            calls: list[str] = []
            self._transcribe(paths, calls)
            cache_path = paths["midi_path"].parent / "cache_index.jsonl"
            entry = json.loads(cache_path.read_text(encoding="utf-8"))
            statuses = {("sample", None): "ok"}
            self.assertIsNone(
                _transcription_evidence_status(
                    "sample", None, paths["midi_path"], statuses,
                    {str(paths["midi_path"]): entry},
                ),
            )
            paths["audio_path"].write_bytes(b"new audio")
            self.assertEqual(
                _transcription_evidence_status(
                    "sample", None, paths["midi_path"], statuses,
                    {str(paths["midi_path"]): entry},
                ),
                "changed_transcription_audio",
            )
            paths["midi_path"].write_bytes(b"new midi")
            self.assertEqual(
                _transcription_evidence_status(
                    "sample", None, paths["midi_path"], statuses,
                    {str(paths["midi_path"]): entry},
                ),
                "changed_transcribed_midi",
            )


if __name__ == "__main__":
    unittest.main()
