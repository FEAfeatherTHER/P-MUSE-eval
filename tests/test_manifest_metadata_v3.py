import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pmuse_eval import manifests


class BenchmarkMetadataValidationTest(unittest.TestCase):
    def test_rejects_missing_record_metadata(self) -> None:
        validator = getattr(manifests, "validate_benchmark_metadata", None)
        self.assertIsNotNone(validator, "benchmark metadata validation is missing")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.jsonl").write_text(
                json.dumps({
                    "record_id": "present",
                    "family": "bass",
                    "dataset_name": "nsynth",
                    "instrument_family": "bass",
                    "instrument_name": "bass_electronic_002",
                }) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing record_id.*absent"):
                validator(
                    root,
                    {"present": "bass", "absent": "bass"},
                    expected_rows=2,
                )

    def test_rejects_unknown_instrument_mapping(self) -> None:
        validator = getattr(manifests, "validate_benchmark_metadata", None)
        self.assertIsNotNone(validator, "benchmark metadata validation is missing")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.jsonl").write_text(
                json.dumps({
                    "record_id": "sample",
                    "family": "guitar",
                    "dataset_name": "unknown",
                    "instrument_family": "guitar",
                    "instrument_name": "unknown patch",
                }) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "No MuScriptor group mapping"):
                validator(root, {"sample": "guitar"}, expected_rows=1)

    def test_rejects_family_mismatch_with_benchmark(self) -> None:
        validator = manifests.validate_benchmark_metadata
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.jsonl").write_text(
                json.dumps({
                    "record_id": "sample",
                    "family": "drum",
                    "dataset_name": "e-gmd",
                    "instrument_family": "drum",
                    "instrument_name": "60s Rock",
                }) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "family mismatch"):
                validator(root, {"sample": "bass"}, expected_rows=1)


if __name__ == "__main__":
    unittest.main()
