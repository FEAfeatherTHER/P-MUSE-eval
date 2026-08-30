import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "run_eval.sh"


class RunEvalV3Test(unittest.TestCase):
    def test_routes_muscriptor_and_onset_without_offset(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            invocation_log = root / "invocations.jsonl"
            fake_python = root / "fake_python.py"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "with Path(os.environ['INVOCATION_LOG']).open('a', encoding='utf-8') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            environment = os.environ | {
                "PYTHON_BIN": str(fake_python),
                "INVOCATION_LOG": str(invocation_log),
                "RUN_VALIDATE": "0",
                "RUN_INSTRUMENT": "0",
                "RUN_TRANSCRIBE": "1",
                "RUN_ONSET": "1",
                "ONSET_MATCH_PITCH": "0",
                "BENCHMARK": "benchmark_style",
                "TASK": "gen",
                "TESTSET": str(root / "testset"),
                "SUBMISSION": str(root / "submission"),
                "GENERATED_MIDI": str(root / "generated"),
                "RESULTS": str(root / "results"),
                "MUSCRIPTOR_MODEL": "large",
                "DEVICE": "cpu",
            }

            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            invocations = [
                json.loads(line)
                for line in invocation_log.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [invocation[1] for invocation in invocations],
            ["pmuse_eval.transcription", "pmuse_eval.onset"],
        )
        self.assertIn("--model", invocations[0])
        self.assertIn("large", invocations[0])
        self.assertNotIn("--yourmt3-root", invocations[0])
        self.assertIn("--no-match-pitch", invocations[1])
        self.assertIn("[3/4] Transcribe audio benchmark_style/gen", result.stdout)
        self.assertIn("[4/4] Onset F1 benchmark_style/gen", result.stdout)
        self.assertNotIn("Offset", result.stdout)


if __name__ == "__main__":
    unittest.main()
