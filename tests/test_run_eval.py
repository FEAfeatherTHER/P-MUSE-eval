import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "run_eval.sh"


class RunEvalTest(unittest.TestCase):
    def test_new_binary_switches_reject_invalid_values_before_running_steps(self) -> None:
        for name in ("RUN_OFFSET", "ONSET_MATCH_PITCH", "OFFSET_MATCH_PITCH"):
            with self.subTest(name=name):
                environment = os.environ | {
                    "RUN_VALIDATE": "0",
                    "RUN_INSTRUMENT": "0",
                    "RUN_TRANSCRIBE": "0",
                    "RUN_ONSET": "0",
                    "RUN_OFFSET": "0",
                    "ONSET_MATCH_PITCH": "1",
                    "OFFSET_MATCH_PITCH": "1",
                    name: "invalid",
                }
                result = subprocess.run(
                    ["bash", str(RUNNER)],
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(f"{name} must be 0 or 1, got: invalid", result.stderr)

    def test_onset_and_offset_route_pitch_switches_to_their_modules(self) -> None:
        cases = (("0", "1"), ("1", "0"))
        for onset_match_pitch, offset_match_pitch in cases:
            with self.subTest(
                onset_match_pitch=onset_match_pitch,
                offset_match_pitch=offset_match_pitch,
            ):
                with TemporaryDirectory() as directory:
                    root = Path(directory)
                    invocation_log = root / "invocations.jsonl"
                    fake_python = root / "fake_python.py"
                    fake_python.write_text(
                        "#!/usr/bin/env python3\n"
                        "import json\n"
                        "import os\n"
                        "import sys\n"
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
                        "RUN_TRANSCRIBE": "0",
                        "RUN_ONSET": "1",
                        "RUN_OFFSET": "1",
                        "ONSET_MATCH_PITCH": onset_match_pitch,
                        "OFFSET_MATCH_PITCH": offset_match_pitch,
                        "BENCHMARK": "benchmark_style",
                        "TASK": "gen",
                        "TESTSET": str(root / "testset"),
                        "GENERATED_MIDI": str(root / "generated"),
                        "RESULTS": str(root / "results"),
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
                        for line in invocation_log.read_text(
                            encoding="utf-8",
                        ).splitlines()
                    ]

                self.assertEqual(
                    [invocation[1] for invocation in invocations],
                    ["pmuse_eval.onset", "pmuse_eval.offset"],
                )
                self.assertEqual(
                    "--no-match-pitch" in invocations[0], onset_match_pitch == "0",
                )
                self.assertEqual(
                    "--no-match-pitch" in invocations[1], offset_match_pitch == "0",
                )
                self.assertIn("[4/5] Onset F1 benchmark_style/gen", result.stdout)
                self.assertIn("[5/5] Offset F1 benchmark_style/gen", result.stdout)
