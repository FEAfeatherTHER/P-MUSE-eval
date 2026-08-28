import contextlib
import io
import unittest
from unittest.mock import patch

from pmuse_eval.offset import main as offset_main
from pmuse_eval.onset import main as onset_main


class MetricCliTest(unittest.TestCase):
    def test_cli_defaults_to_pitch_matching_and_accepts_no_match_pitch(self) -> None:
        arguments = [
            "--testset-root", "testset",
            "--benchmark", "benchmark_style",
            "--task", "gen",
            "--generated-midi-dir", "generated",
            "--results-dir", "results",
        ]
        cases = (
            ("pmuse_eval.onset.evaluate_onset", onset_main),
            ("pmuse_eval.offset.evaluate_offset", offset_main),
        )

        for target, main in cases:
            for flags, match_pitch in (
                ([], True),
                (["--match-pitch"], True),
                (["--no-match-pitch"], False),
            ):
                with self.subTest(
                    target=target, flags=flags, match_pitch=match_pitch,
                ):
                    with patch(target, return_value={}) as evaluate:
                        with contextlib.redirect_stdout(io.StringIO()):
                            main(arguments + flags)
                    self.assertEqual(
                        evaluate.call_args.kwargs["match_pitch"], match_pitch,
                    )
