import unittest

from pmuse_eval.note_metrics import score_note_onsets


class PitchAwareOnsetTest(unittest.TestCase):
    def test_wrong_pitch_only_matches_when_pitch_switch_is_off(self) -> None:
        reference_intervals = [(1.0, 1.5)]
        estimated_intervals = [(1.0, 1.4)]

        pitched = score_note_onsets(
            reference_intervals,
            [60],
            estimated_intervals,
            [61],
            match_pitch=True,
        )
        onset_only = score_note_onsets(
            reference_intervals,
            [60],
            estimated_intervals,
            [61],
            match_pitch=False,
        )

        self.assertEqual(pitched["f1"], 0.0)
        self.assertEqual(onset_only["f1"], 1.0)

    def test_fifty_millisecond_boundary_is_inclusive(self) -> None:
        score = score_note_onsets(
            [(1.0, 1.5)],
            [60],
            [(1.05, 1.4)],
            [60],
            onset_tolerance=0.05,
            match_pitch=True,
        )

        self.assertEqual(score["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
