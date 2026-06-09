import unittest

from mahjong_agent.evaluation import evaluate_duplicate
from mahjong_agent.policies import HeuristicPolicy
from mahjong_agent.training.checkpoint import early_stopping_state


class EvaluationTest(unittest.TestCase):
    def test_duplicate_result_is_labeled_and_has_confidence_interval(self):
        result = evaluate_duplicate(
            HeuristicPolicy(), HeuristicPolicy(), walls=1, seed=7,
            policy_a_name="left", policy_b_name="right")
        self.assertEqual(result["policy_a"], "left")
        self.assertEqual(result["policy_b"], "right")
        self.assertEqual(len(result["score_delta_95_ci"]), 2)

    def test_early_stopping_state_tracks_best_and_stale_epochs(self):
        best, stale, improved = early_stopping_state(0.6, 2, 0.61)
        self.assertEqual((best, stale, improved), (0.61, 0, True))
        self.assertEqual(early_stopping_state(best, stale, 0.60), (0.61, 1, False))


if __name__ == "__main__":
    unittest.main()
