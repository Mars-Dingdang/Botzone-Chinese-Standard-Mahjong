import unittest

from mahjong_agent.training.ppo import generalized_advantage_estimate


class PPOTest(unittest.TestCase):
    def test_gae_propagates_terminal_reward(self):
        advantages, returns = generalized_advantage_estimate([0.0, 1.0], [0.0, 0.0], 1.0, 1.0)
        self.assertEqual(advantages, [1.0, 1.0])
        self.assertEqual(returns, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
