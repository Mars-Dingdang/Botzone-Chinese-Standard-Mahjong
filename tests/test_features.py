import unittest

from mahjong_agent.engine import MahjongEnv
from mahjong_agent.features import FEATURE_SIZE, encode_action, encode_observation


class FeatureTest(unittest.TestCase):
    def test_shapes(self):
        env = MahjongEnv()
        observation = env.observe(env.current_player)
        self.assertEqual(len(encode_observation(observation)), FEATURE_SIZE)
        for action in env.legal_actions():
            self.assertEqual(len(encode_action(action)), 8)


if __name__ == "__main__":
    unittest.main()
