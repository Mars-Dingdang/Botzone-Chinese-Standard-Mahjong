import unittest

from mahjong_agent.engine import MahjongEnv
from mahjong_agent.features import (FEATURE_SIZE, compact_observation,
                                    deserialize_action, encode_action,
                                    encode_observation, expand_observation,
                                    serialize_action)


class FeatureTest(unittest.TestCase):
    def test_shapes(self):
        env = MahjongEnv()
        observation = env.observe(env.current_player)
        self.assertEqual(len(encode_observation(observation)), FEATURE_SIZE)
        for action in env.legal_actions():
            self.assertEqual(len(encode_action(action)), 8)

    def test_compact_round_trip(self):
        env = MahjongEnv()
        observation = env.observe(env.current_player)
        compact = compact_observation(observation)
        self.assertEqual(encode_observation(observation), encode_observation(expand_observation(compact)))
        action = env.legal_actions()[0]
        self.assertEqual(action.key(), deserialize_action(serialize_action(action)).key())


if __name__ == "__main__":
    unittest.main()
