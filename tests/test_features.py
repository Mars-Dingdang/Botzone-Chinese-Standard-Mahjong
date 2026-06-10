import unittest
import copy

from mahjong_agent.engine import MahjongEnv
from mahjong_agent.features import (FEATURE_SIZE, compact_observation,
                                    deserialize_action, encode_action,
                                    encode_observation, encode_observation_v2,
                                    expand_observation,
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

    def test_v2_tokens_are_fixed_shape_and_public_only(self):
        env = MahjongEnv()
        observation = env.observe(0)
        tokens, mask = encode_observation_v2(observation)
        self.assertEqual((len(tokens), len(mask)), (256, 256))
        self.assertNotIn("hands", observation)
        self.assertNotIn("wall", observation)

    def test_v2_relative_seat_rotation_is_invariant(self):
        env = MahjongEnv()
        original = env.observe(0)
        rotated = copy.deepcopy(original)
        rotated["player_id"] = 1
        rotated["current_player"] = (original["current_player"] + 1) % 4
        rotated["prevalent_wind"] = (original["prevalent_wind"] + 1) % 4
        rotated["melds"] = original["melds"][-1:] + original["melds"][:-1]
        rotated["discards"] = original["discards"][-1:] + original["discards"][:-1]
        rotated["wall_remaining_by_player"] = (
            original["wall_remaining_by_player"][-1:] +
            original["wall_remaining_by_player"][:-1])
        rotated["events"] = [
            tuple([event[0]] + [
                (value + 1) % 4 if index == 1 and isinstance(value, int) else value
                for index, value in enumerate(event[1:], 1)
            ]) for event in original["events"]
        ]
        self.assertEqual(encode_observation_v2(original), encode_observation_v2(rotated))


if __name__ == "__main__":
    unittest.main()
