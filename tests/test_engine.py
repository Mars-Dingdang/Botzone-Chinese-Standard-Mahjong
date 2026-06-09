import unittest

from mahjong_agent.engine import Action, ActionType, MahjongEnv
from mahjong_agent.engine.tiles import full_wall, name_to_tile, tile_to_name
from mahjong_agent.policies import RandomPolicy
from mahjong_agent.training.rollout import play_episode


class EngineTest(unittest.TestCase):
    def test_tile_round_trip(self):
        for tile in range(34):
            self.assertEqual(name_to_tile(tile_to_name(tile)), tile)

    def test_reset_is_deterministic(self):
        first = MahjongEnv()
        second = MahjongEnv()
        first.reset(seed=7)
        second.reset(seed=7)
        self.assertEqual(first.hands, second.hands)
        self.assertEqual(first.wall, second.wall)

    def test_play_and_claim_cycle(self):
        env = MahjongEnv()
        env.reset(seed=2)
        source = env.current_player
        action = next(item for item in env.legal_actions() if item.kind == ActionType.PLAY)
        env.step(action)
        self.assertEqual(env.phase, "claim")
        for _ in range(3):
            env.step(Action.pass_())
        self.assertEqual(env.current_player, (source + 1) % 4)
        self.assertEqual(env.phase, "discard")

    def test_random_episode_has_no_invalid_actions(self):
        result, _ = play_episode([RandomPolicy(i) for i in range(4)], seed=3)
        self.assertEqual(result["invalid_actions"], 0)
        self.assertGreater(result["steps"], 0)

    def test_peng_discard_exists_after_claim(self):
        env = MahjongEnv()
        env.reset(seed=1)
        env.hands[1] = [0, 0] + list(range(1, 12))
        env.current_player = 1
        env.phase = "claim"
        env.last_discard = (0, 0)
        pengs = [action for action in env.legal_actions() if action.kind == ActionType.PENG]
        self.assertTrue(pengs)
        self.assertTrue(all(action.discard != 0 for action in pengs))


if __name__ == "__main__":
    unittest.main()
