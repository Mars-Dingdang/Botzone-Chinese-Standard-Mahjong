import unittest

from mahjong_agent.engine import Action, ActionType, MahjongEnv
from mahjong_agent.engine.actions import Meld
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

    def test_claims_are_collected_before_priority_resolution(self):
        env = MahjongEnv()
        env.reset(seed=1)
        env.hands[1] = [0, 0] + list(range(1, 12))
        env.hands[2] = [0, 0] + list(range(1, 12))
        env.current_player = 0
        env.phase = "discard"
        env.hands[0] = [0] + list(range(1, 14))
        env.step(Action.play(0))
        peng = next(action for action in env.legal_actions() if action.kind == ActionType.PENG)
        env.step(peng)
        self.assertEqual(env.current_player, 2)
        self.assertEqual(env.phase, "claim")
        env.step(Action.pass_())
        env.step(Action.pass_())
        self.assertEqual(env.current_player, 2)
        self.assertEqual(env.phase, "claim")
        self.assertEqual(env.melds[1][0].kind, ActionType.PENG)

    def test_bugang_waits_for_rob_kong_responses(self):
        env = MahjongEnv()
        env.reset(seed=1)
        env.current_player = 0
        env.phase = "discard"
        env.hands[0] = [0] + list(range(1, 14))
        env.melds[0] = [Meld(ActionType.PENG, (0, 0, 0), 1)]
        env.step(Action(ActionType.BUGANG, 0))
        self.assertEqual(env.phase, "claim")
        self.assertTrue(env.claim_hu_only)
        self.assertEqual(env.hands[0].count(0), 1)
        for _ in range(3):
            env.step(Action.pass_())
        self.assertFalse(env.claim_hu_only)
        self.assertEqual(env.melds[0][0].kind, ActionType.GANG)

    def test_wall_last_forbids_claims_and_kongs(self):
        env = MahjongEnv()
        env.reset(seed=1)
        env.wall = []
        env.current_player = 1
        env.phase = "claim"
        env.last_discard = (0, 0)
        env.hands[1] = [0, 0, 0] + list(range(1, 11))
        self.assertEqual([action.kind for action in env.legal_actions()], [ActionType.PASS])


    def test_resolved_chi_keeps_sorted_sequence(self):
        env = MahjongEnv()
        env.reset(seed=1)
        env.current_player = 0
        env.phase = "discard"
        env.hands[0] = [1] + list(range(2, 15))
        env.hands[1] = [0, 2] + list(range(3, 14))
        env.step(Action.play(1))
        chi = next(action for action in env.legal_actions()
                   if action.kind == ActionType.CHI and action.sequence == (0, 1, 2))
        env.step(chi)
        env.step(Action.pass_())
        env.step(Action.pass_())
        self.assertEqual(env.melds[1][0].tiles, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
