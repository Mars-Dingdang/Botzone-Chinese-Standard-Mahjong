import json
import os
import tempfile
import unittest

from mahjong_agent.engine.actions import Action, ActionType, Meld
from mahjong_agent.policies.baseline import HeuristicPolicy
from scripts.preprocess_official_full_actions import (FullActionState,
                                                      terminal_aux_labels)


class BCPipelineTest(unittest.TestCase):
    def test_non_winner_does_not_inherit_winner_fan(self):
        winner = terminal_aux_labels(2, 2, 1, [-8, -8, 32, -16], 24)
        other = terminal_aux_labels(0, 2, 1, [-8, -8, 32, -16], 24)
        self.assertEqual((winner["fan_bucket"], winner["eight_fan"]), (3, 1))
        self.assertEqual((other["fan_bucket"], other["eight_fan"]), (0, 0))

    def test_official_replay_upgrades_bugang_meld(self):
        state = FullActionState()
        state.reset()
        state.hands[1] = [3]
        state.melds[1] = [Meld(ActionType.PENG, (3, 3, 3), 0)]
        state.apply(["Player", "1", "BuGang", "W4"])
        self.assertEqual(state.melds[1][0].kind, ActionType.GANG)
        self.assertEqual(state.melds[1][0].tiles, (3, 3, 3, 3))
        self.assertTrue(state.about_kong)
        self.assertTrue(state.claim_hu_only)
        self.assertEqual(set(state.pending), {0, 2, 3})

    def test_baseline_simulates_chi_and_discard(self):
        observation = {
            "player_id": 0, "phase": "claim", "last_discard": (3, 1),
            "hand": [0, 2, 4, 5, 6], "melds": [[], [], [], []],
            "discards": [[], [], [], [1]],
        }
        action = Action(ActionType.CHI, 1, (0, 1, 2), 4)
        hand, melds = HeuristicPolicy._simulate(observation, action)
        self.assertEqual(hand, [5, 6])
        self.assertEqual(melds[0].tiles, (0, 1, 2))

    def test_tensor_batches_shuffle_deterministically_by_epoch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from mahjong_agent.training.dataset import iter_tensor_batches
        with tempfile.TemporaryDirectory() as directory:
            n = 12
            payload = {
                "features": torch.arange(n).view(n, 1, 1).half(),
                "feature_masks": torch.ones(n, 1, dtype=torch.bool),
                "actions": torch.zeros(n, 1, 1, 1).half(),
                "action_token_masks": torch.ones(n, 1, 1, dtype=torch.bool),
                "masks": torch.ones(n, 1, dtype=torch.bool),
                "targets": torch.zeros(n, dtype=torch.long),
                "aux_labels": torch.zeros(n, 4),
                "fan_targets": torch.zeros(n, dtype=torch.long),
                "belief_targets": torch.zeros(n, 3, 34, dtype=torch.long),
            }
            torch.save(payload, os.path.join(directory, "train.pt"))
            with open(os.path.join(directory, "tensor_metadata.json"), "w") as handle:
                json.dump({"shards": [{"path": "train.pt", "split": "train",
                                       "samples": n}]}, handle)

            def order(epoch):
                return torch.cat([batch[0][:, 0, 0] for batch in iter_tensor_batches(
                    directory, "train", 4, seed=9, epoch=epoch, shuffle=True)]).tolist()

            self.assertEqual(order(0), order(0))
            self.assertNotEqual(order(0), order(1))

    def test_tensor_shards_rotate_between_ranks(self):
        from mahjong_agent.training.dataset import tensor_shard_plan
        with tempfile.TemporaryDirectory() as directory:
            shards = [{"path": "s%d.pt" % index, "split": "train", "samples": 100}
                      for index in range(4)]
            with open(os.path.join(directory, "tensor_metadata.json"), "w") as handle:
                json.dump({"shards": shards}, handle)
            first, _ = tensor_shard_plan(directory, "train", 2, 10, seed=5, epoch=0)
            second, _ = tensor_shard_plan(directory, "train", 2, 10, seed=5, epoch=1)
            self.assertNotEqual({item["path"] for item in first[0]},
                                {item["path"] for item in second[0]})
            self.assertTrue({item["path"] for item in first[0]}.isdisjoint(
                {item["path"] for item in first[1]}))


if __name__ == "__main__":
    unittest.main()
