import unittest

from mahjong_agent.engine.actions import Action
from mahjong_agent.policies import HeuristicPolicy
from mahjong_agent.policies.analysis import action_deal_in_risk, direct_deal_in_index
from mahjong_agent.training.ppo import (generalized_advantage_estimate,
                                       potential_shaped_rewards, ppo_update,
                                       rollout_game_indices,
                                       terminal_only_rewards)
from mahjong_agent.training.rollout import play_episode, play_episodes_vectorized


class PPOTest(unittest.TestCase):
    def test_global_rollout_budget_is_partitioned_across_ranks(self):
        first = list(rollout_game_indices(256, rank=0, world_size=2))
        second = list(rollout_game_indices(256, rank=1, world_size=2))
        self.assertEqual(len(first), 128)
        self.assertEqual(len(second), 128)
        self.assertEqual(sorted(first + second), list(range(256)))

    def test_terminal_only_reward_has_no_intermediate_shaping(self):
        self.assertEqual(terminal_only_rewards(3, 0.75), [0.0, 0.0, 0.75])
        self.assertEqual(terminal_only_rewards(0, 0.75), [])

    def test_action_risk_is_zero_for_opponent_safe_tile(self):
        observation = {
            "player_id": 0, "hand": [0, 1, 2], "melds": [[], [], [], []],
            "discards": [[], [3], [3], [3]],
        }
        self.assertEqual(action_deal_in_risk(observation, Action.play(3)), 0.0)

    def test_opponent_pool_falls_back_to_bc_without_ppo_snapshots(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from mahjong_agent.models import create_model
        from scripts.train_ppo import OpponentPool
        pool = OpponentPool(create_model(2, d_model=48, layers=1, heads=6), seed=3)
        counts = {}
        for index in range(200):
            _, name = pool.sample(index)
            counts[name] = counts.get(name, 0) + 1
        self.assertNotIn("ppo_latest", counts)
        self.assertNotIn("ppo_best", counts)
        self.assertNotIn("ppo_history", counts)
        self.assertGreater(counts["bc"], counts["heuristic"])

    def test_opponent_pool_exposes_latest_best_and_history_snapshots(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from mahjong_agent.models import create_model
        from scripts.train_ppo import OpponentPool
        model = create_model(2, d_model=48, layers=1, heads=6)
        pool = OpponentPool(model, seed=3, mix={"ppo_latest": 1.0})
        snapshot = pool.add_history(model)
        _, name = pool.sample(1)
        self.assertEqual(name, "ppo_latest")
        self.assertTrue(pool.update_best(snapshot, 1.0))
        pool.mix = {"ppo_best": 1.0}
        self.assertEqual(pool.sample(2)[1], "ppo_best")
        pool.mix = {"ppo_history": 1.0}
        self.assertEqual(pool.sample(3)[1], "ppo_history")

    def test_direct_deal_in_marks_last_discard_action_only(self):
        records = [{"action": Action.play(1)}, {"action": Action.pass_()}]
        self.assertEqual(direct_deal_in_index(records, {"loser": 0}, 0), 0)
        self.assertEqual(direct_deal_in_index(records, {"loser": 1}, 0), -1)

    def test_vectorized_rollout_matches_serial_fixed_seed(self):
        policies = [HeuristicPolicy() for _ in range(4)]
        serial, _ = play_episode(policies, seed=17)
        vector, _ = play_episodes_vectorized([policies], [17])[0]
        self.assertEqual(serial["scores"], vector["scores"])
        self.assertEqual(serial["fan_count"], vector["fan_count"])

    def test_ppo_auxiliary_heads_receive_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from mahjong_agent.models import create_model
        model = create_model(2, d_model=48, layers=1, heads=6)
        features = torch.zeros(2, 4, 12)
        actions = torch.zeros(2, 2, 4, 12)
        masks = torch.ones(2, 2, dtype=torch.bool)
        feature_masks = torch.ones(2, 4, dtype=torch.bool)
        action_token_masks = torch.ones(2, 2, 4, dtype=torch.bool)
        chosen = torch.tensor([0, 1])
        with torch.no_grad():
            logits = model(features, actions, masks, feature_masks,
                           action_token_masks)["logits"]
        batch = {
            "features": features, "actions": actions, "masks": masks,
            "feature_masks": feature_masks, "action_token_masks": action_token_masks,
            "chosen": chosen,
            "old_log_probs": torch.distributions.Categorical(logits=logits).log_prob(chosen),
            "advantages": torch.tensor([1.0, -1.0]), "returns": torch.zeros(2),
            "reference_logits": logits, "aux_labels": torch.zeros(2, 4),
            "action_aux_labels": torch.zeros(2, 4),
            "fan_targets": torch.zeros(2, dtype=torch.long),
            "belief_targets": torch.zeros(2, 3, 34, dtype=torch.long),
        }
        ppo_update(model, torch.optim.SGD(model.parameters(), lr=.01), batch,
                   epochs=1, minibatch_size=2, aux_coef=.1)
        self.assertIsNotNone(model.action_outcome_head.weight.grad)
        self.assertIsNotNone(model.action_fan_head.weight.grad)
        self.assertIsNotNone(model.belief_head.weight.grad)

    def test_gae_propagates_terminal_reward(self):
        advantages, returns = generalized_advantage_estimate([0.0, 1.0], [0.0, 0.0], 1.0, 1.0)
        self.assertEqual(advantages, [1.0, 1.0])
        self.assertEqual(returns, [1.0, 1.0])

    def test_potential_shaping_returns_to_zero_at_terminal(self):
        rewards = potential_shaped_rewards([2.0, 3.0], 1.0, gamma=1.0, coefficient=0.5)
        self.assertEqual(rewards, [0.5, -0.5])

    def test_ppo_reports_reference_policy_kl(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super(TinyPolicy, self).__init__()
                self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.2]))
                self.value = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, features, actions, masks):
                batch = features.size(0)
                return {
                    "logits": self.logits.unsqueeze(0).expand(batch, 2),
                    "value": self.value.expand(batch),
                    "aux": self.value.expand(batch, 1),
                }

        model = TinyPolicy()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        chosen = torch.tensor([0, 1])
        logits = model.logits.detach().unsqueeze(0).expand(2, 2)
        old_log_probs = torch.distributions.Categorical(logits=logits).log_prob(chosen)
        batch = {
            "features": torch.zeros(2, 1), "actions": torch.zeros(2, 2, 1),
            "masks": torch.ones(2, 2, dtype=torch.bool), "chosen": chosen,
            "old_log_probs": old_log_probs, "advantages": torch.tensor([1.0, -1.0]),
            "returns": torch.zeros(2), "reference_logits": torch.zeros(2, 2),
        }
        metrics = ppo_update(model, optimizer, batch, epochs=1, minibatch_size=1)
        self.assertIn("bc_kl", metrics)
        self.assertEqual(metrics["samples"], 2)

    def test_ppo_update_disables_dropout_noise(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")

        class DropoutPolicy(torch.nn.Module):
            def __init__(self):
                super(DropoutPolicy, self).__init__()
                self.weight = torch.nn.Parameter(torch.ones(2))
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, features, actions, masks):
                logits = self.dropout(self.weight).unsqueeze(0).expand(features.size(0), 2)
                value = self.weight[0].expand(features.size(0))
                return {"logits": logits, "value": value, "aux": value.unsqueeze(1)}

        model = DropoutPolicy()
        logits = model.weight.detach().unsqueeze(0).expand(2, 2)
        chosen = torch.tensor([0, 1])
        batch = {
            "features": torch.zeros(2, 1), "actions": torch.zeros(2, 2, 1),
            "masks": torch.ones(2, 2, dtype=torch.bool), "chosen": chosen,
            "old_log_probs": torch.distributions.Categorical(logits=logits).log_prob(chosen),
            "advantages": torch.tensor([1.0, -1.0]), "returns": torch.zeros(2),
            "reference_logits": logits,
        }
        ppo_update(model, torch.optim.SGD(model.parameters(), lr=0.0), batch,
                   epochs=1, minibatch_size=2)
        self.assertFalse(model.dropout.training)


if __name__ == "__main__":
    unittest.main()
