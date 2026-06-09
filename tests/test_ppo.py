import unittest

from mahjong_agent.training.ppo import generalized_advantage_estimate, potential_shaped_rewards, ppo_update


class PPOTest(unittest.TestCase):
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
