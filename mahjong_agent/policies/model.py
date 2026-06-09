from mahjong_agent.features import encode_action, encode_observation


class ModelPolicy(object):
    def __init__(self, model, stochastic=False):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("ModelPolicy requires PyTorch") from exc
        self.torch = torch
        self.model = model
        self.stochastic = stochastic
        self.model.eval()

    def act(self, observation, legal_actions):
        torch = self.torch
        device = next(self.model.parameters()).device
        with torch.no_grad():
            features = torch.tensor(
                [encode_observation(observation)], dtype=torch.float32, device=device
            )
            actions = torch.tensor(
                [[encode_action(a) for a in legal_actions]], dtype=torch.float32, device=device
            )
            logits = self.model(features, actions)["logits"][0]
            if self.stochastic:
                index = torch.distributions.Categorical(logits=logits).sample().item()
            else:
                index = logits.argmax().item()
        return legal_actions[index]
