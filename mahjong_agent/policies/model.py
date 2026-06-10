from mahjong_agent.features import (encode_action, encode_action_v2,
                                    encode_observation, encode_observation_v2)


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
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            version_owner = getattr(self.model, "module", self.model)
            if int(getattr(version_owner, "feature_version", 1)) == 2:
                feature, feature_mask = encode_observation_v2(observation)
                encoded = [encode_action_v2(action) for action in legal_actions]
                actions = [item[0] for item in encoded]
                action_token_mask = [item[1] for item in encoded]
                output = self.model(
                    torch.tensor([feature], dtype=torch.float32, device=device),
                    torch.tensor([actions], dtype=torch.float32, device=device),
                    feature_mask=torch.tensor([feature_mask], dtype=torch.bool, device=device),
                    action_token_mask=torch.tensor(
                        [action_token_mask], dtype=torch.bool, device=device))
            else:
                features = torch.tensor(
                    [encode_observation(observation)], dtype=torch.float32, device=device)
                actions = torch.tensor(
                    [[encode_action(a) for a in legal_actions]], dtype=torch.float32, device=device)
                output = self.model(features, actions)
            logits = output["logits"][0]
            if self.stochastic:
                index = torch.distributions.Categorical(logits=logits).sample().item()
            else:
                index = logits.argmax().item()
        return legal_actions[index]
