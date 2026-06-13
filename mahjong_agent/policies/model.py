from mahjong_agent.features import (encode_action, encode_action_v2,
                                    encode_observation, encode_observation_v2)
from mahjong_agent.engine.actions import ActionType


class ModelPolicy(object):
    def __init__(self, model, stochastic=False):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("ModelPolicy requires PyTorch") from exc
        # model 是 V1 HybridTransformer 或 V2 TokenTransformer；stochastic 控制采样/贪心。
        self.torch = torch
        self.model = model
        self.stochastic = stochastic
        self.model.eval()

    def act(self, observation, legal_actions):
        # 单局接口复用 batch_act，返回一个 Action。
        return self.batch_act([observation], [legal_actions])[0]

    def batch_act(self, observations, legal_actions_batch):
        # observations 长度为 B；legal_actions_batch 每行是长度可变的 list[Action]。
        if not observations:
            return []
        # 合法动作含 HU 时直接和牌，不进入神经网络。
        forced = {}
        pending = []
        for row, legal_actions in enumerate(legal_actions_batch):
            hu = next((action for action in legal_actions if action.kind == ActionType.HU), None)
            if hu is not None:
                forced[row] = hu
            else:
                pending.append(row)
        result = [None] * len(observations)
        for row, action in forced.items():
            result[row] = action
        if not pending:
            return result
        torch = self.torch
        self.model.eval()
        device = next(self.model.parameters()).device
        # 推理不构建梯度图；device 跟随模型参数。
        with torch.no_grad():
            version_owner = getattr(self.model, "module", self.model)
            if int(getattr(version_owner, "feature_version", 1)) == 2:
                # V2 features: [B,256,12]；actions padding 后为 [B,A,4,12]。
                encoded_features = [encode_observation_v2(observations[row]) for row in pending]
                encoded_actions = [[encode_action_v2(action) for action in legal_actions_batch[row]]
                                   for row in pending]
                # A 是当前 batch 中最大的合法动作数，较短行以全零动作 padding。
                maximum = max(len(items) for items in encoded_actions)
                zero_action = [[0.0] * 12 for _ in range(4)]
                actions, action_token_mask, action_mask = [], [], []
                for items in encoded_actions:
                    count = len(items)
                    actions.append([item[0] for item in items] +
                                   [zero_action] * (maximum - count))
                    action_token_mask.append([item[1] for item in items] +
                                             [[0] * 4] * (maximum - count))
                    action_mask.append([1] * count + [0] * (maximum - count))
                output = self.model(
                    torch.tensor([item[0] for item in encoded_features],
                                 dtype=torch.float32, device=device),
                    torch.tensor(actions, dtype=torch.float32, device=device),
                    action_mask=torch.tensor(action_mask, dtype=torch.bool, device=device),
                    feature_mask=torch.tensor([item[1] for item in encoded_features],
                                              dtype=torch.bool, device=device),
                    action_token_mask=torch.tensor(
                        action_token_mask, dtype=torch.bool, device=device))
            else:
                # V1 features: [B,FEATURE_SIZE]；actions: [B,A,ACTION_SIZE]。
                features = torch.tensor(
                    [encode_observation(observations[row]) for row in pending],
                    dtype=torch.float32, device=device)
                maximum = max(len(legal_actions_batch[row]) for row in pending)
                action_size = len(encode_action(legal_actions_batch[pending[0]][0]))
                encoded = []
                masks = []
                for row in pending:
                    items = [encode_action(a) for a in legal_actions_batch[row]]
                    encoded.append(items + [[0.0] * action_size] * (maximum - len(items)))
                    masks.append([1] * len(items) + [0] * (maximum - len(items)))
                output = self.model(
                    features, torch.tensor(encoded, dtype=torch.float32, device=device),
                    torch.tensor(masks, dtype=torch.bool, device=device))
            # logits shape=[待决策局数,A]；mask 已保证不会选到 padding 动作。
            logits = output["logits"]
            if self.stochastic:
                indices = torch.distributions.Categorical(logits=logits).sample().tolist()
            else:
                indices = logits.argmax(-1).tolist()
        for row, index in zip(pending, indices):
            result[row] = legal_actions_batch[row][index]
        return result
