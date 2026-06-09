"""PyTorch 1.4-compatible candidate-action Transformer scorer."""

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("HybridTransformer requires PyTorch") from exc

from mahjong_agent.features.encoder import ACTION_SIZE, FEATURE_SIZE


class HybridTransformer(nn.Module):
    def __init__(self, feature_size=FEATURE_SIZE, action_size=ACTION_SIZE,
                 d_model=192, layers=4, heads=6, dropout=0.1):
        super(HybridTransformer, self).__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        self.d_model = d_model
        self.state_encoder = nn.Sequential(
            nn.Linear(feature_size, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 4),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.aux_head = nn.Linear(d_model, 3)

    def forward(self, features, actions, action_mask=None):
        batch = features.size(0)
        state_tokens = self.state_encoder(features).view(batch, 4, self.d_model)
        encoded = self.transformer(state_tokens.transpose(0, 1)).transpose(0, 1)
        state = encoded.mean(dim=1)
        action_embeddings = self.action_encoder(actions)
        expanded_state = state.unsqueeze(1).expand(
            batch, action_embeddings.size(1), self.d_model
        )
        logits = self.policy_head(
            torch.cat([expanded_state, action_embeddings], dim=-1)
        ).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        return {
            "logits": logits,
            "value": self.value_head(state).squeeze(-1),
            "aux": self.aux_head(state),
        }
