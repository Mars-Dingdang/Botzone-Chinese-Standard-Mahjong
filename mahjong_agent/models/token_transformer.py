"""Feature V2 public-token actor-critic with auxiliary prediction heads."""

import torch
from torch import nn

from mahjong_agent.features.token_encoder import TOKEN_SIZE


class TokenTransformer(nn.Module):
    feature_version = 2

    def __init__(self, token_size=TOKEN_SIZE, d_model=192, layers=4, heads=6,
                 dropout=0.1, belief_mode="aux"):
        super(TokenTransformer, self).__init__()
        self.token_size = token_size
        self.d_model = d_model
        self.belief_mode = belief_mode
        self.model_config = {
            "token_size": token_size, "d_model": d_model, "layers": layers,
            "heads": heads, "dropout": dropout, "belief_mode": belief_mode,
        }
        self.state_encoder = nn.Linear(token_size, d_model)
        self.action_encoder = nn.Linear(token_size, d_model)
        self.kind_embedding = nn.Embedding(16, d_model)
        self.tile_embedding = nn.Embedding(36, d_model)
        self.player_embedding = nn.Embedding(8, d_model)
        self.action_embedding = nn.Embedding(16, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            dropout=dropout)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.outcome_head = nn.Linear(d_model, 4)  # win, deal-in, score, 8-fan
        self.fan_head = nn.Linear(d_model, 5)
        self.belief_head = nn.Linear(d_model, 3 * 34 * 5)
        self.belief_adapter = nn.Linear(3 * 34 * 5, d_model)

    def _embed(self, tokens, encoder):
        continuous = encoder(tokens)
        return (continuous
                + self.kind_embedding(tokens[..., 0].long().clamp(0, 15))
                + self.tile_embedding(tokens[..., 1].long().clamp(0, 35))
                + self.player_embedding(tokens[..., 2].long().clamp(0, 7))
                + self.action_embedding(tokens[..., 3].long().clamp(0, 15)))

    def forward(self, features, actions, action_mask=None, feature_mask=None,
                action_token_mask=None):
        state_tokens = self._embed(features, self.state_encoder)
        encoded = self.transformer(
            state_tokens.transpose(0, 1),
            src_key_padding_mask=(feature_mask == 0) if feature_mask is not None else None,
        ).transpose(0, 1)
        if feature_mask is None:
            state = encoded.mean(1)
        else:
            weight = feature_mask.float().unsqueeze(-1)
            state = (encoded * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        initial_belief = self.belief_head(state).view(-1, 3, 34, 5)
        if self.belief_mode == "actor":
            probabilities = initial_belief.softmax(-1).detach()
            state = state + self.belief_adapter(probabilities.reshape(state.size(0), -1))
        action_encoded = self._embed(actions, self.action_encoder)
        if action_token_mask is None:
            action_state = action_encoded.mean(2)
        else:
            weight = action_token_mask.float().unsqueeze(-1)
            action_state = (action_encoded * weight).sum(2) / weight.sum(2).clamp_min(1.0)
        expanded = state.unsqueeze(1).expand_as(action_state)
        logits = self.policy_head(torch.cat((expanded, action_state), -1)).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e4)
        outcome = self.outcome_head(state)
        return {
            "logits": logits, "value": self.value_head(state).squeeze(-1),
            "aux": outcome, "outcome": outcome, "fan_logits": self.fan_head(state),
            "belief_logits": self.belief_head(state).view(-1, 3, 34, 5),
        }
