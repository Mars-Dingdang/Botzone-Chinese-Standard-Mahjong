#!/usr/bin/env python3
"""Botzone JSON-lines entry point.

Loads ``data/model.pt`` when the Botzone runtime provides PyTorch, otherwise
falls back to the dependency-free heuristic policy.
"""
import json
import os
import sys

from mahjong_agent.botzone.legality import sanitize_action, strict_legal_actions
from mahjong_agent.botzone.protocol import ProtocolState, action_to_text
from mahjong_agent.policies import HeuristicPolicy


def load_policy(model_path=None):
    model_path = model_path or os.environ.get("MAHJONG_MODEL", os.path.join("data", "model.pt"))
    try:
        import torch
        from mahjong_agent.policies.model import ModelPolicy
        from mahjong_agent.training.checkpoint import load_model_from_checkpoint
        model, _ = load_model_from_checkpoint(model_path)
        model.to(torch.device("cpu"))
        return ModelPolicy(model)
    except Exception:
        if os.environ.get("MAHJONG_REQUIRE_MODEL") == "1":
            raise
        return HeuristicPolicy()


def main():
    state = ProtocolState()
    payload = json.loads(sys.stdin.readline())
    requests = payload.get("requests", [])
    responses = payload.get("responses", [])
    for index, request in enumerate(requests):
        previous_response = responses[index - 1] if index and index - 1 < len(responses) else None
        state.apply(request, previous_response)
    legal = strict_legal_actions(state)
    proposed = legal[0] if len(legal) == 1 else load_policy().act(state.observation(), legal)
    action, reason = sanitize_action(state, proposed)
    if reason:
        sys.stderr.write("sanitized action: %s\n" % reason)
    print(json.dumps({"response": action_to_text(action, state)}))


if __name__ == "__main__":
    main()
