#!/usr/bin/env python3
"""Botzone JSON-lines entry point.

Defaults to the dependency-free heuristic policy. A model policy can be wired
in after packaging a verified PyTorch runtime and model.pt.
"""
import json
import sys

from mahjong_agent.botzone.protocol import ProtocolState, action_to_text
from mahjong_agent.engine.actions import Action, ActionType
from mahjong_agent.engine.tiles import is_suited
from mahjong_agent.policies import HeuristicPolicy
from mahjong_agent.rules import default_backend


def legal_from_protocol(state):
    observation = state.observation()
    hand = observation["hand"]
    if observation["phase"] == "discard":
        actions = [Action.play(tile) for tile in sorted(set(hand))]
        counts = [hand.count(tile) for tile in range(34)]
        if default_backend.can_hu(counts):
            actions.append(Action.hu())
        return actions
    actions = [Action.pass_()]
    if not observation["last_discard"]:
        return actions
    source, tile = observation["last_discard"]
    counts = [hand.count(item) for item in range(34)]
    counts[tile] += 1
    if default_backend.can_hu(counts, win_tile=tile):
        actions.append(Action.hu())
    counts[tile] -= 1
    if hand.count(tile) >= 2:
        remaining = list(hand)
        remaining.remove(tile)
        remaining.remove(tile)
        for discard in sorted(set(remaining)):
            actions.append(Action(ActionType.PENG, tile, (), discard))
    if hand.count(tile) >= 3:
        actions.append(Action(ActionType.GANG, tile))
    if state.player_id == (source + 1) % 4 and is_suited(tile):
        base = tile - tile % 9
        offset = tile % 9
        for start in range(max(0, offset - 2), min(6, offset) + 1):
            sequence = (base + start, base + start + 1, base + start + 2)
            needed = list(sequence)
            needed.remove(tile)
            if all(hand.count(item) >= needed.count(item) for item in set(needed)):
                remaining = list(hand)
                for item in needed:
                    remaining.remove(item)
                for discard in sorted(set(remaining)):
                    actions.append(Action(ActionType.CHI, tile, sequence, discard))
    return actions


def main():
    state = ProtocolState()
    payload = json.loads(sys.stdin.readline())
    for request in payload.get("requests", []):
        state.apply(request)
    legal = legal_from_protocol(state)
    action = HeuristicPolicy().act(state.observation(), legal)
    print(json.dumps({"response": action_to_text(action)}))


if __name__ == "__main__":
    main()
