"""Dependency-free hybrid feature encoder."""

from collections import Counter

from mahjong_agent.engine.actions import ActionType

FEATURE_SIZE = 34 + 34 + 4 * 34 + 4 * 34 + 8
ACTION_SIZE = 8


def _counts(tiles):
    counter = Counter(tile for tile in tiles if 0 <= tile < 34)
    return [float(counter.get(tile, 0)) / 4.0 for tile in range(34)]


def encode_observation(observation, rules=None):
    values = []
    values.extend(_counts(observation["hand"]))
    visible = []
    for river in observation["discards"]:
        visible.extend(river)
    for melds in observation["melds"]:
        for meld in melds:
            visible.extend(meld.tiles)
    values.extend(_counts(visible))
    for player in range(4):
        values.extend(_counts(observation["discards"][player]))
    for player in range(4):
        meld_tiles = []
        for meld in observation["melds"][player]:
            meld_tiles.extend(meld.tiles)
        values.extend(_counts(meld_tiles))
    player_id = observation["player_id"]
    current = observation["current_player"]
    phase = observation["phase"]
    values.extend([
        player_id / 3.0,
        current / 3.0,
        1.0 if player_id == current else 0.0,
        1.0 if phase == "discard" else 0.0,
        1.0 if phase == "claim" else 0.0,
        min(observation["wall_remaining"], 136) / 136.0,
        min(len(observation["events"]), 128) / 128.0,
        1.0 if observation["last_discard"] else 0.0,
    ])
    return values


def encode_action(action):
    return [
        int(action.kind) / float(len(ActionType) - 1),
        (action.tile + 1) / 34.0,
        (action.discard + 1) / 34.0,
        (action.sequence[0] + 1) / 34.0 if action.sequence else 0.0,
        (action.sequence[-1] + 1) / 34.0 if action.sequence else 0.0,
        1.0 if action.kind == ActionType.HU else 0.0,
        1.0 if action.kind in (ActionType.CHI, ActionType.PENG, ActionType.GANG) else 0.0,
        1.0 if action.kind == ActionType.PLAY else 0.0,
    ]
