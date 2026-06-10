"""Public-information-only token encoder for Feature V2."""

from collections import Counter

from mahjong_agent.engine.actions import ActionType

FEATURE_VERSION = 2
TOKEN_SIZE = 12
MAX_STATE_TOKENS = 256
MAX_ACTION_TOKENS = 4

TOKEN_GLOBAL = 1
TOKEN_HAND = 2
TOKEN_DISCARD = 3
TOKEN_MELD = 4
TOKEN_EVENT = 5
TOKEN_UNSEEN = 6
TOKEN_ACTION = 7

EVENT_IDS = {
    "DRAW": 1, "PLAY": 2, "CHI": 3, "PENG": 4, "GANG": 5,
    "ANGANG": 6, "BUGANG": 7, "HU": 8, "BUHUA": 9,
}


def _relative(player, observer):
    return (int(player) - int(observer)) % 4


def _token(kind, tile=-1, player=-1, action=-1, order=0, flags=0,
           value=0.0, extra=0.0):
    return [
        float(kind), float(tile + 1), float(player + 1), float(action + 1),
        float(order) / 128.0, float(flags), float(value), float(extra),
        0.0, 0.0, 0.0, 0.0,
    ]


def _pad(tokens, maximum):
    tokens = tokens[:maximum]
    mask = [1] * len(tokens)
    while len(tokens) < maximum:
        tokens.append([0.0] * TOKEN_SIZE)
        mask.append(0)
    return tokens, mask


def encode_observation_v2(observation):
    observer = int(observation["player_id"])
    last = observation.get("last_discard")
    phase = {"ack": 0, "draw": 1, "discard": 2, "claim": 3, "terminal": 4}.get(
        observation.get("phase"), 0)
    flags = (
        int(bool(observation.get("wall_last"))) |
        int(bool(observation.get("about_kong"))) << 1 |
        int(bool(observation.get("claim_hu_only"))) << 2
    )
    tokens = [_token(
        TOKEN_GLOBAL, tile=last[1] if last else -1,
        player=_relative(observation.get("current_player", observer), observer),
        action=phase, flags=flags,
        value=float(observation.get("wall_remaining", 0)) / 84.0,
        extra=float(_relative(observation.get("prevalent_wind", 0), observer)) / 3.0,
    )]
    for order, tile in enumerate(observation["hand"]):
        tokens.append(_token(TOKEN_HAND, tile=tile, player=0, order=order))

    claimed = Counter()
    for owner, melds in enumerate(observation["melds"]):
        for meld in melds:
            if meld.from_player != -1 and meld.from_player != owner:
                claimed[(meld.from_player, meld.tiles[-1])] += 1
    for relative in range(4):
        player = (observer + relative) % 4
        river = observation["discards"][player]
        consumed = Counter(claimed)
        for order, tile in enumerate(river):
            used = consumed[(player, tile)] > 0
            if used:
                consumed[(player, tile)] -= 1
            tokens.append(_token(
                TOKEN_DISCARD, tile=tile, player=_relative(player, observer),
                order=order, flags=int(used)))

    for relative in range(4):
        player = (observer + relative) % 4
        melds = observation["melds"][player]
        for order, meld in enumerate(melds):
            for tile_order, tile in enumerate(meld.tiles):
                tokens.append(_token(
                    TOKEN_MELD, tile=tile, player=_relative(player, observer),
                    action=int(meld.kind), order=order,
                    flags=_relative(meld.from_player, observer),
                    extra=float(tile_order) / 4.0))

    for order, event in enumerate(observation.get("events", [])[-64:]):
        if not event:
            continue
        kind = str(event[0]).upper()
        player = event[1] if len(event) > 1 and isinstance(event[1], int) else observer
        tile = next((item for item in event[2:] if isinstance(item, int) and 0 <= item < 34), -1)
        tokens.append(_token(
            TOKEN_EVENT, tile=tile, player=_relative(player, observer),
            action=EVENT_IDS.get(kind, 0), order=order))

    visible = Counter(observation["hand"])
    for river in observation["discards"]:
        visible.update(river)
    for melds in observation["melds"]:
        for meld in melds:
            visible.update(meld.tiles)
    for tile in range(34):
        tokens.append(_token(
            TOKEN_UNSEEN, tile=tile, value=max(0, 4 - visible[tile]) / 4.0))
    for relative in range(4):
        player = (observer + relative) % 4
        meld_tiles = sum(len(meld.tiles) for meld in observation["melds"][player])
        concealed = 13 - (meld_tiles // 3) * 3
        if player == observation.get("current_player") and observation.get("phase") == "discard":
            concealed += 1
        tokens.append(_token(
            TOKEN_GLOBAL, player=_relative(player, observer),
            value=max(0, concealed) / 14.0,
            extra=float(observation.get("wall_remaining_by_player", [0] * 4)[player]) / 21.0))
    return _pad(tokens, MAX_STATE_TOKENS)


def encode_action_v2(action):
    tiles = list(action.sequence) if action.sequence else ([action.tile] if action.tile >= 0 else [])
    tokens = [_token(
        TOKEN_ACTION, tile=action.tile, action=int(action.kind),
        flags=int(action.kind == ActionType.HU), extra=(action.discard + 1) / 34.0)]
    for order, tile in enumerate(tiles[:3]):
        tokens.append(_token(TOKEN_ACTION, tile=tile, action=int(action.kind), order=order + 1))
    return _pad(tokens, MAX_ACTION_TOKENS)
