"""Dependency-free hybrid feature encoder."""

from collections import Counter
from functools import lru_cache

from mahjong_agent.engine.actions import Action, ActionType, Meld
from mahjong_agent.rules import default_backend

EVENT_TYPES = ("DRAW", "PLAY", "CHI", "PENG", "GANG", "BUGANG", "HU")
# V1 状态向量由手牌34、可见牌34、四家弃牌136、四家副露136、有用牌34、
# 事件计数7和全局标量13拼接而成；动作向量固定为8维。
FEATURE_SIZE = 34 + 34 + 4 * 34 + 4 * 34 + 34 + len(EVENT_TYPES) + 13
ACTION_SIZE = 8


def _counts(tiles):
    # 输入是 tile id 列表；输出 shape=[34]，每种牌张数除以4归一化到 [0,1]。
    counter = Counter(tile for tile in tiles if 0 <= tile < 34)
    return [float(counter.get(tile, 0)) / 4.0 for tile in range(34)]


def serialize_meld(meld):
    # Meld -> 可 JSON 序列化的 [kind, from_player, tile0, ...]。
    return [int(meld.kind), int(meld.from_player)] + [int(tile) for tile in meld.tiles]


def deserialize_meld(data):
    # data 至少含两个整数，后续元素为副露中的牌。
    return Meld(ActionType(int(data[0])), tuple(int(tile) for tile in data[2:]), int(data[1]))


def serialize_action(action):
    # 固定长度6：[kind, tile, discard, seq0, seq1, seq2]，缺失顺子牌以 -1 padding。
    sequence = list(action.sequence)[:3]
    sequence += [-1] * (3 - len(sequence))
    return [int(action.kind), int(action.tile), int(action.discard)] + sequence


def deserialize_action(data):
    sequence = tuple(int(tile) for tile in data[3:6] if int(tile) >= 0)
    return Action(int(data[0]), int(data[1]), sequence, int(data[2]))


def compact_observation(observation):
    # 将含 namedtuple 的运行时 observation 转换为仅含 JSON 基础类型的 dict。
    events = []
    for event in observation["events"]:
        if not event:
            continue
        integers = [item for item in event[1:] if isinstance(item, int)]
        events.append({
            "kind": str(event[0]).upper(),
            "player": integers[0] if integers else -1,
            "tile": next((item for item in integers[1:] if 0 <= item < 34), -1),
            "extra": integers[-1] if len(integers) > 2 else -1,
        })
    return {
        "player_id": int(observation["player_id"]),
        "current_player": int(observation["current_player"]),
        "phase": observation["phase"],
        "hand": [int(tile) for tile in observation["hand"]],
        "melds": [[serialize_meld(meld) for meld in melds] for melds in observation["melds"]],
        "discards": [[int(tile) for tile in river] for river in observation["discards"]],
        "wall_remaining": int(observation["wall_remaining"]),
        "wall_remaining_by_player": [int(value) for value in observation.get("wall_remaining_by_player", [0, 0, 0, 0])],
        "events": events,
        "last_discard": list(observation["last_discard"]) if observation.get("last_discard") else [-1, -1],
        "prevalent_wind": int(observation.get("prevalent_wind", 0)),
        "wall_last": bool(observation.get("wall_last", False)),
        "about_kong": bool(observation.get("about_kong", False)),
        "claim_hu_only": bool(observation.get("claim_hu_only", False)),
    }


def expand_observation(observation):
    # compact_observation 的逆操作：重建 tuple 事件和 Meld 对象。
    events = []
    for event in observation.get("events", []):
        if isinstance(event, dict):
            values = [event.get("kind", "")]
            values.extend(value for value in (
                int(event.get("player", -1)), int(event.get("tile", -1)),
                int(event.get("extra", -1))) if value >= 0)
            events.append(tuple(values))
        elif isinstance(event, (list, tuple)):
            events.append(tuple(event))
        elif event:
            events.append((event,))
    return {
        "player_id": int(observation["player_id"]),
        "current_player": int(observation["current_player"]),
        "phase": observation["phase"],
        "hand": [int(tile) for tile in observation["hand"]],
        "melds": [[deserialize_meld(meld) for meld in melds] for melds in observation["melds"]],
        "discards": [[int(tile) for tile in river] for river in observation["discards"]],
        "wall_remaining": int(observation["wall_remaining"]),
        "wall_remaining_by_player": [int(value) for value in observation.get("wall_remaining_by_player", [0, 0, 0, 0])],
        "events": events,
        "last_discard": None if not observation.get("last_discard") or int(observation["last_discard"][0]) < 0 else (int(observation["last_discard"][0]), int(observation["last_discard"][1])),
        "prevalent_wind": int(observation.get("prevalent_wind", 0)),
        "wall_last": bool(observation.get("wall_last", False)),
        "about_kong": bool(observation.get("about_kong", False)),
        "claim_hu_only": bool(observation.get("claim_hu_only", False)),
    }


def _meld_signature(melds):
    # 转为可 hash 的 tuple，供 lru_cache 作为 key。
    return tuple((int(meld.kind), tuple(int(tile) for tile in meld.tiles), int(meld.from_player)) for meld in melds)


@lru_cache(maxsize=200000)
def _cached_default_stats(counts_key, meld_key):
    melds = tuple(Meld(ActionType(kind), tiles, from_player) for kind, tiles, from_player in meld_key)
    counts = list(counts_key)
    shanten = default_backend.shanten(counts, melds)
    useful = tuple(default_backend.useful_tiles(counts, melds))
    return shanten, useful


def observation_stats(counts, melds, rules=None):
    # counts shape=[34]；返回 (shanten:int, useful_tiles:tuple[int,...])。
    rules = rules or default_backend
    if rules is default_backend:
        return _cached_default_stats(tuple(counts), _meld_signature(melds))
    return rules.shanten(counts, melds), tuple(rules.useful_tiles(counts, melds))


def encode_observation(observation, rules=None):
    # 输出 V1 float 特征 list，固定 shape=[FEATURE_SIZE]。
    rules = rules or default_backend
    if observation.get("melds") and observation["melds"] and observation["melds"][0] and not hasattr(observation["melds"][0][0], "tiles"):
        observation = expand_observation(observation)
    # 依次拼接各特征块；所有计数特征都按每种牌最多4张归一化。
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
    counter = Counter(observation["hand"])
    counts = [counter.get(tile, 0) for tile in range(34)]
    melds = observation["melds"][observation["player_id"]]
    shanten, useful_tiles = observation_stats(counts, melds, rules)
    useful = set(useful_tiles)
    values.extend([1.0 if tile in useful else 0.0 for tile in range(34)])
    event_counts = Counter(str(event[0]).upper() for event in observation["events"] if event)
    values.extend([min(event_counts.get(kind, 0), 16) / 16.0 for kind in EVENT_TYPES])
    player_id = observation["player_id"]
    current = observation["current_player"]
    phase = observation["phase"]
    wall_by_player = observation.get("wall_remaining_by_player", [0, 0, 0, 0])
    last_discard = observation.get("last_discard")
    values.extend([
        player_id / 3.0,
        current / 3.0,
        1.0 if player_id == current else 0.0,
        1.0 if phase == "discard" else 0.0,
        1.0 if phase == "claim" else 0.0,
        min(observation["wall_remaining"], 136) / 136.0,
        min(len(observation["events"]), 128) / 128.0,
        1.0 if last_discard else 0.0,
        observation.get("prevalent_wind", 0) / 3.0,
        (min(max(shanten, -1), 8) + 1) / 9.0,
        min(len(useful), 34) / 34.0,
        (last_discard[1] + 1) / 34.0 if last_discard else 0.0,
        sum(wall_by_player) / 84.0,
    ])
    # values 的布局必须与 FEATURE_SIZE 常量保持一致。
    return values


def encode_action(action):
    # 输出 shape=[ACTION_SIZE=8]；牌 id 使用 +1 后除以34，令缺失值 -1 映射为0。
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
