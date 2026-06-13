"""Public-information-only token encoder for Feature V2."""

from collections import Counter

from mahjong_agent.engine.actions import ActionType

# V2 特征的版本号；checkpoint 和模型工厂用它判断应采用哪套编码器。
FEATURE_VERSION = 2
# 每个 token 是长度 12 的 float 列表；前 8 维有含义，后 4 维预留。
TOKEN_SIZE = 12
# 状态和单个候选动作分别 padding/truncate 到固定 token 数，便于组成 batch。
MAX_STATE_TOKENS = 256
MAX_ACTION_TOKENS = 4

# token 第 0 维的类别编号；0 留给 padding token。
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
    # 将绝对座位号转换为观察者视角：自己=0，下家=1，对家=2，上家=3。
    return (int(player) - int(observer)) % 4


def _token(kind, tile=-1, player=-1, action=-1, order=0, flags=0,
           value=0.0, extra=0.0):
    # 返回 shape=[TOKEN_SIZE=12]。类别字段统一 +1，使 0 可以专门表示 padding/未知。
    return [
        float(kind), float(tile + 1), float(player + 1), float(action + 1),
        float(order) / 128.0, float(flags), float(value), float(extra),
        0.0, 0.0, 0.0, 0.0,
    ]


def _pad(tokens, maximum):
    # 输入 tokens: list[list[float]]，每项 shape=[12]；输出固定 shape=[maximum, 12]。
    tokens = tokens[:maximum]
    # mask shape=[maximum]；1 表示真实 token，0 表示 padding。
    mask = [1] * len(tokens)
    while len(tokens) < maximum:
        tokens.append([0.0] * TOKEN_SIZE)
        mask.append(0)
    return tokens, mask


def encode_observation_v2(observation):
    # observation 是公开信息 dict；observer 是当前决策者的绝对座位号 int[0,3]。
    observer = int(observation["player_id"])
    last = observation.get("last_discard")
    phase = {"ack": 0, "draw": 1, "discard": 2, "claim": 3, "terminal": 4}.get(
        observation.get("phase"), 0)
    # flags 是 bit field：bit0=牌墙末尾，bit1=杠相关，bit2=抢杠和牌阶段。
    flags = (
        int(bool(observation.get("wall_last"))) |
        int(bool(observation.get("about_kong"))) << 1 |
        int(bool(observation.get("claim_hu_only"))) << 2
    )
    # 第一个全局 token 汇总阶段、当前玩家、牌墙余量和最近弃牌。
    tokens = [_token(
        TOKEN_GLOBAL, tile=last[1] if last else -1,
        player=_relative(observation.get("current_player", observer), observer),
        action=phase, flags=flags,
        value=float(observation.get("wall_remaining", 0)) / 84.0,
        extra=float(_relative(observation.get("prevalent_wind", 0), observer)) / 3.0,
    )]
    for order, tile in enumerate(observation["hand"]):
        # 手牌逐张编码；重复牌保留为多个 token，order 表示排序后的下标。
        tokens.append(_token(TOKEN_HAND, tile=tile, player=0, order=order))

    # claimed[(来源玩家, 牌)] 记录已被副露吃碰杠消耗的河牌数。
    claimed = Counter()
    for owner, melds in enumerate(observation["melds"]):
        for meld in melds:
            if meld.from_player != -1 and meld.from_player != owner:
                claimed[(meld.from_player, meld.tiles[-1])] += 1
    for relative in range(4):
        # 按观察者的相对座次顺序编码四家的弃牌河。
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
        # 每个副露中的每张牌各占一个 token；action 字段保存副露类型。
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
        # 只保留最近 64 个公开事件，防止历史过长挤掉更重要的 token。
        if not event:
            continue
        kind = str(event[0]).upper()
        player = event[1] if len(event) > 1 and isinstance(event[1], int) else observer
        tile = next((item for item in event[2:] if isinstance(item, int) and 0 <= item < 34), -1)
        tokens.append(_token(
            TOKEN_EVENT, tile=tile, player=_relative(player, observer),
            action=EVENT_IDS.get(kind, 0), order=order))

    # visible 汇总自己手牌、所有弃牌和所有副露，用于推导每种牌的不可见数量。
    visible = Counter(observation["hand"])
    for river in observation["discards"]:
        visible.update(river)
    for melds in observation["melds"]:
        for meld in melds:
            visible.update(meld.tiles)
    for tile in range(34):
        # value 归一化到 [0,1]，表示该牌最多四张中仍不可见的比例。
        tokens.append(_token(
            TOKEN_UNSEEN, tile=tile, value=max(0, 4 - visible[tile]) / 4.0))
    for relative in range(4):
        # 为每位玩家补充一个全局 token，估计其暗手张数和个人牌墙余量。
        player = (observer + relative) % 4
        meld_tiles = sum(len(meld.tiles) for meld in observation["melds"][player])
        concealed = 13 - (meld_tiles // 3) * 3
        if player == observation.get("current_player") and observation.get("phase") == "discard":
            concealed += 1
        tokens.append(_token(
            TOKEN_GLOBAL, player=_relative(player, observer),
            value=max(0, concealed) / 14.0,
            extra=float(observation.get("wall_remaining_by_player", [0] * 4)[player]) / 21.0))
    # 返回 (tokens, mask)，shape 分别为 [256,12] 和 [256]。
    return _pad(tokens, MAX_STATE_TOKENS)


def encode_action_v2(action):
    # action 是 Action；动作主牌/顺子牌被展开为最多 4 个 action token。
    tiles = list(action.sequence) if action.sequence else ([action.tile] if action.tile >= 0 else [])
    tokens = [_token(
        TOKEN_ACTION, tile=action.tile, action=int(action.kind),
        flags=int(action.kind == ActionType.HU), extra=(action.discard + 1) / 34.0)]
    for order, tile in enumerate(tiles[:3]):
        tokens.append(_token(TOKEN_ACTION, tile=tile, action=int(action.kind), order=order + 1))
    # 返回 shape=[4,12] 的 token 和 shape=[4] 的有效位 mask。
    return _pad(tokens, MAX_ACTION_TOKENS)
