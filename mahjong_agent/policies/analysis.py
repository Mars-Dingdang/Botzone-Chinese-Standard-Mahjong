"""Public-information hand potential and discard-risk analysis."""

from collections import Counter

from mahjong_agent.engine.actions import ActionType, Meld
from mahjong_agent.rules import default_backend


def visible_tiles(observation):
    # 返回 Counter[tile_id -> 公开可见张数]，不含自己的暗手。
    visible = Counter(tile for river in observation["discards"] for tile in river)
    for melds in observation["melds"]:
        for meld in melds:
            visible.update(meld.tiles)
    return visible


def simulate_action(observation, action):
    # 在复制出的 hand:list[int] 与 melds:list[Meld] 上模拟，不修改原 observation。
    player = observation["player_id"]
    hand = list(observation["hand"])
    melds = list(observation["melds"][player])
    if action is None or action.kind == ActionType.PASS:
        return hand, melds
    if action.kind == ActionType.PLAY:
        hand.remove(action.tile)
    elif action.kind == ActionType.CHI:
        needed = list(action.sequence)
        needed.remove(action.tile)
        for tile in needed:
            hand.remove(tile)
        hand.remove(action.discard)
        source = (observation.get("last_discard") or (-1, -1))[0]
        melds.append(Meld(ActionType.CHI, action.sequence, source))
    elif action.kind == ActionType.PENG:
        hand.remove(action.tile)
        hand.remove(action.tile)
        hand.remove(action.discard)
        source = (observation.get("last_discard") or (-1, -1))[0]
        melds.append(Meld(ActionType.PENG, (action.tile,) * 3, source))
    elif action.kind == ActionType.GANG:
        remove_count = 3 if observation.get("phase") == "claim" else 4
        for _ in range(remove_count):
            hand.remove(action.tile)
        source = (observation.get("last_discard") or (-1, -1))[0]
        melds.append(Meld(ActionType.GANG, (action.tile,) * 4, source))
    elif action.kind == ActionType.BUGANG:
        hand.remove(action.tile)
        for index, meld in enumerate(melds):
            if meld.kind == ActionType.PENG and meld.tiles[0] == action.tile:
                melds[index] = Meld(ActionType.GANG, (action.tile,) * 4, meld.from_player)
                break
    return hand, melds


def hand_potential(observation, action=None, rules=None):
    # 使用公开信息估计动作后的向听数、有效牌余量和番种结构潜力。
    rules = rules or default_backend
    hand, melds = simulate_action(observation, action)
    counter = Counter(hand)
    # counts shape=[34]；remaining 只为有效牌记录尚未可见的估计张数。
    counts = [counter.get(tile, 0) for tile in range(34)]
    shanten = rules.shanten(counts, melds)
    useful = rules.useful_tiles(counts, melds)
    visible = visible_tiles(observation)
    remaining = {tile: max(0, 4 - counts[tile] - visible[tile]) for tile in useful}
    useful_remaining = sum(remaining.values())
    legal_waits = 0
    expected_fan = 0.0
    # 听牌时逐种有效牌计算是否达到最低8番，并按剩余张数加权。
    if shanten == 0:
        for tile in useful:
            work = list(counts)
            work[tile] += 1
            context = {
                "player_id": observation["player_id"],
                "seat_wind": observation["player_id"],
                "prevalent_wind": observation.get("prevalent_wind", 0),
                "self_drawn": True, "fourth_tile": False, "about_kong": False,
                "wall_last": False, "flower_count": 0,
            }
            fan = rules.fan(work, melds, tile, context)
            if fan >= 8:
                legal_waits += remaining[tile]
                expected_fan += remaining[tile] * min(fan, 88)
    suited = [sum(counts[base:base + 9]) for base in (0, 9, 18)]
    honors = sum(counts[27:])
    triplets = sum(value >= 3 for value in counts)
    pairs = sum(value >= 2 for value in counts)
    terminals = sum(counts[tile] for tile in (0, 8, 9, 17, 18, 26))
    concentration = max(suited) - (sum(suited) - max(suited)) * 0.25
    structure = (float(not melds) + triplets * 0.7 + pairs * 0.15 +
                 concentration * 0.08 + honors * 0.12 + terminals * 0.04)
    # 返回值均只依赖公开信息，可安全用于策略输入或奖励塑形。
    return {
        "shanten": shanten, "useful_remaining": useful_remaining,
        "qualifying_waits": legal_waits, "expected_fan": expected_fan,
        "fan_structure": structure,
    }


def action_discard(action):
    # 返回动作最终会打出的 tile id；不产生弃牌的动作返回 -1。
    if action is None:
        return -1
    if action.kind == ActionType.PLAY:
        return action.tile
    if action.kind in (ActionType.CHI, ActionType.PENG):
        return action.discard
    return -1


def action_deal_in_risk(observation, action):
    """Estimate public-information risk of the tile discarded by ``action``."""
    tile = action_discard(action)
    if tile < 0:
        return 0.0
    visible = visible_tiles(observation)
    visible.update(observation["hand"])
    player = observation["player_id"]
    # 分别估计三名对手的风险，最后取最危险者。
    risks = []
    for opponent in range(4):
        if opponent == player:
            continue
        river = observation["discards"][opponent]
        if tile in river:
            risks.append(0.0)
            continue
        risk = max(0.0, 1.0 - visible[tile] / 4.0)
        if tile >= 27:
            risk *= 0.6
        else:
            base, rank = tile - tile % 9, tile % 9
            if ((rank >= 3 and base + rank - 3 in river) or
                    (rank <= 5 and base + rank + 3 in river)):
                risk *= 0.65
        openness = len(observation["melds"][opponent])
        risks.append(risk * (1.0 + 0.15 * openness))
    return max(risks) if risks else 0.0


def direct_deal_in_index(records, result, learner_seat):
    # 若学习者放铳，逆序寻找其轨迹中最后一次实际弃牌的位置。
    if result["loser"] != learner_seat:
        return -1
    return next((index for index in range(len(records) - 1, -1, -1)
                 if action_discard(records[index]["action"]) >= 0), -1)
