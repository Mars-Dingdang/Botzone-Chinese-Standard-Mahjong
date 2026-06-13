"""Strict Botzone action generation and outbound validation."""

from collections import Counter

from mahjong_agent.engine.actions import Action, ActionType
from mahjong_agent.engine.tiles import is_suited, name_to_tile
from mahjong_agent.rules import default_backend
from mahjong_agent.rules.legality import can_kong


def hu_context(state, self_drawn):
    # state 是 ProtocolState；返回 MahjongGB 严格和牌计算所需的上下文 dict。
    tile = state.drawn_tile if self_drawn else (
        state.last_discard[1] if state.last_discard else -1)
    visible = sum(river.count(tile) for river in state.discards) if tile >= 0 else 0
    if tile >= 0:
        for owner, melds in enumerate(state.melds):
            for meld in melds:
                visible += meld.tiles.count(tile)
                # ProtocolState retains claimed discards in river history. Remove
                # one possible overlap per exposed meld; under-counting here is
                # safer than incorrectly declaring an otherwise sub-eight-fan HU.
                if meld.from_player != owner and tile in meld.tiles:
                    visible -= 1
    return {
        "player_id": state.player_id, "seat_wind": state.player_id,
        "prevalent_wind": state.prevalent_wind, "self_drawn": self_drawn,
        "fourth_tile": tile >= 0 and visible + int(self_drawn) >= 4,
        "about_kong": state.claim_hu_only, "wall_last": state.wall_last,
        "flower_count": state.flower_counts[state.player_id],
    }


def _strict_can_hu(state, tile, self_drawn):
    # 无官方 MahjongGB 时宁可禁止 HU，避免 Botzone 输出无法证明合法的动作。
    rules = default_backend
    if not rules.has_official or tile is None or tile < 0:
        return False
    # counts shape=[34]，表示当前玩家暗手中各牌数量。
    counts = [state.hand.count(item) for item in range(34)]
    if not self_drawn:
        counts[tile] += 1
    return rules.strict_can_hu(
        counts, state.melds[state.player_id], tile,
        hu_context(state, self_drawn), min_fan=8)


def strict_legal_actions(state):
    # 返回 list[Action]，只包含可安全提交给 Botzone 的严格合法动作。
    hand = list(state.hand)
    if state.phase == "ack":
        return [Action.pass_()]
    if state.phase == "discard":
        # 自己摸牌后的决策：打牌、和牌、暗杠或补杠。
        actions = [Action.play(tile) for tile in sorted(set(hand))]
        if _strict_can_hu(state, state.drawn_tile, True):
            actions.append(Action.hu())
        if can_kong(state.wall_last, state.wall_remaining_by_player[state.player_id]):
            counts = Counter(hand)
            actions.extend(Action(ActionType.GANG, tile)
                           for tile, count in counts.items() if count == 4)
            actions.extend(
                Action(ActionType.BUGANG, meld.tiles[0])
                for meld in state.melds[state.player_id]
                if meld.kind == ActionType.PENG and counts[meld.tiles[0]] > 0)
        return actions

    # claim 阶段至少始终允许 PASS。
    actions = [Action.pass_()]
    if not state.last_discard:
        return actions
    source, tile = state.last_discard
    if source == state.player_id:
        return actions
    if _strict_can_hu(state, tile, False):
        actions.append(Action.hu())
    if state.claim_hu_only or state.wall_last:
        return actions
    counts = Counter(hand)
    if counts[tile] >= 2:
        remaining = list(hand)
        remaining.remove(tile)
        remaining.remove(tile)
        actions.extend(Action(ActionType.PENG, tile, (), discard)
                       for discard in sorted(set(remaining)))
    if counts[tile] >= 3 and can_kong(
            state.wall_last, state.wall_remaining_by_player[state.player_id]):
        actions.append(Action(ActionType.GANG, tile))
    if state.player_id == (source + 1) % 4 and is_suited(tile):
        base, offset = tile - tile % 9, tile % 9
        for start in range(max(0, offset - 2), min(6, offset) + 1):
            sequence = (base + start, base + start + 1, base + start + 2)
            needed = list(sequence)
            needed.remove(tile)
            if all(counts[item] >= needed.count(item) for item in set(needed)):
                remaining = list(hand)
                for item in needed:
                    remaining.remove(item)
                actions.extend(Action(ActionType.CHI, tile, sequence, discard)
                               for discard in sorted(set(remaining)))
    # 用规范 action key 去重，同时保持首次出现的顺序。
    return list(dict((action.key(), action) for action in actions).values())


def validate_action(state, action):
    # 返回 (is_valid:bool, reason:str)，便于上层记录动作被修正的原因。
    legal = dict((item.key(), item) for item in strict_legal_actions(state))
    if action.key() in legal:
        return True, "legal"
    if state.last_discard and state.last_discard[0] == state.player_id and action.kind != ActionType.PASS:
        return False, "cannot claim own discard"
    if state.claim_hu_only and action.kind not in (ActionType.PASS, ActionType.HU):
        return False, "only HU or PASS allowed after BUGANG"
    if state.wall_last and action.kind in (
            ActionType.CHI, ActionType.PENG, ActionType.GANG, ActionType.BUGANG):
        return False, "meld or kong forbidden on wall last"
    if action.kind == ActionType.HU and not default_backend.has_official:
        return False, "official MahjongGB unavailable; HU cannot be proven legal"
    return False, "action is not in strict legal action set"


def sanitize_action(state, proposed):
    # 非法提案在 claim 阶段回退为 PASS，在 discard 阶段回退为启发式合法打牌。
    valid, reason = validate_action(state, proposed)
    if valid:
        return proposed, None
    legal = strict_legal_actions(state)
    if state.phase != "discard":
        return Action.pass_(), reason
    plays = [action for action in legal if action.kind == ActionType.PLAY]
    if not plays:
        return legal[0], reason
    try:
        from mahjong_agent.policies import HeuristicPolicy
        return HeuristicPolicy().act(state.observation(), plays), reason
    except Exception:
        return plays[0], reason


def response_to_action(state, response):
    # response 是 Botzone 输出字符串；解析为内部不可变 Action。
    parts = response.strip().split()
    if not parts:
        raise ValueError("empty response")
    kind = parts[0].upper()
    if kind == "PASS":
        return Action.pass_()
    if kind == "HU":
        return Action.hu()
    if kind == "PLAY" and len(parts) == 2:
        return Action.play(name_to_tile(parts[1]))
    if kind == "BUGANG" and len(parts) == 2:
        return Action(ActionType.BUGANG, name_to_tile(parts[1]))
    if kind == "GANG":
        tile = name_to_tile(parts[1]) if len(parts) == 2 else (
            state.last_discard[1] if state.last_discard else -1)
        return Action(ActionType.GANG, tile)
    if kind == "PENG" and len(parts) == 2 and state.last_discard:
        return Action(ActionType.PENG, state.last_discard[1], (), name_to_tile(parts[1]))
    if kind == "CHI" and len(parts) == 3 and state.last_discard:
        middle = name_to_tile(parts[1])
        return Action(ActionType.CHI, state.last_discard[1],
                      (middle - 1, middle, middle + 1), name_to_tile(parts[2]))
    raise ValueError("unsupported response: %s" % response)
