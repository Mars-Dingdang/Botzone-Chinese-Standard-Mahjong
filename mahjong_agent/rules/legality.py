"""Rule constraints shared by the simulator and Botzone legality layer."""

from mahjong_agent.engine.actions import ActionType


def can_claim_discard(player, source, wall_last=False, hu_only=False):
    # 不能响应自己的弃牌；抢杠/牌墙末尾阶段禁止普通副露声明。
    return player != source and not hu_only and not wall_last


def can_kong(wall_last=False, replacement_tiles=0):
    # 杠后必须仍有补牌可摸。
    return not wall_last and replacement_tiles > 0


def action_allowed_in_claim(kind, player, source, wall_last=False,
                            hu_only=False, replacement_tiles=0):
    # 这是 claim 阶段的共享约束；CHI 的座次限制由调用方另行判断。
    if kind == ActionType.PASS:
        return True
    if player == source:
        return False
    if kind == ActionType.HU:
        return True
    if hu_only:
        return False
    if wall_last:
        return False
    if kind == ActionType.GANG:
        return can_kong(wall_last, replacement_tiles)
    return kind in (ActionType.CHI, ActionType.PENG)
