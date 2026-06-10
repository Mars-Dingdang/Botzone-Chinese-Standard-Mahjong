"""Rule constraints shared by the simulator and Botzone legality layer."""

from mahjong_agent.engine.actions import ActionType


def can_claim_discard(player, source, wall_last=False, hu_only=False):
    return player != source and not hu_only and not wall_last


def can_kong(wall_last=False, replacement_tiles=0):
    return not wall_last and replacement_tiles > 0


def action_allowed_in_claim(kind, player, source, wall_last=False,
                            hu_only=False, replacement_tiles=0):
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
