from collections import namedtuple
from enum import IntEnum


class ActionType(IntEnum):
    # 整数编号会进入特征、协议和 checkpoint，不能随意调整顺序。
    PASS = 0
    PLAY = 1
    CHI = 2
    PENG = 3
    GANG = 4
    BUGANG = 5
    HU = 6


# Meld 字段：kind:ActionType，tiles:tuple[int]，from_player:int座位号。
Meld = namedtuple("Meld", "kind tiles from_player")


class Action(namedtuple("ActionBase", "kind tile sequence discard")):
    # Action 是不可变 tuple：主动作类型、目标牌、吃牌顺子、动作后弃牌。
    __slots__ = ()

    def __new__(cls, kind, tile=-1, sequence=(), discard=-1):
        # 在构造边界统一类型，-1 表示该字段不适用于当前动作。
        return super(Action, cls).__new__(
            cls, ActionType(kind), int(tile), tuple(sequence), int(discard)
        )

    @classmethod
    def pass_(cls):
        return cls(ActionType.PASS)

    @classmethod
    def play(cls, tile):
        return cls(ActionType.PLAY, tile=tile)

    @classmethod
    def hu(cls):
        return cls(ActionType.HU)

    def key(self):
        # 可 hash 的规范键，用于合法动作去重和精确比较。
        return (int(self.kind), self.tile, self.sequence, self.discard)
