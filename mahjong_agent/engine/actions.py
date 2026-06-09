from collections import namedtuple
from enum import IntEnum


class ActionType(IntEnum):
    PASS = 0
    PLAY = 1
    CHI = 2
    PENG = 3
    GANG = 4
    BUGANG = 5
    HU = 6


Meld = namedtuple("Meld", "kind tiles from_player")


class Action(namedtuple("ActionBase", "kind tile sequence discard")):
    __slots__ = ()

    def __new__(cls, kind, tile=-1, sequence=(), discard=-1):
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
        return (int(self.kind), self.tile, self.sequence, self.discard)
