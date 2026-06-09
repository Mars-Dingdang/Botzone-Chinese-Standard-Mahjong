"""Canonical 34-tile encoding used throughout the project."""

SUITS = ("W", "T", "B")
HONORS = ("F1", "F2", "F3", "F4", "J1", "J2", "J3")
TILE_NAMES = tuple(
    ["%s%d" % (suit, rank) for suit in SUITS for rank in range(1, 10)]
    + list(HONORS)
)
NAME_TO_TILE = dict((name, index) for index, name in enumerate(TILE_NAMES))


def tile_to_name(tile):
    if tile < 0 or tile >= 34:
        raise ValueError("tile must be in [0, 33]")
    return TILE_NAMES[tile]


def name_to_tile(name):
    try:
        return NAME_TO_TILE[name.upper()]
    except KeyError:
        raise ValueError("unknown tile: %s" % name)


def is_suited(tile):
    return 0 <= tile < 27


def suit(tile):
    return tile // 9 if is_suited(tile) else 3


def rank(tile):
    return tile % 9 + 1 if is_suited(tile) else tile - 26


def full_wall(include_flowers=False):
    wall = []
    for tile in range(34):
        wall.extend([tile] * 4)
    if include_flowers:
        wall.extend(range(34, 42))
    return wall
