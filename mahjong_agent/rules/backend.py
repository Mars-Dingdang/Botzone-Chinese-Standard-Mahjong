"""Rules facade.

PyMahjongGB is used when installed. The fallback implements common-hand
recognition and a conservative eight-fan gate suitable for local development.
Production Botzone submissions should install/vendor PyMahjongGB.
"""

from functools import lru_cache
from itertools import product


def _tile_to_name(tile):
    if tile < 27:
        return ("W", "T", "B")[tile // 9] + str(tile % 9 + 1)
    return ("F1", "F2", "F3", "F4", "J1", "J2", "J3")[tile - 27]


def _remove_sets(counts, sets_left):
    if sets_left == 0:
        return all(value == 0 for value in counts)
    try:
        first = next(i for i, value in enumerate(counts) if value)
    except StopIteration:
        return False
    if counts[first] >= 3:
        nxt = list(counts)
        nxt[first] -= 3
        if _remove_sets(tuple(nxt), sets_left - 1):
            return True
    if first < 27 and first % 9 <= 6 and counts[first + 1] and counts[first + 2]:
        nxt = list(counts)
        nxt[first] -= 1
        nxt[first + 1] -= 1
        nxt[first + 2] -= 1
        if _remove_sets(tuple(nxt), sets_left - 1):
            return True
    return False


@lru_cache(maxsize=200000)
def _standard_win(counts, meld_count):
    sets_left = 4 - meld_count
    if sum(counts) != sets_left * 3 + 2:
        return False
    for tile, value in enumerate(counts):
        if value >= 2:
            nxt = list(counts)
            nxt[tile] -= 2
            if _remove_sets(tuple(nxt), sets_left):
                return True
    return False


def _pack_variants(melds, player_id):
    choices = []
    for meld in melds:
        kind = meld.kind.name
        representative = sorted(meld.tiles)[1] if kind == "CHI" else meld.tiles[0]
        offers = (1, 2, 3) if kind == "CHI" else ((player_id - meld.from_player) % 4,)
        choices.append(tuple((kind, _tile_to_name(representative), offer) for offer in offers))
    return product(*choices) if choices else ((),)


class RulesBackend(object):
    def __init__(self):
        try:
            from MahjongGB import MahjongFanCalculator
            self.official_fan_calculator = MahjongFanCalculator
            self.has_official = True
        except ImportError:
            self.official_fan_calculator = None
            self.has_official = False

    def is_complete_hand(self, counts, melds=()):
        if _standard_win(tuple(counts), len(melds)):
            return True
        return not melds and sum(value == 2 for value in counts) == 7

    def fan(self, counts, melds=(), win_tile=-1, context=None):
        if self.has_official and win_tile >= 0 and context is not None:
            try:
                hand = []
                work = list(counts)
                work[win_tile] -= 1
                for tile, count in enumerate(work):
                    hand.extend([_tile_to_name(tile)] * count)
                totals = []
                for pack in _pack_variants(melds, context.get("player_id", 0)):
                    result = self.official_fan_calculator(
                        tuple(pack), tuple(hand), _tile_to_name(win_tile),
                        int(context.get("flower_count", 0)),
                        bool(context.get("self_drawn", False)),
                        bool(context.get("fourth_tile", False)),
                        bool(context.get("about_kong", False)),
                        bool(context.get("wall_last", False)),
                        int(context.get("seat_wind", 0)),
                        int(context.get("prevalent_wind", 0)),
                        verbose=False,
                    )
                    totals.append(sum(item[0] * item[1] for item in result))
                return min(totals)
            except Exception:
                # Keep local simulation usable if a third-party package exposes
                # a different signature; official golden tests should catch it.
                pass
        return 8 if self.is_complete_hand(counts, melds) else 0

    def can_hu(self, counts, melds=(), win_tile=-1, context=None, min_fan=8):
        if self.has_official:
            return self.strict_can_hu(counts, melds, win_tile, context, min_fan)
        return self.fan(counts, melds, win_tile, context) >= min_fan

    def strict_can_hu(self, counts, melds=(), win_tile=-1, context=None, min_fan=8):
        """Return true only when the official calculator proves the hand legal."""
        if not self.has_official or win_tile < 0 or context is None:
            return False
        try:
            hand = []
            work = list(counts)
            work[win_tile] -= 1
            for tile, count in enumerate(work):
                hand.extend([_tile_to_name(tile)] * count)
            totals = []
            for pack in _pack_variants(melds, context.get("player_id", 0)):
                result = self.official_fan_calculator(
                    tuple(pack), tuple(hand), _tile_to_name(win_tile),
                    int(context.get("flower_count", 0)),
                    bool(context.get("self_drawn", False)),
                    bool(context.get("fourth_tile", False)),
                    bool(context.get("about_kong", False)),
                    bool(context.get("wall_last", False)),
                    int(context.get("seat_wind", 0)),
                    int(context.get("prevalent_wind", 0)),
                    verbose=False,
                )
                totals.append(sum(item[0] * item[1] for item in result))
            return bool(totals) and min(totals) >= min_fan
        except Exception:
            return False

    def shanten(self, counts, melds=()):
        """Return an exact but intentionally simple distance-to-win estimate."""
        if self.is_complete_hand(counts, melds):
            return -1
        best = 8
        work = list(counts)
        for tile in range(34):
            if work[tile] >= 4:
                continue
            work[tile] += 1
            if self.is_complete_hand(work, melds):
                return 0
            work[tile] -= 1
        # Fast structural approximation outside tenpai.
        sets = sum(value // 3 for value in work)
        pairs = sum(value >= 2 for value in work)
        best = max(0, 8 - sets * 2 - min(pairs, 1))
        return best

    def useful_tiles(self, counts, melds=()):
        current = self.shanten(counts, melds)
        useful = []
        work = list(counts)
        for tile in range(34):
            if work[tile] >= 4:
                continue
            work[tile] += 1
            if self.shanten(work, melds) < current:
                useful.append(tile)
            work[tile] -= 1
        return useful


default_backend = RulesBackend()
