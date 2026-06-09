"""Botzone protocol helpers.

This parser supports the public action vocabulary and is deliberately tolerant
of surrounding whitespace. Full replay is represented by ProtocolState so the
Botzone entry point does not depend on the local simulator's private wall.
"""

from mahjong_agent.engine.actions import Action, ActionType, Meld
from mahjong_agent.engine.tiles import name_to_tile, tile_to_name


def action_to_text(action):
    if action.kind in (ActionType.PASS, ActionType.HU):
        return action.kind.name
    if action.kind in (ActionType.PLAY, ActionType.GANG, ActionType.BUGANG):
        return "%s %s" % (action.kind.name, tile_to_name(action.tile))
    if action.kind == ActionType.PENG:
        return "PENG %s" % tile_to_name(action.discard)
    if action.kind == ActionType.CHI:
        return "CHI %s %s" % (
            tile_to_name(action.sequence[1]), tile_to_name(action.discard)
        )
    raise ValueError("unsupported action")


def parse_request(text):
    parts = text.strip().split()
    if not parts:
        return ()
    return tuple(parts)


class ProtocolState(object):
    def __init__(self):
        self.player_id = 0
        self.hand = []
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.events = []
        self.current_player = 0
        self.phase = "claim"
        self.last_discard = None

    def apply(self, request):
        parts = parse_request(request)
        if not parts:
            return
        code = parts[0]
        if code == "0":
            self.player_id = int(parts[1])
        elif code == "1":
            self.hand = sorted(name_to_tile(tile) for tile in parts[-13:])
            self.current_player = self.player_id
            self.phase = "draw"
        elif code == "2":
            self.current_player = self.player_id
            self.hand.append(name_to_tile(parts[1]))
            self.hand.sort()
            self.phase = "discard"
            self.events.append(("DRAW", self.player_id))
        elif code == "3":
            player = int(parts[1])
            action = parts[2]
            self.current_player = player
            if action == "DRAW":
                self.events.append(("DRAW", player))
            elif action == "PLAY":
                tile = name_to_tile(parts[3])
                if player == self.player_id and tile in self.hand:
                    self.hand.remove(tile)
                self.discards[player].append(tile)
                self.last_discard = (player, tile)
                self.phase = "claim"
                self.events.append(("PLAY", player, tile))
            elif action in ("PENG", "CHI", "GANG", "BUGANG"):
                last_source, claimed = self.last_discard if self.last_discard else (-1, -1)
                if action == "CHI":
                    middle = name_to_tile(parts[3])
                    discard = name_to_tile(parts[4])
                    sequence = (middle - 1, middle, middle + 1)
                    self.melds[player].append(Meld(ActionType.CHI, sequence, last_source))
                    if player == self.player_id:
                        needed = list(sequence)
                        needed.remove(claimed)
                        for tile in needed:
                            self.hand.remove(tile)
                        self.hand.remove(discard)
                    self.discards[player].append(discard)
                    self.last_discard = (player, discard)
                elif action == "PENG":
                    discard = name_to_tile(parts[3])
                    self.melds[player].append(
                        Meld(ActionType.PENG, (claimed,) * 3, last_source)
                    )
                    if player == self.player_id:
                        self.hand.remove(claimed)
                        self.hand.remove(claimed)
                        self.hand.remove(discard)
                    self.discards[player].append(discard)
                    self.last_discard = (player, discard)
                elif action == "GANG":
                    tile = claimed if len(parts) == 3 else name_to_tile(parts[3])
                    self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, last_source))
                    if player == self.player_id:
                        remove_count = 3 if tile == claimed else 4
                        for _ in range(remove_count):
                            self.hand.remove(tile)
                    self.last_discard = None
                elif action == "BUGANG":
                    tile = name_to_tile(parts[3])
                    for index, meld in enumerate(self.melds[player]):
                        if meld.kind == ActionType.PENG and meld.tiles[0] == tile:
                            self.melds[player][index] = Meld(
                                ActionType.GANG, (tile,) * 4, meld.from_player
                            )
                            break
                    if player == self.player_id:
                        self.hand.remove(tile)
                self.events.append(tuple(parts[2:]))

    def observation(self):
        return {
            "player_id": self.player_id,
            "current_player": self.current_player,
            "phase": self.phase,
            "hand": list(self.hand),
            "melds": self.melds,
            "discards": self.discards,
            "wall_remaining": max(0, 83 - len(self.events)),
            "events": list(self.events[-128:]),
            "last_discard": self.last_discard,
        }
