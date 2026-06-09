"""Single-hand four-player Mahjong environment.

The environment implements turn progression, draws, discards, chi/peng/gang
claims and a configurable rules backend. Flower replacement and multi-winner
settlement are intentionally outside the first Botzone-focused release.
"""

import random
from collections import Counter

from .actions import Action, ActionType, Meld
from .tiles import full_wall, is_suited
from mahjong_agent.rules import default_backend


class MahjongEnv(object):
    def __init__(self, rules=None, min_fan=8):
        self.rules = rules or default_backend
        self.min_fan = min_fan
        self.reset()

    def reset(self, seed=None, wall=None):
        rng = random.Random(seed)
        self.wall = list(wall) if wall is not None else full_wall()
        if wall is None:
            rng.shuffle(self.wall)
        if len(self.wall) < 53:
            raise ValueError("wall must contain enough tiles to deal")
        self.hands = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.events = []
        self.current_player = 0
        self.prevalent_wind = (seed or 0) % 4
        self.phase = "draw"
        self.last_discard = None
        self.pending_claimers = []
        self.claim_responses = {}
        self.terminal = False
        self.winner = None
        self.loser = None
        self.fan_count = 0
        self.drawn_tile = None
        self.about_kong = False
        self.invalid_actions = 0
        for _ in range(13):
            for player in range(4):
                self.hands[player].append(self.wall.pop())
        for hand in self.hands:
            hand.sort()
        self._draw()
        return self.observe(0)

    def _draw(self):
        if not self.wall:
            self.terminal = True
            self.phase = "terminal"
            self.events.append(("DRAW_GAME",))
            return
        tile = self.wall.pop()
        self.drawn_tile = tile
        self.hands[self.current_player].append(tile)
        self.hands[self.current_player].sort()
        self.phase = "discard"
        self.events.append(("DRAW", self.current_player))

    def _counts(self, player):
        counter = Counter(self.hands[player])
        return [counter.get(tile, 0) for tile in range(34)]

    def observe(self, player_id):
        return {
            "player_id": player_id,
            "current_player": self.current_player,
            "phase": self.phase,
            "hand": list(self.hands[player_id]),
            "melds": [list(melds) for melds in self.melds],
            "discards": [list(tiles) for tiles in self.discards],
            "wall_remaining": len(self.wall),
            "wall_remaining_by_player": [len(self.wall) // 4] * 4,
            "prevalent_wind": self.prevalent_wind,
            "events": list(self.events[-128:]),
            "last_discard": self.last_discard,
        }

    def _claim_actions(self, player):
        source, tile = self.last_discard
        actions = [Action.pass_()]
        if player == source:
            return actions
        counts = self._counts(player)
        counts[tile] += 1
        if self.rules.can_hu(counts, self.melds[player], tile,
                             context=self._fan_context(player, False), min_fan=self.min_fan):
            actions.append(Action.hu())
        counts[tile] -= 1
        hand_counts = self._counts(player)
        if hand_counts[tile] >= 2:
            remaining = list(self.hands[player])
            remaining.remove(tile)
            remaining.remove(tile)
            for discard in sorted(set(remaining)):
                actions.append(Action(ActionType.PENG, tile, (), discard))
        if hand_counts[tile] >= 3:
            actions.append(Action(ActionType.GANG, tile))
        if player == (source + 1) % 4 and is_suited(tile):
            base = tile - tile % 9
            rank_index = tile % 9
            for start in range(max(0, rank_index - 2), min(6, rank_index) + 1):
                seq = (base + start, base + start + 1, base + start + 2)
                needed = list(seq)
                needed.remove(tile)
                if all(hand_counts[item] >= needed.count(item) for item in set(needed)):
                    for discard in sorted(set(self.hands[player])):
                        if discard not in needed or hand_counts[discard] > needed.count(discard):
                            actions.append(Action(ActionType.CHI, tile, seq, discard))
        return actions

    def legal_actions(self, player_id=None):
        player = self.current_player if player_id is None else player_id
        if self.terminal or player != self.current_player:
            return []
        if self.phase == "claim":
            return self._claim_actions(player)
        actions = [Action.play(tile) for tile in sorted(set(self.hands[player]))]
        counts = self._counts(player)
        if self.drawn_tile is not None and self.rules.can_hu(
                counts, self.melds[player], self.drawn_tile,
                context=self._fan_context(player, True), min_fan=self.min_fan):
            actions.append(Action.hu())
        for tile, count in enumerate(counts):
            if count == 4:
                actions.append(Action(ActionType.GANG, tile))
        for meld in self.melds[player]:
            if meld.kind == ActionType.PENG and counts[meld.tiles[0]]:
                actions.append(Action(ActionType.BUGANG, meld.tiles[0]))
        return actions

    def step(self, action):
        legal = dict((item.key(), item) for item in self.legal_actions())
        if action.key() not in legal:
            self.invalid_actions += 1
            raise ValueError("illegal action: %r" % (action,))
        action = legal[action.key()]
        player = self.current_player
        if action.kind == ActionType.HU:
            if self.phase == "claim":
                self._record_claim(action)
            else:
                self._finish_hu(player, True)
        elif action.kind == ActionType.PLAY:
            self.hands[player].remove(action.tile)
            self.discards[player].append(action.tile)
            self.last_discard = (player, action.tile)
            self.events.append(("PLAY", player, action.tile))
            self.pending_claimers = [(player + offset) % 4 for offset in (1, 2, 3)]
            self.claim_responses = {}
            self.current_player = self.pending_claimers[0]
            self.phase = "claim"
            self.drawn_tile = None
        elif action.kind == ActionType.PASS:
            self._record_claim(action)
        elif action.kind in (ActionType.CHI, ActionType.PENG):
            self._record_claim(action)
        elif action.kind == ActionType.GANG:
            if self.phase == "claim":
                self._record_claim(action)
                return self.observe(player)
            else:
                tile = action.tile
                for _ in range(4):
                    self.hands[player].remove(tile)
                self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, player))
            self.events.append(("GANG", player, tile))
            self.last_discard = None
            self.phase = "draw"
            self.about_kong = True
            self._draw()
        elif action.kind == ActionType.BUGANG:
            tile = action.tile
            self.hands[player].remove(tile)
            index = next(i for i, meld in enumerate(self.melds[player])
                         if meld.kind == ActionType.PENG and meld.tiles[0] == tile)
            old = self.melds[player][index]
            self.melds[player][index] = Meld(ActionType.GANG, (tile,) * 4, old.from_player)
            self.events.append(("BUGANG", player, tile))
            self.phase = "draw"
            self.about_kong = True
            self._draw()
        return self.observe(player)

    def _fan_context(self, player, self_drawn):
        tile = self.drawn_tile if self_drawn else (self.last_discard[1] if self.last_discard else -1)
        visible = sum(river.count(tile) for river in self.discards)
        visible += sum(meld.tiles.count(tile) for melds in self.melds for meld in melds)
        return {"player_id": player, "seat_wind": player,
                "prevalent_wind": self.prevalent_wind, "self_drawn": self_drawn,
                "fourth_tile": tile >= 0 and visible + int(self_drawn) >= 4,
                "about_kong": self.about_kong, "wall_last": not self.wall,
                "flower_count": 0}

    def _finish_hu(self, player, self_drawn):
        win_tile = self.drawn_tile if self_drawn else self.last_discard[1]
        counts = self._counts(player)
        if not self_drawn:
            counts[win_tile] += 1
        self.fan_count = self.rules.fan(counts, self.melds[player], win_tile,
                                        self._fan_context(player, self_drawn))
        self.terminal = True
        self.phase = "terminal"
        self.winner = player
        self.loser = None if self_drawn else self.last_discard[0]
        self.events.append(("HU", player, self.loser, self.fan_count))

    def _record_claim(self, action):
        self.claim_responses[self.current_player] = action
        remaining = [item for item in self.pending_claimers if item not in self.claim_responses]
        if remaining:
            self.current_player = remaining[0]
        else:
            self._resolve_claims()

    def _resolve_claims(self):
        source, tile = self.last_discard
        for kind in (ActionType.HU, ActionType.PENG, ActionType.GANG, ActionType.CHI):
            for player in self.pending_claimers:
                action = self.claim_responses[player]
                if action.kind != kind or (kind == ActionType.CHI and player != (source + 1) % 4):
                    continue
                self.current_player = player
                if kind == ActionType.HU:
                    self._finish_hu(player, False)
                    return
                if kind in (ActionType.CHI, ActionType.PENG):
                    required = list(action.sequence if kind == ActionType.CHI else (tile, tile, tile))
                    required.remove(tile)
                    for item in required:
                        self.hands[player].remove(item)
                    self.melds[player].append(Meld(kind, tuple(required + [tile]), source))
                    self.hands[player].remove(action.discard)
                    self.discards[player].append(action.discard)
                    self.last_discard = (player, action.discard)
                    self.events.append((kind.name, player, tile, action.discard))
                    self.pending_claimers = [(player + offset) % 4 for offset in (1, 2, 3)]
                    self.claim_responses = {}
                    self.current_player = self.pending_claimers[0]
                    self.phase = "claim"
                    return
                for _ in range(3):
                    self.hands[player].remove(tile)
                self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, source))
                self.events.append(("GANG", player, tile))
                self.last_discard = None
                self.phase = "draw"
                self.about_kong = True
                self._draw()
                return
        self.current_player = (source + 1) % 4
        self.last_discard = None
        self.phase = "draw"
        self.about_kong = False
        self._draw()

    def is_terminal(self):
        return self.terminal

    def result(self):
        scores = [0, 0, 0, 0]
        if self.winner is not None:
            fan = max(self.min_fan, self.fan_count)
            if self.loser is None:
                scores = [-(8 + fan) if i != self.winner else 3 * (8 + fan)
                          for i in range(4)]
            else:
                scores = [-8] * 4
                scores[self.winner] = 24 + fan
                scores[self.loser] -= fan
        return {
            "winner": self.winner,
            "loser": self.loser,
            "scores": scores,
            "draw": self.winner is None,
            "invalid_actions": self.invalid_actions,
            "fan_count": self.fan_count,
        }
