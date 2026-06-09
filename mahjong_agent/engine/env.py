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
        self.phase = "draw"
        self.last_discard = None
        self.pending_claimers = []
        self.terminal = False
        self.winner = None
        self.loser = None
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
        if self.rules.can_hu(counts, self.melds[player], tile, min_fan=self.min_fan):
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
        if self.rules.can_hu(counts, self.melds[player], min_fan=self.min_fan):
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
            self.terminal = True
            self.phase = "terminal"
            self.winner = player
            self.loser = self.last_discard[0] if self.last_discard else None
            self.events.append(("HU", player, self.loser))
        elif action.kind == ActionType.PLAY:
            self.hands[player].remove(action.tile)
            self.discards[player].append(action.tile)
            self.last_discard = (player, action.tile)
            self.events.append(("PLAY", player, action.tile))
            self.current_player = (player + 1) % 4
            self.phase = "claim"
        elif action.kind == ActionType.PASS:
            source = self.last_discard[0]
            if self.current_player == source:
                self.current_player = (source + 1) % 4
                self.phase = "draw"
                self._draw()
            else:
                self.current_player = (self.current_player + 1) % 4
                if self.current_player == source:
                    self.current_player = (source + 1) % 4
                    self.phase = "draw"
                    self._draw()
        elif action.kind in (ActionType.CHI, ActionType.PENG):
            source, tile = self.last_discard
            required = list(action.sequence if action.kind == ActionType.CHI else (tile, tile, tile))
            required.remove(tile)
            for item in required:
                self.hands[player].remove(item)
            self.melds[player].append(Meld(action.kind, tuple(required + [tile]), source))
            self.hands[player].remove(action.discard)
            self.discards[player].append(action.discard)
            self.last_discard = (player, action.discard)
            self.events.append((action.kind.name, player, tile, action.discard))
            self.current_player = (player + 1) % 4
            self.phase = "claim"
        elif action.kind == ActionType.GANG:
            if self.phase == "claim":
                source, tile = self.last_discard
                for _ in range(3):
                    self.hands[player].remove(tile)
                self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, source))
            else:
                tile = action.tile
                for _ in range(4):
                    self.hands[player].remove(tile)
                self.melds[player].append(Meld(ActionType.GANG, (tile,) * 4, player))
            self.events.append(("GANG", player, tile))
            self.last_discard = None
            self.phase = "draw"
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
            self._draw()
        return self.observe(player)

    def is_terminal(self):
        return self.terminal

    def result(self):
        scores = [0, 0, 0, 0]
        if self.winner is not None:
            scores[self.winner] = 24 if self.loser is None else 8
            if self.loser is None:
                scores = [-8 if i != self.winner else 24 for i in range(4)]
            else:
                scores[self.loser] = -8
        return {
            "winner": self.winner,
            "loser": self.loser,
            "scores": scores,
            "draw": self.winner is None,
            "invalid_actions": self.invalid_actions,
        }
