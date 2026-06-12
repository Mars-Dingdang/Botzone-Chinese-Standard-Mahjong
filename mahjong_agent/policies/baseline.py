import random
from collections import Counter

from mahjong_agent.engine.actions import ActionType, Meld
from mahjong_agent.rules import default_backend


class RandomPolicy(object):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, observation, legal_actions):
        if not legal_actions:
            raise ValueError("no legal actions")
        return self.rng.choice(legal_actions)


class HeuristicPolicy(object):
    """Fast efficiency and fan-potential baseline."""

    def __init__(self, rules=None):
        self.rules = rules or default_backend

    @staticmethod
    def _visible(observation):
        visible = Counter(tile for river in observation["discards"] for tile in river)
        for melds in observation["melds"]:
            for meld in melds:
                visible.update(meld.tiles)
        return visible

    @staticmethod
    def _simulate(observation, action):
        player = observation["player_id"]
        hand = list(observation["hand"])
        melds = list(observation["melds"][player])
        if action.kind == ActionType.PASS:
            return hand, melds
        if action.kind == ActionType.PLAY:
            hand.remove(action.tile)
        elif action.kind == ActionType.CHI:
            needed = list(action.sequence)
            needed.remove(action.tile)
            for tile in needed:
                hand.remove(tile)
            hand.remove(action.discard)
            source = (observation.get("last_discard") or (-1, -1))[0]
            melds.append(Meld(ActionType.CHI, action.sequence, source))
        elif action.kind == ActionType.PENG:
            hand.remove(action.tile)
            hand.remove(action.tile)
            hand.remove(action.discard)
            source = (observation.get("last_discard") or (-1, -1))[0]
            melds.append(Meld(ActionType.PENG, (action.tile,) * 3, source))
        elif action.kind == ActionType.GANG:
            remove_count = 3 if observation.get("phase") == "claim" else 4
            for _ in range(remove_count):
                hand.remove(action.tile)
            source = (observation.get("last_discard") or (-1, -1))[0]
            melds.append(Meld(ActionType.GANG, (action.tile,) * 4, source))
        elif action.kind == ActionType.BUGANG:
            hand.remove(action.tile)
            for index, meld in enumerate(melds):
                if meld.kind == ActionType.PENG and meld.tiles[0] == action.tile:
                    melds[index] = Meld(ActionType.GANG, (action.tile,) * 4, meld.from_player)
                    break
        return hand, melds

    def _fan_potential(self, observation, counts, melds, shanten, useful, visible):
        remaining = {tile: max(0, 4 - counts[tile] - visible[tile]) for tile in useful}
        legal_waits = 0
        expected_fan = 0.0
        if shanten == 0:
            for tile in useful:
                work = list(counts)
                work[tile] += 1
                context = {
                    "player_id": observation["player_id"],
                    "seat_wind": observation["player_id"],
                    "prevalent_wind": observation.get("prevalent_wind", 0),
                    "self_drawn": True, "fourth_tile": False, "about_kong": False,
                    "wall_last": False, "flower_count": 0,
                }
                fan = self.rules.fan(work, melds, tile, context)
                if fan >= 8:
                    legal_waits += remaining[tile]
                    expected_fan += remaining[tile] * min(fan, 88)
        suited = [sum(counts[base:base + 9]) for base in (0, 9, 18)]
        honors = sum(counts[27:])
        triplets = sum(value >= 3 for value in counts)
        pairs = sum(value >= 2 for value in counts)
        terminals = sum(counts[tile] for tile in (0, 8, 9, 17, 18, 26))
        closed_bonus = 1.0 if not melds else 0.0
        concentration = max(suited) - (sum(suited) - max(suited)) * 0.25
        structure = (closed_bonus + triplets * 0.7 + pairs * 0.15 +
                     concentration * 0.08 + honors * 0.12 + terminals * 0.04)
        return legal_waits, expected_fan, structure

    def _score(self, observation, action):
        hand, melds = self._simulate(observation, action)
        counter = Counter(hand)
        counts = [counter.get(tile, 0) for tile in range(34)]
        shanten = self.rules.shanten(counts, melds)
        useful = self.rules.useful_tiles(counts, melds)
        visible = self._visible(observation)
        remaining = sum(max(0, 4 - counts[tile] - visible[tile]) for tile in useful)
        legal_waits, expected_fan, structure = self._fan_potential(
            observation, counts, melds, shanten, useful, visible)
        meld_penalty = 0.25 if action.kind in (ActionType.CHI, ActionType.PENG) else 0.0
        # Ordering is intentional: legal 8-fan waits, efficiency, then structure.
        return (legal_waits * 1000.0 + expected_fan * 0.2 -
                shanten * 100.0 + remaining * 2.0 + structure - meld_penalty)

    def act(self, observation, legal_actions):
        for action in legal_actions:
            if action.kind == ActionType.HU:
                return action
        scored = [(self._score(observation, action), action.key(), action)
                  for action in legal_actions]
        return max(scored, key=lambda item: (item[0], item[1]))[2]
