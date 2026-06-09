import random
from collections import Counter

from mahjong_agent.engine.actions import ActionType
from mahjong_agent.rules import default_backend


class RandomPolicy(object):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, observation, legal_actions):
        if not legal_actions:
            raise ValueError("no legal actions")
        return self.rng.choice(legal_actions)


class HeuristicPolicy(object):
    """Efficiency-first deterministic baseline and BC teacher."""

    def __init__(self, rules=None):
        self.rules = rules or default_backend

    def _discard_score(self, observation, action):
        hand = list(observation["hand"])
        hand.remove(action.tile if action.kind == ActionType.PLAY else action.discard)
        counter = Counter(hand)
        counts = [counter.get(tile, 0) for tile in range(34)]
        melds = observation["melds"][observation["player_id"]]
        shanten = self.rules.shanten(counts, melds)
        useful = self.rules.useful_tiles(counts, melds)
        visible = Counter()
        for river in observation["discards"]:
            visible.update(river)
        remaining = sum(max(0, 4 - counts[tile] - visible[tile]) for tile in useful)
        honor_penalty = 0.2 if action.tile >= 27 and visible[action.tile] == 0 else 0.0
        return -10.0 * shanten + remaining - honor_penalty

    def act(self, observation, legal_actions):
        for action in legal_actions:
            if action.kind == ActionType.HU:
                return action
        scored = []
        for action in legal_actions:
            if action.kind == ActionType.PASS:
                score = 0.0
            elif action.kind in (ActionType.PLAY, ActionType.CHI, ActionType.PENG):
                score = self._discard_score(observation, action)
                if action.kind in (ActionType.CHI, ActionType.PENG):
                    score -= 1.0
            elif action.kind in (ActionType.GANG, ActionType.BUGANG):
                score = 2.0
            else:
                score = -100.0
            scored.append((score, action.key(), action))
        return max(scored, key=lambda item: (item[0], item[1]))[2]
