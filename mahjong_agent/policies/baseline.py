import random

from mahjong_agent.engine.actions import ActionType
from mahjong_agent.policies.analysis import hand_potential, simulate_action
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
    def _simulate(observation, action):
        return simulate_action(observation, action)

    def _score(self, observation, action):
        potential = hand_potential(observation, action, self.rules)
        meld_penalty = 0.25 if action.kind in (ActionType.CHI, ActionType.PENG) else 0.0
        # Ordering is intentional: legal 8-fan waits, efficiency, then structure.
        return (potential["qualifying_waits"] * 1000.0 +
                potential["expected_fan"] * 0.2 -
                potential["shanten"] * 100.0 +
                potential["useful_remaining"] * 2.0 +
                potential["fan_structure"] - meld_penalty)

    def act(self, observation, legal_actions):
        for action in legal_actions:
            if action.kind == ActionType.HU:
                return action
        scored = [(self._score(observation, action), action.key(), action)
                  for action in legal_actions]
        return max(scored, key=lambda item: (item[0], item[1]))[2]
