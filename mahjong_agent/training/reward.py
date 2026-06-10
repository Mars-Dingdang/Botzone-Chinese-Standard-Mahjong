"""Auditable public-information reward components."""

import math
from collections import Counter

from mahjong_agent.rules import default_backend


def public_potential(observation, coefficients=None):
    coefficients = coefficients or {}
    hand = Counter(observation["hand"])
    counts = [hand[tile] for tile in range(34)]
    melds = observation["melds"][observation["player_id"]]
    shanten = default_backend.shanten(counts, melds)
    useful = default_backend.useful_tiles(counts, melds)
    visible = Counter(tile for river in observation["discards"] for tile in river)
    for meld_list in observation["melds"]:
        for meld in meld_list:
            visible.update(meld.tiles)
    useful_remaining = sum(max(0, 4 - counts[tile] - visible[tile]) for tile in useful)
    danger = 0.0
    if observation.get("last_discard"):
        danger = visible[observation["last_discard"][1]] / 4.0
    fan_feasibility = 1.0 / float(max(1, shanten + 2))
    components = {
        "efficiency": -float(shanten) + math.log1p(useful_remaining),
        "fan_feasibility": fan_feasibility,
        "deal_in_risk": -danger,
        "draw_tenpai": float(shanten == 0),
    }
    total = sum(float(coefficients.get(name, 0.0)) * value
                for name, value in components.items())
    return total, components


def shaped_rewards(potentials, terminal_reward, gamma=0.99, step_cap=0.1,
                   episode_cap=1.0):
    rewards = [0.0] * len(potentials)
    components = []
    running = 0.0
    for index, (current, detail) in enumerate(potentials):
        following = potentials[index + 1][0] if index + 1 < len(potentials) else 0.0
        value = max(-step_cap, min(step_cap, gamma * following - current))
        value = max(-episode_cap - running, min(episode_cap - running, value))
        running += value
        rewards[index] = value
        components.append(dict(detail, shaping=value))
    if rewards:
        rewards[-1] += terminal_reward
    return rewards, components
