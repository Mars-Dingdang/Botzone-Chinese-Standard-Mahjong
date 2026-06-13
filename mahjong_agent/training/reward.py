"""Auditable public-information reward components."""

import math

from mahjong_agent.policies.analysis import action_deal_in_risk, hand_potential


def public_potential(observation, coefficients=None, action=None):
    coefficients = coefficients or {}
    potential = hand_potential(observation, action)
    fan_feasibility = (
        potential["qualifying_waits"] +
        potential["expected_fan"] / 88.0 +
        potential["fan_structure"] / float(max(1, potential["shanten"] + 2))
    )
    risk = -action_deal_in_risk(observation, action)
    components = {
        "efficiency": -float(potential["shanten"]) +
                      math.log1p(potential["useful_remaining"]),
        "fan_feasibility": fan_feasibility,
        "qualifying_waits": float(potential["qualifying_waits"]),
        "expected_fan": float(potential["expected_fan"]),
        "deal_in_risk": risk,
        "action_risk_reward": float(coefficients.get("deal_in_risk", 0.0)) * risk,
        "draw_tenpai": float(potential["shanten"] == 0),
    }
    total = sum(float(coefficients.get(name, 0.0)) * value
                for name, value in components.items()
                if name not in ("deal_in_risk", "action_risk_reward"))
    return total, components


def shaped_rewards(potentials, terminal_reward, gamma=0.99, step_cap=0.1,
                   episode_cap=1.0):
    rewards = [0.0] * len(potentials)
    components = []
    running = 0.0
    for index, (current, detail) in enumerate(potentials):
        following = potentials[index + 1][0] if index + 1 < len(potentials) else 0.0
        immediate = float(detail.get("action_risk_reward", 0.0))
        value = max(-step_cap, min(step_cap, gamma * following - current + immediate))
        value = max(-episode_cap - running, min(episode_cap - running, value))
        running += value
        rewards[index] = value
        components.append(dict(detail, shaping=value))
    if rewards:
        rewards[-1] += terminal_reward
    return rewards, components
