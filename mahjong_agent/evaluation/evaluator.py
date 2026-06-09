import math
import time

from mahjong_agent.engine.tiles import full_wall
from mahjong_agent.training.rollout import play_episode


def evaluate(policies, games=20, seed=0):
    totals = [0, 0, 0, 0]
    wins = [0, 0, 0, 0]
    steps = 0
    started = time.time()
    invalid = 0
    for game in range(games):
        result, _ = play_episode(policies, seed=seed + game)
        totals = [a + b for a, b in zip(totals, result["scores"])]
        if result["winner"] is not None:
            wins[result["winner"]] += 1
        invalid += result["invalid_actions"]
        steps += result["steps"]
    return {
        "games": games,
        "average_scores": [value / float(games) for value in totals],
        "win_rates": [value / float(games) for value in wins],
        "invalid_actions": invalid,
        "steps_per_second": steps / max(1e-6, time.time() - started),
    }


def _confidence_interval(values):
    if not values:
        return [0.0, 0.0]
    mean = sum(values) / float(len(values))
    if len(values) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return [mean - margin, mean + margin]


def evaluate_duplicate(policy_a, policy_b, walls=4, seed=0,
                       policy_a_name="policy_a", policy_b_name="policy_b"):
    import random
    scores = {"a": 0, "b": 0}
    wall_deltas = []
    games = 0
    for wall_index in range(walls):
        wall = full_wall()
        random.Random(seed + wall_index).shuffle(wall)
        current_wall_deltas = []
        for a_seat in range(4):
            policies = [policy_b, policy_b, policy_b, policy_b]
            policies[a_seat] = policy_a
            result, _ = play_episode(policies, wall=wall)
            scores["a"] += result["scores"][a_seat]
            scores["b"] += sum(
                result["scores"][seat] for seat in range(4) if seat != a_seat
            ) / 3.0
            current_wall_deltas.append(
                result["scores"][a_seat]
                - sum(result["scores"][seat] for seat in range(4) if seat != a_seat) / 3.0
            )
            games += 1
        wall_deltas.append(sum(current_wall_deltas) / float(len(current_wall_deltas)))
    average_a = scores["a"] / games
    average_b = scores["b"] / games
    return {
        "games": games, "walls": walls, "seed": seed,
        "policy_a": policy_a_name, "policy_b": policy_b_name,
        "average_score_a": average_a, "average_score_b": average_b,
        "score_delta": average_a - average_b,
        "score_delta_95_ci": _confidence_interval(wall_deltas),
    }
