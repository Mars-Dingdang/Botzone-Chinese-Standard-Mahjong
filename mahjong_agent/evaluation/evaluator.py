import json
import math
import os
import random
import time
from collections import Counter

from mahjong_agent.engine.tiles import full_wall
from mahjong_agent.training.rollout import play_episode


def _mean(values):
    return sum(values) / float(max(1, len(values)))


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def confidence_interval(values):
    mean = _mean(values)
    margin = 1.96 * _std(values) / math.sqrt(max(1, len(values)))
    return [mean - margin, mean + margin]


def create_wall_manifest(walls=100, seed=2026):
    result = []
    for index in range(walls):
        wall = full_wall()
        random.Random(seed + index).shuffle(wall)
        result.append({"id": index, "seed": seed + index,
                       "prevalent_wind": (seed + index) % 4, "wall": wall})
    return {"version": 1, "seed": seed, "walls": result}


def load_wall_manifest(path=None, walls=100, seed=2026):
    if not path or not os.path.exists(path):
        return create_wall_manifest(walls, seed)
    with open(path) as handle:
        return json.load(handle)


def save_wall_manifest(path, manifest):
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _policy_metrics(results, target_seats):
    scores = [result["scores"][seat] for result, seat in zip(results, target_seats)]
    wins = [result["winner"] == seat for result, seat in zip(results, target_seats)]
    self_draws = [win and result["self_drawn"] for win, result in zip(wins, results)]
    deal_ins = [result["loser"] == seat for result, seat in zip(results, target_seats)]
    draw_tenpai = [result["draw"] and result["tenpai"][seat]
                   for result, seat in zip(results, target_seats)]
    fans = [result["fan_count"] for result, win in zip(results, wins) if win]
    qualifying_wins = [win and result["fan_count"] >= 8
                       for result, win in zip(results, wins)]
    actions = Counter()
    latencies = []
    for result, seat in zip(results, target_seats):
        actions.update(result["action_counts_by_player"][seat])
        latencies.extend(result["decision_latencies_by_player"][seat])
    ordered_latency = sorted(latencies)
    percentile = ordered_latency[min(len(ordered_latency) - 1, int(len(ordered_latency) * .95))] if ordered_latency else 0.0
    return {
        "games": len(results), "average_score": _mean(scores), "score_std": _std(scores),
        "score_95_ci": confidence_interval(scores), "win_rate": _mean(wins),
        "self_draw_rate": _mean(self_draws), "deal_in_rate": _mean(deal_ins),
        "qualifying_win_rate": _mean(qualifying_wins),
        "draw_tenpai_rate": _mean(draw_tenpai), "average_fan": _mean(fans),
        "fan_distribution": dict(Counter(fans)), "action_distribution": dict(actions),
        "invalid_actions": sum(result["invalid_actions"] for result in results),
        "latency_seconds": {"mean": _mean(latencies), "p95": percentile,
                            "max": max(latencies) if latencies else 0.0},
        "scores": scores,
    }


def evaluate(policies, games=20, seed=0, progress=False):
    results = []
    iterator = range(games)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="evaluate", unit="game")
    for game in iterator:
        result, _ = play_episode(policies, seed=seed + game)
        results.append(result)
    metrics = [_policy_metrics(results, [seat] * len(results)) for seat in range(4)]
    return {"games": games, "players": metrics}


def evaluate_duplicate(policy_a, policy_b, walls=4, seed=0,
                       policy_a_name="policy_a", policy_b_name="policy_b",
                       manifest=None, progress=False):
    manifest = manifest or create_wall_manifest(walls, seed)
    results = []
    seats = []
    iterator = manifest["walls"][:walls]
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="duplicate-eval", unit="wall")
    wall_scores = []
    started = time.time()
    for entry in iterator:
        current = []
        for a_seat in range(4):
            policies = [policy_b] * 4
            policies[a_seat] = policy_a
            result, _ = play_episode(
                policies, wall=entry["wall"], prevalent_wind=entry["prevalent_wind"])
            results.append(result)
            seats.append(a_seat)
            current.append(result["scores"][a_seat])
        wall_scores.append(_mean(current))
    metrics = _policy_metrics(results, seats)
    wall_deltas = [value * 4.0 / 3.0 for value in wall_scores]
    return dict(metrics, walls=len(wall_scores), seed=manifest["seed"],
                policy_a=policy_a_name, policy_b=policy_b_name,
                average_score_a=metrics["average_score"],
                average_score_b=-metrics["average_score"] / 3.0,
                score_delta=metrics["average_score"] * 4.0 / 3.0,
                score_delta_95_ci=confidence_interval(wall_deltas),
                wall_scores=wall_scores, elapsed_seconds=time.time() - started)


def paired_delta(candidate, baseline):
    values = [left - right for left, right in zip(
        candidate["wall_scores"], baseline["wall_scores"])]
    return {"score_delta": _mean(values), "score_delta_std": _std(values),
            "score_delta_95_ci": confidence_interval(values), "wall_deltas": values}
