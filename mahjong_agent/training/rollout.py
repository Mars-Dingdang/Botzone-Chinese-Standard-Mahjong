"""Self-play rollout utilities shared by data generation and PPO."""

import time
from collections import Counter

from mahjong_agent.engine import MahjongEnv
from mahjong_agent.rules import default_backend


def play_episode(policies, seed=None, wall=None, prevalent_wind=None,
                 max_steps=512, collect=False):
    env = MahjongEnv()
    env.reset(seed=seed, wall=wall, prevalent_wind=prevalent_wind)
    trajectory = []
    steps = 0
    action_counts = Counter()
    action_counts_by_player = [Counter() for _ in range(4)]
    latencies = []
    latencies_by_player = [[] for _ in range(4)]
    while not env.is_terminal() and steps < max_steps:
        player = env.current_player
        observation = env.observe(player)
        legal = env.legal_actions(player)
        started = time.time()
        action = policies[player].act(observation, legal)
        latency = time.time() - started
        latencies.append(latency)
        latencies_by_player[player].append(latency)
        action_counts[action.kind.name] += 1
        action_counts_by_player[player][action.kind.name] += 1
        if collect:
            trajectory.append({
                "player": player,
                "observation": observation,
                "legal_actions": legal,
                "action": action,
                "privileged_hands": [list(hand) for hand in env.hands],
            })
        env.step(action)
        steps += 1
    if not env.is_terminal():
        env.terminal = True
        env.phase = "terminal"
    result = env.result()
    result["steps"] = steps
    result["action_counts"] = dict(action_counts)
    result["action_counts_by_player"] = [dict(value) for value in action_counts_by_player]
    result["decision_latencies"] = latencies
    result["decision_latencies_by_player"] = latencies_by_player
    result["tenpai"] = []
    for player in range(4):
        counts = env._counts(player)
        result["tenpai"].append(default_backend.shanten(counts, env.melds[player]) == 0)
    return result, trajectory


def play_episodes_vectorized(policy_sets, seeds=None, max_steps=512, collect=False):
    """Synchronously advance multiple environments and batch compatible policies."""
    seeds = seeds or list(range(len(policy_sets)))
    envs = []
    trajectories = [[] for _ in policy_sets]
    steps = [0] * len(policy_sets)
    action_counts = [Counter() for _ in policy_sets]
    action_counts_by_player = [[Counter() for _ in range(4)] for _ in policy_sets]
    latencies = [[] for _ in policy_sets]
    latencies_by_player = [[[] for _ in range(4)] for _ in policy_sets]
    inference_batch_sizes = []
    for seed in seeds:
        env = MahjongEnv()
        env.reset(seed=seed)
        envs.append(env)
    while True:
        pending = {}
        active = False
        for index, (env, policies) in enumerate(zip(envs, policy_sets)):
            if env.is_terminal() or steps[index] >= max_steps:
                continue
            active = True
            player = env.current_player
            observation = env.observe(player)
            legal = env.legal_actions(player)
            policy = policies[player]
            pending.setdefault(id(policy), {"policy": policy, "items": []})["items"].append(
                (index, player, observation, legal))
        if not active:
            break
        for group in pending.values():
            policy, items = group["policy"], group["items"]
            if hasattr(policy, "batch_act"):
                inference_batch_sizes.append(len(items))
            started = time.time()
            if hasattr(policy, "batch_act"):
                chosen = policy.batch_act(
                    [item[2] for item in items], [item[3] for item in items])
            else:
                chosen = [policy.act(item[2], item[3]) for item in items]
            elapsed = time.time() - started
            per_action = elapsed / max(1, len(items))
            for (index, player, observation, legal), action in zip(items, chosen):
                env = envs[index]
                latencies[index].append(per_action)
                latencies_by_player[index][player].append(per_action)
                action_counts[index][action.kind.name] += 1
                action_counts_by_player[index][player][action.kind.name] += 1
                if collect:
                    trajectories[index].append({
                        "player": player, "observation": observation,
                        "legal_actions": legal, "action": action,
                        "privileged_hands": [list(hand) for hand in env.hands],
                    })
                env.step(action)
                steps[index] += 1
    results = []
    for index, env in enumerate(envs):
        if not env.is_terminal():
            env.terminal = True
            env.phase = "terminal"
        result = env.result()
        result.update({
            "steps": steps[index], "action_counts": dict(action_counts[index]),
            "action_counts_by_player": [
                dict(value) for value in action_counts_by_player[index]],
            "decision_latencies": latencies[index],
            "decision_latencies_by_player": latencies_by_player[index],
            "inference_batches": len(inference_batch_sizes),
            "mean_inference_batch_size": (
                sum(inference_batch_sizes) / float(max(1, len(inference_batch_sizes)))),
            "max_inference_batch_size": max(inference_batch_sizes) if inference_batch_sizes else 1,
            "tenpai": [
                default_backend.shanten(env._counts(player), env.melds[player]) == 0
                for player in range(4)
            ],
        })
        results.append(result)
    return list(zip(results, trajectories))
