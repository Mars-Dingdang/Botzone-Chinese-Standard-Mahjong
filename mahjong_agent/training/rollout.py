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
