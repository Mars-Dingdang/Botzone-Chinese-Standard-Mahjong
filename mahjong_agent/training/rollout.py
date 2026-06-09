"""Self-play rollout utilities shared by data generation and PPO."""

from mahjong_agent.engine import MahjongEnv


def play_episode(policies, seed=None, wall=None, max_steps=512, collect=False):
    env = MahjongEnv()
    env.reset(seed=seed, wall=wall)
    trajectory = []
    steps = 0
    while not env.is_terminal() and steps < max_steps:
        player = env.current_player
        observation = env.observe(player)
        legal = env.legal_actions(player)
        action = policies[player].act(observation, legal)
        if collect:
            trajectory.append({
                "player": player,
                "observation": observation,
                "legal_actions": legal,
                "action": action,
            })
        env.step(action)
        steps += 1
    if not env.is_terminal():
        env.terminal = True
        env.phase = "terminal"
    result = env.result()
    result["steps"] = steps
    return result, trajectory
