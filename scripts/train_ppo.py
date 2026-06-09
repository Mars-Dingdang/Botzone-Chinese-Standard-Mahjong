#!/usr/bin/env python3
"""Small single-machine PPO fine-tuner.

For multi-GPU, launch with torchrun; each rank independently collects rollouts
and DDP synchronizes gradient updates.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mahjong_agent.features import encode_action, encode_observation
from mahjong_agent.models.hybrid_transformer import HybridTransformer
from mahjong_agent.policies import HeuristicPolicy, RandomPolicy
from mahjong_agent.policies.model import ModelPolicy
from mahjong_agent.training.checkpoint import load_checkpoint, save_checkpoint
from mahjong_agent.training.dataset import collate_records
from mahjong_agent.training.ppo import ppo_update
from mahjong_agent.training.ppo import generalized_advantage_estimate
from mahjong_agent.training.rollout import play_episode
from mahjong_agent.rules import default_backend


def potential(observation):
    from collections import Counter
    counts = [0] * 34
    for tile, count in Counter(observation["hand"]).items():
        counts[tile] = count
    melds = observation["melds"][observation["player_id"]]
    shanten = default_backend.shanten(counts, melds)
    useful = default_backend.useful_tiles(counts, melds)
    visible = Counter(tile for river in observation["discards"] for tile in river)
    remaining = sum(max(0, 4 - counts[tile] - visible[tile]) for tile in useful)
    return -float(shanten) + 0.1 * float(remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/bc_model.pt")
    parser.add_argument("--output", default="artifacts/ppo_model.pt")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--games-per-update", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--potential-coef", type=float, default=0.02)
    args = parser.parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = HybridTransformer()
    load_checkpoint(args.checkpoint, model)
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_update = 0
    if args.resume:
        metadata = load_checkpoint(args.resume, model.module if distributed else model, optimizer)
        start_update = int(metadata.get("updates", 0))
    learner = ModelPolicy(model, stochastic=True)
    for update in range(start_update, args.updates):
        records = []
        advantages_all = []
        returns_all = []
        for game in range(args.games_per_update):
            rank = torch.distributed.get_rank() if distributed else 0
            learner_seat = (rank + update + game) % 4
            policies = [HeuristicPolicy(), RandomPolicy(update + game), HeuristicPolicy(), RandomPolicy(game)]
            policies[learner_seat] = learner
            result, trajectory = play_episode(
                policies, seed=rank * 1000000 + update * 1000 + game, collect=True
            )
            terminal_reward = float(torch.tanh(torch.tensor(result["scores"][learner_seat] / 64.0)))
            learner_records = [record for record in trajectory if record["player"] == learner_seat]
            game_records = []
            for record in learner_records:
                legal = record["legal_actions"]
                game_records.append({
                    "features": encode_observation(record["observation"]),
                    "actions": [encode_action(action) for action in legal],
                    "target": next(i for i, action in enumerate(legal)
                                   if action.key() == record["action"].key()),
                })
            if not game_records:
                continue
            game_features, game_actions, game_masks, game_chosen = collate_records(game_records, torch)
            with torch.no_grad():
                values = model(game_features.to(device), game_actions.to(device), game_masks.to(device))["value"].cpu().tolist()
            shaped_rewards = [0.0] * len(game_records)
            shaped_rewards[-1] = terminal_reward
            if args.potential_coef:
                potentials = [potential(record["observation"]) for record in learner_records]
                for index in range(len(potentials) - 1):
                    shaped_rewards[index] += args.potential_coef * (args.gamma * potentials[index + 1] - potentials[index])
            game_advantages, game_returns = generalized_advantage_estimate(
                shaped_rewards, values, args.gamma, args.gae_lambda)
            records.extend(game_records)
            advantages_all.extend(game_advantages)
            returns_all.extend(game_returns)
        if not records:
            continue
        features, actions, masks, chosen = collate_records(records, torch)
        features = features.to(device)
        actions = actions.to(device)
        masks = masks.to(device)
        chosen = chosen.to(device)
        with torch.no_grad():
            old = model(features, actions, masks)
            old_log_probs = torch.distributions.Categorical(
                logits=old["logits"]
            ).log_prob(chosen)
        returns = torch.tensor(returns_all, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages_all, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        batch = dict(features=features, actions=actions, masks=masks, chosen=chosen,
                     old_log_probs=old_log_probs, advantages=advantages, returns=returns)
        metrics = ppo_update(model, optimizer, batch, target_kl=args.target_kl)
        if not distributed or torch.distributed.get_rank() == 0:
            print("update=%d metrics=%r" % (update + 1, metrics), flush=True)
            if (update + 1) % args.save_every == 0:
                saved_model = model.module if distributed else model
                save_checkpoint(args.output, saved_model, optimizer, {"algorithm": "ppo", "updates": update + 1})
    if not distributed or torch.distributed.get_rank() == 0:
        saved_model = model.module if distributed else model
        save_checkpoint(args.output, saved_model, optimizer,
                        {"algorithm": "ppo", "updates": args.updates})
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
