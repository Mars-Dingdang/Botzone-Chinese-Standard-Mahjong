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
from mahjong_agent.training.rollout import play_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/bc_model.pt")
    parser.add_argument("--output", default="artifacts/ppo_model.pt")
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--games-per-update", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    learner = ModelPolicy(model, stochastic=True)
    for update in range(args.updates):
        records = []
        rewards = []
        for game in range(args.games_per_update):
            policies = [learner, HeuristicPolicy(), RandomPolicy(update), HeuristicPolicy()]
            rank = torch.distributed.get_rank() if distributed else 0
            result, trajectory = play_episode(
                policies, seed=rank * 1000000 + update * 1000 + game, collect=True
            )
            reward = result["scores"][0] / 24.0
            for record in trajectory:
                if record["player"] != 0:
                    continue
                legal = record["legal_actions"]
                records.append({
                    "features": encode_observation(record["observation"]),
                    "actions": [encode_action(action) for action in legal],
                    "target": next(i for i, action in enumerate(legal)
                                   if action.key() == record["action"].key()),
                })
                rewards.append(reward)
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
        returns = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = returns - old["value"].detach()
        batch = dict(features=features, actions=actions, masks=masks, chosen=chosen,
                     old_log_probs=old_log_probs, advantages=advantages, returns=returns)
        metrics = ppo_update(model, optimizer, batch)
        print("update=%d metrics=%r" % (update + 1, metrics))
    if not distributed or torch.distributed.get_rank() == 0:
        saved_model = model.module if distributed else model
        save_checkpoint(args.output, saved_model, optimizer,
                        {"algorithm": "ppo", "updates": args.updates})
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
