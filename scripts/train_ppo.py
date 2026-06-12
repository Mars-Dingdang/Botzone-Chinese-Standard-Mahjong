#!/usr/bin/env python3
"""Feature V2 PPO trainer with mixed opponents and auditable rewards."""
import argparse
import copy
import glob
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mahjong_agent.features import encode_action_v2, encode_observation_v2
from mahjong_agent.models import create_model
from mahjong_agent.policies import HeuristicPolicy, RandomPolicy
from mahjong_agent.policies.model import ModelPolicy
from mahjong_agent.training.checkpoint import load_checkpoint, save_checkpoint
from mahjong_agent.training.dataset import collate_records
from mahjong_agent.training.ppo import generalized_advantage_estimate, ppo_update
from mahjong_agent.training.reward import public_potential, shaped_rewards
from mahjong_agent.training.rollout import play_episode


def jsonl(path, value):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def padded_record(record, result, learner_seat, max_actions=64):
    features, feature_mask = encode_observation_v2(record["observation"])
    encoded = [encode_action_v2(action) for action in record["legal_actions"]]
    count = len(encoded)
    if count > max_actions:
        raise ValueError("action count exceeds max_actions=%d" % max_actions)
    zero_action = [[[0.0] * 12] * 4][0]
    actions = [item[0] for item in encoded] + [copy.deepcopy(zero_action) for _ in range(max_actions - count)]
    token_masks = [item[1] for item in encoded] + [[0] * 4 for _ in range(max_actions - count)]
    target = next(i for i, action in enumerate(record["legal_actions"])
                  if action.key() == record["action"].key())
    belief = []
    for relative in (1, 2, 3):
        hand = record["privileged_hands"][(learner_seat + relative) % 4]
        belief.append([hand.count(tile) for tile in range(34)])
    fan = result["fan_count"]
    return {
        "features": features, "feature_mask": feature_mask, "actions": actions,
        "action_token_masks": token_masks, "mask": [1] * count + [0] * (max_actions - count),
        "target": target,
        "aux_labels": [int(result["winner"] == learner_seat),
                       int(result["loser"] == learner_seat),
                       result["scores"][learner_seat] / 64.0,
                       int(result["winner"] == learner_seat and fan >= 8)],
        "fan_target": min(4, fan // 8), "belief_targets": belief,
    }


class OpponentPool(object):
    def __init__(self, current_model, bc_model, seed=0, history_capacity=8):
        self.current_model = current_model
        self.best_model = bc_model
        self.bc_model = bc_model
        self.random = random.Random(seed)
        self.history = []
        self.history_capacity = history_capacity

    def add_history(self, model):
        snapshot = create_model(2)
        snapshot.load_state_dict(copy.deepcopy(model.state_dict()))
        snapshot.eval()
        self.history.append(snapshot)
        self.history = self.history[-self.history_capacity:]

    def sample(self, seed):
        value = self.random.random()
        if value < 0.40:
            selected = self.current_model if self.random.random() < .5 else self.best_model
            return ModelPolicy(selected), "current_best"
        if value < 0.70 and self.history:
            return ModelPolicy(self.random.choice(self.history)), "history"
        if value < 0.95:
            return (ModelPolicy(self.bc_model), "bc") if self.random.random() < .5 else (
                HeuristicPolicy(), "heuristic")
        return RandomPolicy(seed), "random"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/ppo_model.pt")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--games-per-update", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--history-every", type=int, default=10)
    parser.add_argument("--history-capacity", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--target-kl", type=float, default=.02)
    parser.add_argument("--bc-kl-coef", type=float, default=.01)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--score-scale", type=float, default=64.0)
    parser.add_argument("--step-shaping-cap", type=float, default=.1)
    parser.add_argument("--episode-shaping-cap", type=float, default=1.0)
    parser.add_argument("--efficiency-coef", type=float, default=.02)
    parser.add_argument("--fan-feasibility-coef", type=float, default=.02)
    parser.add_argument("--deal-in-risk-coef", type=float, default=.01)
    parser.add_argument("--draw-tenpai-coef", type=float, default=.0)
    parser.add_argument("--aux-coef", type=float, default=.05)
    parser.add_argument("--belief-mode", choices=("none", "aux", "actor"), default="aux")
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if distributed:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = create_model(2)
    model.belief_mode = args.belief_mode
    load_checkpoint(args.checkpoint, model)
    model.to(device)
    reference = create_model(2)
    load_checkpoint(args.checkpoint, reference)
    reference.to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_update = 0
    if args.resume:
        metadata = load_checkpoint(args.resume, model, optimizer)
        start_update = int(metadata.get("updates", 0))
        random.setstate(tuple(metadata["random_state"]) if metadata.get("random_state") else random.getstate())
    pool = OpponentPool(model, reference, args.seed + rank, args.history_capacity)
    if args.resume:
        for path in sorted(glob.glob(args.output.replace(".pt", ".update-*.pt")))[-args.history_capacity:]:
            historical = create_model(2)
            load_checkpoint(path, historical)
            pool.history.append(historical.eval())
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=True)
    coefficients = {
        "efficiency": args.efficiency_coef, "fan_feasibility": args.fan_feasibility_coef,
        "deal_in_risk": args.deal_in_risk_coef, "draw_tenpai": args.draw_tenpai_coef,
    }
    from tqdm import tqdm
    updates = tqdm(range(start_update, args.updates), desc="ppo", unit="update",
                   disable=not sys.stderr.isatty())
    for update in updates:
        pool.random.seed(args.seed + rank + update * 100003)
        records = []
        advantages_all = []
        returns_all = []
        opponent_counts = Counter()
        raw_scores = []
        reward_totals = Counter()
        for game in range(args.games_per_update):
            learner_seat = (rank + update + game) % 4
            policies = []
            for seat in range(4):
                policy, name = pool.sample(args.seed + update * 1000 + game * 4 + seat)
                policies.append(policy)
                opponent_counts[name] += int(seat != learner_seat)
            policies[learner_seat] = ModelPolicy(model, stochastic=True)
            result, trajectory = play_episode(
                policies, seed=args.seed + rank * 1000000 + update * 1000 + game,
                collect=True)
            raw_score = result["scores"][learner_seat]
            raw_scores.append(raw_score)
            terminal = float(torch.tanh(torch.tensor(raw_score / args.score_scale)))
            learner_records = [item for item in trajectory if item["player"] == learner_seat]
            game_records = [padded_record(item, result, learner_seat) for item in learner_records]
            if not game_records:
                continue
            batch = collate_records(game_records, torch)
            features, feature_masks, actions, action_token_masks, masks = batch[:5]
            with torch.no_grad():
                values = model(features.to(device), actions.to(device), masks.to(device),
                               feature_masks.to(device), action_token_masks.to(device))["value"].cpu().tolist()
            potentials = [public_potential(item["observation"], coefficients) for item in learner_records]
            rewards, reward_details = shaped_rewards(
                potentials, terminal, args.gamma, args.step_shaping_cap,
                args.episode_shaping_cap)
            for detail in reward_details:
                reward_totals.update(detail)
            advantages, returns = generalized_advantage_estimate(
                rewards, values, args.gamma, args.gae_lambda)
            records.extend(game_records)
            advantages_all.extend(advantages)
            returns_all.extend(returns)
        if not records:
            continue
        batch_values = collate_records(records, torch)
        keys = ("features", "feature_masks", "actions", "action_token_masks", "masks",
                "chosen", "aux_labels", "fan_targets", "belief_targets")
        batch = {key: value.to(device) for key, value in zip(keys, batch_values)}
        with torch.no_grad():
            model.eval()
            old = model(batch["features"], batch["actions"], batch["masks"],
                        batch["feature_masks"], batch["action_token_masks"])
            batch["old_log_probs"] = torch.distributions.Categorical(
                logits=old["logits"]).log_prob(batch["chosen"])
            ref = reference(batch["features"], batch["actions"], batch["masks"],
                            batch["feature_masks"], batch["action_token_masks"])
            batch["reference_logits"] = ref["logits"]
        batch["returns"] = torch.tensor(returns_all, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages_all, dtype=torch.float32, device=device)
        batch["advantages"] = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8)
        metrics = ppo_update(
            model, optimizer, batch, target_kl=args.target_kl,
            bc_kl_coef=args.bc_kl_coef, minibatch_size=args.minibatch_size,
            aux_coef=0.0 if args.belief_mode == "none" else args.aux_coef)
        metrics.update({
            "update": update + 1, "raw_score_mean": sum(raw_scores) / max(1, len(raw_scores)),
            "opponent_samples": dict(opponent_counts),
            "reward_components": dict(reward_totals),
        })
        saved = model.module if distributed else model
        if (update + 1) % args.history_every == 0:
            pool.add_history(saved)
        if not distributed or rank == 0:
            jsonl(args.metrics_jsonl, metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if (update + 1) % args.save_every == 0:
                path = args.output.replace(".pt", ".update-%04d.pt" % (update + 1))
                save_checkpoint(path, saved, optimizer, {
                    "algorithm": "ppo", "updates": update + 1, "metrics": metrics,
                    "seed": args.seed, "opponent_pool": dict(opponent_counts),
                    "reward_coefficients": coefficients,
                    "belief_mode": args.belief_mode,
                })
    if not distributed or rank == 0:
        saved = model.module if distributed else model
        save_checkpoint(args.output, saved, optimizer, {
            "algorithm": "ppo", "updates": args.updates, "seed": args.seed,
            "reward_coefficients": coefficients,
            "belief_mode": args.belief_mode,
        })
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
