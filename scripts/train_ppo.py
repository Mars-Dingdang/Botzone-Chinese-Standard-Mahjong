#!/usr/bin/env python3
"""Feature V2 PPO trainer with mixed opponents and auditable rewards."""
import argparse
import copy
import glob
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mahjong_agent.features import encode_action_v2, encode_observation_v2
from mahjong_agent.models import create_model
from mahjong_agent.evaluation import create_wall_manifest, evaluate_duplicate
from mahjong_agent.policies import HeuristicPolicy
from mahjong_agent.policies.analysis import direct_deal_in_index
from mahjong_agent.policies.model import ModelPolicy
from mahjong_agent.training.checkpoint import (load_checkpoint,
                                               load_model_from_checkpoint,
                                               save_checkpoint)
from mahjong_agent.training.dataset import collate_records
from mahjong_agent.training.ppo import (generalized_advantage_estimate, ppo_update,
                                        rollout_game_indices, terminal_only_rewards)
from mahjong_agent.training.reward import public_potential, shaped_rewards
from mahjong_agent.training.rollout import play_episodes_vectorized


def jsonl(path, value):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def padded_record(record, result, learner_seat, direct_deal_in=False, max_actions=64):
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
        "action_aux_labels": [int(result["winner"] == learner_seat),
                              int(direct_deal_in),
                              result["scores"][learner_seat] / 64.0,
                              int(result["winner"] == learner_seat and fan >= 8)],
        "fan_target": min(4, fan // 8) if result["winner"] == learner_seat else 0,
        "belief_targets": belief,
    }


class OpponentPool(object):
    def __init__(self, bc_model, seed=0, history_capacity=8, mix=None):
        self.bc_model = bc_model
        self.random = random.Random(seed)
        self.history = []
        self.latest_model = None
        self.best_model = None
        self.best_score = None
        self.snapshots_seen = 0
        self.history_capacity = history_capacity
        self.mix = mix or {
            "bc": .40, "ppo_latest": .20, "ppo_best": .15,
            "ppo_history": .20, "heuristic": .05,
        }
        self._policies = {"bc": ModelPolicy(bc_model), "heuristic": HeuristicPolicy()}

    @staticmethod
    def _snapshot(model):
        snapshot = create_model(
            model.feature_version, **dict(getattr(model, "model_config", {})))
        if hasattr(snapshot, "belief_mode"):
            snapshot.belief_mode = getattr(model, "belief_mode", snapshot.belief_mode)
        snapshot.load_state_dict({
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        })
        snapshot.eval()
        return snapshot

    def add_history(self, model):
        snapshot = self._snapshot(model)
        self.latest_model = snapshot
        self.snapshots_seen += 1
        if len(self.history) < self.history_capacity:
            self.history.append(snapshot)
        elif self.history_capacity:
            replace = self.random.randrange(self.snapshots_seen)
            if replace < self.history_capacity:
                self.history[replace] = snapshot
        self._policies.clear()
        self._policies.update({"bc": ModelPolicy(self.bc_model), "heuristic": HeuristicPolicy()})
        return snapshot

    def update_best(self, model, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_model = self._snapshot(model)
            self._policies.pop("ppo_best", None)
            return True
        return False

    def _model_policy(self, name, model):
        key = "%s:%d" % (name, id(model))
        if key not in self._policies:
            self._policies[key] = ModelPolicy(model)
        return self._policies[key]

    def sample(self, seed):
        names = ("bc", "ppo_latest", "ppo_best", "ppo_history", "heuristic")
        value, selected = self.random.random(), "bc"
        cumulative = 0.0
        for name in names:
            cumulative += float(self.mix.get(name, 0.0))
            if value < cumulative:
                selected = name
                break
        if selected == "heuristic":
            return self._policies["heuristic"], selected
        if selected == "ppo_latest" and self.latest_model is not None:
            return self._model_policy(selected, self.latest_model), selected
        if selected == "ppo_best" and self.best_model is not None:
            return self._model_policy(selected, self.best_model), selected
        if selected == "ppo_history" and self.history:
            model = self.random.choice(self.history)
            return self._model_policy(selected, model), selected
        return self._policies["bc"], "bc"


def _yaml(path):
    if not path:
        return {}
    import yaml
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def main():
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--config", default="configs/train/ppo.yaml")
    known, _ = early.parse_known_args()
    cfg = _yaml(known.config)
    reward_cfg = cfg.get("reward", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/ppo_model.pt")
    parser.add_argument("--updates", type=int, default=cfg.get("updates", 100))
    parser.add_argument("--games-per-update", type=int, default=cfg.get("games_per_update", 8))
    parser.add_argument("--rollout-envs", type=int, default=cfg.get("rollout_envs", 8))
    parser.add_argument("--lr", type=float, default=cfg.get("learning_rate", 1e-4))
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--history-every", type=int, default=cfg.get("history_every", 10))
    parser.add_argument("--history-capacity", type=int, default=cfg.get("history_capacity", 8))
    parser.add_argument("--league-games", type=int, default=cfg.get("league_games", 8))
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--target-kl", type=float, default=cfg.get("target_kl", .02))
    parser.add_argument("--bc-kl-coef", type=float, default=cfg.get("bc_kl_coef", .01))
    parser.add_argument("--clip-ratio", type=float, default=cfg.get("clip_ratio", .2))
    parser.add_argument("--value-coef", type=float, default=cfg.get("value_coef", .5))
    parser.add_argument("--entropy-coef", type=float, default=cfg.get("entropy_coef", .01))
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--reward-mode", choices=("terminal_only", "shaped"),
                        default=reward_cfg.get("mode", "shaped"))
    parser.add_argument("--score-scale", type=float, default=reward_cfg.get("score_scale", 64.0))
    parser.add_argument("--qualifying-win-bonus", type=float, default=reward_cfg.get("qualifying_win_bonus", .5))
    parser.add_argument("--direct-deal-in-penalty", type=float, default=reward_cfg.get("direct_deal_in_penalty", .25))
    parser.add_argument("--step-shaping-cap", type=float, default=reward_cfg.get("step_shaping_cap", .1))
    parser.add_argument("--episode-shaping-cap", type=float, default=reward_cfg.get("episode_shaping_cap", 1.0))
    parser.add_argument("--efficiency-coef", type=float, default=reward_cfg.get("efficiency_coef", .01))
    parser.add_argument("--fan-feasibility-coef", type=float, default=reward_cfg.get("fan_feasibility_coef", .05))
    parser.add_argument("--deal-in-risk-coef", type=float, default=reward_cfg.get("deal_in_risk_coef", .03))
    parser.add_argument("--draw-tenpai-coef", type=float, default=reward_cfg.get("draw_tenpai_coef", .0))
    parser.add_argument("--aux-coef", type=float, default=.05)
    parser.add_argument("--deal-in-pos-weight", type=float,
                        default=cfg.get("deal_in_pos_weight", 3.0))
    parser.add_argument("--direct-deal-in-pos-weight", type=float,
                        default=cfg.get("direct_deal_in_pos_weight", 8.0))
    parser.add_argument("--belief-mode", choices=("none", "aux", "actor"),
                        default=cfg.get("belief_mode", "aux"))
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.games_per_update < world or args.games_per_update % world:
        raise ValueError(
            "games_per_update must be divisible by WORLD_SIZE and at least WORLD_SIZE")
    if distributed:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model, checkpoint_meta = load_model_from_checkpoint(args.checkpoint)
    if int(getattr(model, "feature_version", 1)) != 2:
        raise ValueError("PPO requires a Feature V2 BC checkpoint")
    model.belief_mode = args.belief_mode
    if (args.belief_mode == "actor" and checkpoint_meta.get("belief_mode", "aux") != "actor"):
        torch.nn.init.zeros_(model.belief_adapter.weight)
        torch.nn.init.zeros_(model.belief_adapter.bias)
    model.to(device)
    reference, _ = load_model_from_checkpoint(args.checkpoint)
    reference.belief_mode = args.belief_mode
    reference.to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_update = 0
    if args.resume:
        metadata = load_checkpoint(args.resume, model, optimizer)
        start_update = int(metadata.get("updates", 0))
        random.setstate(tuple(metadata["random_state"]) if metadata.get("random_state") else random.getstate())
    pool = OpponentPool(reference, args.seed + rank, args.history_capacity,
                        cfg.get("opponent_mix"))
    if args.resume:
        for path in sorted(glob.glob(args.output.replace(".pt", ".update-*.pt")))[-args.history_capacity:]:
            historical, _ = load_model_from_checkpoint(path)
            pool.history.append(historical.eval())
        if pool.history:
            pool.latest_model = pool.history[-1]
            pool.best_model = pool.history[-1]
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
        rollout_started = time.time()
        learner_policy = ModelPolicy(model, stochastic=True)
        jobs = []
        # games_per_update is the global rollout budget. DDP ranks process
        # disjoint game indices so two GPUs increase throughput without
        # silently multiplying the requested sample budget.
        for game in rollout_game_indices(args.games_per_update, rank, world):
            learner_seat = (update + game) % 4
            policies = [None] * 4
            for seat in range(4):
                if seat == learner_seat:
                    policies[seat] = learner_policy
                else:
                    policies[seat], name = pool.sample(
                        args.seed + update * 1000 + game * 4 + seat)
                    opponent_counts[name] += 1
            jobs.append((game, learner_seat, policies))
        outcomes = []
        rollout_envs = max(1, args.rollout_envs)
        for start in range(0, len(jobs), rollout_envs):
            wave = jobs[start:start + rollout_envs]
            outcomes.extend(play_episodes_vectorized(
                [item[2] for item in wave],
                [args.seed + update * 1000000 + item[0] for item in wave],
                collect=True))
        for (_, learner_seat, _), (result, trajectory) in zip(jobs, outcomes):
            raw_score = result["scores"][learner_seat]
            raw_scores.append(raw_score)
            terminal = float(torch.tanh(torch.tensor(raw_score / args.score_scale)))
            learner_records = [item for item in trajectory if item["player"] == learner_seat]
            direct_index = direct_deal_in_index(learner_records, result, learner_seat)
            if args.reward_mode == "shaped":
                if direct_index >= 0:
                    terminal -= args.direct_deal_in_penalty
                if result["winner"] == learner_seat and result["fan_count"] >= 8:
                    terminal += args.qualifying_win_bonus
            game_records = [padded_record(item, result, learner_seat, index == direct_index)
                            for index, item in enumerate(learner_records)]
            if not game_records:
                continue
            batch = collate_records(game_records, torch)
            features, feature_masks, actions, action_token_masks, masks = batch[:5]
            with torch.no_grad():
                values = model(features.to(device), actions.to(device), masks.to(device),
                               feature_masks.to(device), action_token_masks.to(device))["value"].cpu().tolist()
            if args.reward_mode == "terminal_only":
                rewards = terminal_only_rewards(len(learner_records), terminal)
                reward_details = [{"terminal_reward": terminal}
                                  if index == len(learner_records) - 1 else {}
                                  for index in range(len(learner_records))]
            else:
                potentials = [
                    public_potential(item["observation"], coefficients, item["action"])
                    for item in learner_records
                ]
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
        batch["action_aux_labels"] = torch.tensor(
            [record["action_aux_labels"] for record in records],
            dtype=torch.float32, device=device)
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
            model, optimizer, batch, clip_ratio=args.clip_ratio,
            value_coef=args.value_coef, entropy_coef=args.entropy_coef,
            target_kl=args.target_kl,
            bc_kl_coef=args.bc_kl_coef, minibatch_size=args.minibatch_size,
            aux_coef=0.0 if args.belief_mode == "none" else args.aux_coef,
            deal_in_pos_weight=args.deal_in_pos_weight,
            direct_deal_in_pos_weight=args.direct_deal_in_pos_weight)
        metrics.update({
            "update": update + 1, "raw_score_mean": sum(raw_scores) / max(1, len(raw_scores)),
            "reward_mode": args.reward_mode,
            "rollout_games_global": args.games_per_update,
            "rollout_games_rank": len(raw_scores),
            "opponent_samples": dict(opponent_counts),
            "reward_components": dict(reward_totals),
            "rollout_games_per_second": len(raw_scores) / max(1e-6, time.time() - rollout_started),
            "rollout_decisions_per_second": len(records) / max(1e-6, time.time() - rollout_started),
            "mean_inference_batch_size": sum(
                result.get("mean_inference_batch_size", 1.0) for result, _ in outcomes
            ) / max(1, len(outcomes)),
            "max_inference_batch_size": max(
                [result.get("max_inference_batch_size", 1) for result, _ in outcomes] or [1]),
        })
        saved = model.module if distributed else model
        if (update + 1) % args.history_every == 0:
            snapshot = pool.add_history(saved)
            league = evaluate_duplicate(
                ModelPolicy(snapshot), ModelPolicy(reference),
                walls=max(1, args.league_games // 4), seed=args.seed,
                manifest=create_wall_manifest(max(1, args.league_games // 4), args.seed))
            pool.update_best(snapshot, (
                league.get("qualifying_win_rate", league["win_rate"]),
                league["average_score"]))
            metrics["league_average_score"] = league["average_score"]
            metrics["league_qualifying_win_rate"] = league.get(
                "qualifying_win_rate", league["win_rate"])
        if not distributed or rank == 0:
            jsonl(args.metrics_jsonl, metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if (update + 1) % args.save_every == 0:
                path = args.output.replace(".pt", ".update-%04d.pt" % (update + 1))
                save_checkpoint(path, saved, optimizer, {
                    "algorithm": "ppo", "updates": update + 1, "metrics": metrics,
                    "seed": args.seed, "opponent_pool": dict(opponent_counts),
                    "reward_coefficients": coefficients,
                    "reward_mode": args.reward_mode,
                    "belief_mode": args.belief_mode,
                    "games_per_update": args.games_per_update,
                    "rollout_envs": args.rollout_envs,
                    "league_games": args.league_games,
                    "learning_rate": args.lr,
                    "target_kl": args.target_kl,
                    "bc_kl_coef": args.bc_kl_coef,
                })
    if not distributed or rank == 0:
        saved = model.module if distributed else model
        save_checkpoint(args.output, saved, optimizer, {
            "algorithm": "ppo", "updates": args.updates, "seed": args.seed,
            "reward_coefficients": coefficients,
            "reward_mode": args.reward_mode,
            "belief_mode": args.belief_mode,
            "games_per_update": args.games_per_update,
            "rollout_envs": args.rollout_envs,
            "league_games": args.league_games,
            "learning_rate": args.lr,
            "target_kl": args.target_kl,
            "bc_kl_coef": args.bc_kl_coef,
        })
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
