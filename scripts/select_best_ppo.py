#!/usr/bin/env python3
"""Evaluate BC and PPO checkpoints on identical duplicate walls."""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mahjong_agent.evaluation import (evaluate_duplicate, load_wall_manifest,
                                      paired_delta, save_wall_manifest)
from mahjong_agent.policies import HeuristicPolicy
from mahjong_agent.policies.model import ModelPolicy
from mahjong_agent.training.checkpoint import load_model_from_checkpoint


def load_policy(path):
    model, _ = load_model_from_checkpoint(path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return ModelPolicy(model)


def evaluate_path(path, games, seed, manifest):
    return evaluate_duplicate(
        load_policy(path), HeuristicPolicy(), walls=max(1, games // 4), seed=seed,
        policy_a_name=os.path.basename(path), policy_b_name="heuristic",
        manifest=manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", required=True)
    parser.add_argument("--ppo-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--wall-manifest", default="")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--min-score-ci-lower", type=float, default=-1.0)
    parser.add_argument("--max-deal-in-increase", type=float, default=.01)
    args = parser.parse_args()
    results_dir = args.results_dir or os.path.join(
        os.path.dirname(args.report) or ".", "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    results = []
    manifest = load_wall_manifest(args.wall_manifest, max(1, args.games // 4), args.seed)
    if args.wall_manifest and not os.path.exists(args.wall_manifest):
        save_wall_manifest(args.wall_manifest, manifest)
    bc_result = evaluate_path(args.bc, args.games, args.seed, manifest)
    with open(os.path.join(results_dir, "bc_eval.json"), "w") as handle:
        json.dump(bc_result, handle, indent=2, sort_keys=True)
    results.append({"path": args.bc, "qualified": True, "evaluation": bc_result})
    best_path = args.bc
    best_key = (bc_result.get("qualifying_win_rate", bc_result["win_rate"]),
                bc_result["average_score"])
    for path in sorted(glob.glob(args.ppo_glob)):
        result = evaluate_path(path, args.games, args.seed, manifest)
        with open(os.path.join(
                results_dir, os.path.basename(path) + ".eval.json"), "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        delta = paired_delta(result, bc_result)
        qualified = (
            delta["score_delta_95_ci"][0] >= args.min_score_ci_lower and
            result["deal_in_rate"] <= bc_result["deal_in_rate"] + args.max_deal_in_increase
        )
        results.append({"path": path, "qualified": qualified, "evaluation": result,
                        "relative_to_bc": delta})
        key = (result.get("qualifying_win_rate", result["win_rate"]),
               result["average_score"])
        if qualified and key > best_key:
            best_path = path
            best_key = key
    shutil.copyfile(best_path, args.output)
    sidecar = best_path + ".json"
    if os.path.exists(sidecar):
        shutil.copyfile(sidecar, args.output + ".json")
    with open(args.report, "w") as handle:
        json.dump({"selected": best_path, "selection_rule":
                   "guardrails_then_qualifying_win_rate_then_average_score",
                   "min_score_ci_lower": args.min_score_ci_lower,
                   "max_deal_in_increase": args.max_deal_in_increase,
                   "results": results}, handle, indent=2, sort_keys=True)
    print("selected %s -> %s" % (best_path, args.output))


if __name__ == "__main__":
    main()
