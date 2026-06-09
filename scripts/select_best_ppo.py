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

from mahjong_agent.evaluation import evaluate_duplicate
from mahjong_agent.models.hybrid_transformer import HybridTransformer
from mahjong_agent.policies import HeuristicPolicy
from mahjong_agent.policies.model import ModelPolicy
from mahjong_agent.training.checkpoint import load_checkpoint


def load_policy(path):
    model = HybridTransformer()
    load_checkpoint(path, model)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return ModelPolicy(model)


def evaluate_path(path, games, seed):
    return evaluate_duplicate(
        load_policy(path), HeuristicPolicy(), walls=max(1, games // 4), seed=seed,
        policy_a_name=os.path.basename(path), policy_b_name="heuristic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", required=True)
    parser.add_argument("--ppo-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    results = []
    bc_result = evaluate_path(args.bc, args.games, args.seed)
    results.append({"path": args.bc, "qualified": True, "evaluation": bc_result})
    best_path = args.bc
    best_score = bc_result["average_score_a"]
    bc_lower = bc_result["score_delta_95_ci"][0]
    for path in sorted(glob.glob(args.ppo_glob)):
        result = evaluate_path(path, args.games, args.seed)
        qualified = result["average_score_a"] >= best_score and result["score_delta_95_ci"][0] >= bc_lower
        results.append({"path": path, "qualified": qualified, "evaluation": result})
        if qualified:
            best_path = path
            best_score = result["average_score_a"]
    shutil.copyfile(best_path, args.output)
    sidecar = best_path + ".json"
    if os.path.exists(sidecar):
        shutil.copyfile(sidecar, args.output + ".json")
    with open(args.report, "w") as handle:
        json.dump({"selected": best_path, "results": results}, handle, indent=2, sort_keys=True)
    print("selected %s -> %s" % (best_path, args.output))


if __name__ == "__main__":
    main()
