#!/usr/bin/env python3
"""Select a BC epoch checkpoint by fixed-wall duplicate score."""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mahjong_agent.evaluation import evaluate_duplicate, load_wall_manifest, save_wall_manifest
from mahjong_agent.policies import HeuristicPolicy
from mahjong_agent.policies.model import ModelPolicy
from mahjong_agent.training.checkpoint import load_model_from_checkpoint


def evaluate_path(path, games, seed, manifest):
    model, metadata = load_model_from_checkpoint(path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_duplicate(
        ModelPolicy(model), HeuristicPolicy(), walls=max(1, games // 4), seed=seed,
        manifest=manifest, policy_a_name=os.path.basename(path),
        policy_b_name="heuristic")
    return result, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--wall-manifest", required=True)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.checkpoint_glob))
    if not paths:
        raise ValueError("no BC checkpoints match %s" % args.checkpoint_glob)
    manifest = load_wall_manifest(args.wall_manifest, max(1, args.games // 4), args.seed)
    if not os.path.exists(args.wall_manifest):
        save_wall_manifest(args.wall_manifest, manifest)
    os.makedirs(args.results_dir, exist_ok=True)
    results = []
    best_path = None
    best_key = None
    for path in paths:
        evaluation, metadata = evaluate_path(path, args.games, args.seed, manifest)
        validation = metadata.get("val", {})
        key = (evaluation.get("qualifying_win_rate", evaluation["win_rate"]),
               evaluation["average_score"],
               validation.get("eight_fan_exact_accuracy", 0.0),
               validation.get("exact_accuracy", validation.get("accuracy", 0.0)))
        results.append({"path": path, "evaluation": evaluation,
                        "validation": metadata.get("val", {})})
        with open(os.path.join(
                args.results_dir, os.path.basename(path) + ".eval.json"), "w") as handle:
            json.dump(evaluation, handle, indent=2, sort_keys=True)
        if best_key is None or key > best_key:
            best_key = key
            best_path = path
    shutil.copyfile(best_path, args.output)
    if os.path.exists(best_path + ".json"):
        shutil.copyfile(best_path + ".json", args.output + ".json")
    with open(args.report, "w") as handle:
        json.dump({"selected": best_path, "selection_rule":
                   "qualifying_win_rate_then_average_score_then_eight_fan_accuracy",
                   "results": results}, handle, indent=2, sort_keys=True)
    print("selected %s -> %s" % (best_path, args.output))


if __name__ == "__main__":
    main()
