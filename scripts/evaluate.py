#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_agent.evaluation import (evaluate, evaluate_duplicate,
                                      load_wall_manifest, save_wall_manifest)
from mahjong_agent.policies import HeuristicPolicy, RandomPolicy


def load_policy(path):
    if not path:
        return HeuristicPolicy()
    import torch
    from mahjong_agent.policies.model import ModelPolicy
    from mahjong_agent.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return ModelPolicy(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--duplicate", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--policy-name", default="model")
    parser.add_argument("--wall-manifest", default="")
    parser.add_argument("--save-wall-manifest", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    policy = load_policy(args.model)
    if args.duplicate:
        manifest = load_wall_manifest(args.wall_manifest, max(1, args.games // 4), args.seed)
        if args.save_wall_manifest:
            save_wall_manifest(args.save_wall_manifest, manifest)
        result = evaluate_duplicate(
            policy, HeuristicPolicy(), walls=max(1, args.games // 4),
            seed=args.seed, manifest=manifest, policy_a_name=args.policy_name,
            policy_b_name="heuristic", progress=args.progress)
    else:
        result = evaluate([policy, HeuristicPolicy(), RandomPolicy(2), RandomPolicy(3)],
                          games=args.games, seed=args.seed, progress=args.progress)
    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
