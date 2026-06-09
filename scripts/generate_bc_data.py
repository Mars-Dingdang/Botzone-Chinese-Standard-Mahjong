#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_agent.policies import HeuristicPolicy
from mahjong_agent.training.dataset import record_to_json
from mahjong_agent.training.rollout import play_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--output", default="artifacts/bc_data.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    policies = [HeuristicPolicy() for _ in range(4)]
    records = 0
    with open(args.output, "w") as handle:
        for game in range(args.games):
            _, trajectory = play_episode(policies, seed=args.seed + game, collect=True)
            for record in trajectory:
                handle.write(json.dumps(record_to_json(record)) + "\n")
                records += 1
    print("wrote %d records to %s" % (records, args.output))


if __name__ == "__main__":
    main()
