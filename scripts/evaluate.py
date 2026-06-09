#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_agent.evaluation import evaluate, evaluate_duplicate
from mahjong_agent.policies import HeuristicPolicy, RandomPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--duplicate", action="store_true")
    args = parser.parse_args()
    if args.duplicate:
        result = evaluate_duplicate(HeuristicPolicy(), RandomPolicy(0), walls=max(1, args.games // 4))
    else:
        result = evaluate([HeuristicPolicy(), RandomPolicy(1), RandomPolicy(2), RandomPolicy(3)],
                          games=args.games)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
