#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/bc_model.pt")
    args = parser.parse_args()
    import torch
    from mahjong_agent.models.hybrid_transformer import HybridTransformer
    from mahjong_agent.training.checkpoint import load_checkpoint
    model = HybridTransformer()
    metadata = load_checkpoint(args.model, model)
    print("model loaded; torch=%s metadata=%r" % (torch.__version__, metadata))


if __name__ == "__main__":
    main()
