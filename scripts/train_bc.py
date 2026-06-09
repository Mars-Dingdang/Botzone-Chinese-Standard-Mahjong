#!/usr/bin/env python3
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mahjong_agent.models.hybrid_transformer import HybridTransformer
from mahjong_agent.training.checkpoint import save_checkpoint
from mahjong_agent.training.dataset import collate_records, load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/bc_data.jsonl")
    parser.add_argument("--output", default="artifacts/bc_model.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    records = load_jsonl(args.data)
    model = HybridTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        random.shuffle(records)
        correct = total = 0
        for start in range(0, len(records), args.batch_size):
            batch = collate_records(records[start:start + args.batch_size], torch)
            features, actions, masks, targets = batch
            output = model(features, actions, masks)
            loss = torch.nn.functional.cross_entropy(output["logits"], targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += int((output["logits"].argmax(1) == targets).sum().item())
            total += len(targets)
        print("epoch=%d accuracy=%.4f" % (epoch + 1, correct / float(total)))
    save_checkpoint(args.output, model, optimizer, {"algorithm": "bc", "epochs": args.epochs})


if __name__ == "__main__":
    main()
