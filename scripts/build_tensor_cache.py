#!/usr/bin/env python3
"""Build fixed-shape PyTorch tensor shards from archival Parquet BC data."""
import argparse, glob, json, multiprocessing as mp, os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_agent.features import (FEATURE_SIZE, deserialize_action,
                                    encode_action, encode_observation,
                                    expand_observation)

ACTION_SIZE = 8


def convert(task):
    path, output_dir, max_actions = task
    import pyarrow.parquet as pq
    import torch
    table = pq.read_table(path)
    rows = table.to_pylist()
    result = []
    stem = os.path.splitext(os.path.basename(path))[0]
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        if not selected: continue
        n = len(selected)
        features = torch.empty(n, FEATURE_SIZE, dtype=torch.float16)
        actions = torch.zeros(n, max_actions, ACTION_SIZE, dtype=torch.float16)
        masks = torch.zeros(n, max_actions, dtype=torch.bool)
        targets = torch.tensor([row["target"] for row in selected], dtype=torch.long)
        for index, row in enumerate(selected):
            if "features" in row:
                features[index] = torch.tensor(row["features"], dtype=torch.float16)
                encoded_actions = row["actions"]
            else:
                observation = expand_observation(row["observation"])
                features[index] = torch.tensor(encode_observation(observation), dtype=torch.float16)
                encoded_actions = [encode_action(deserialize_action(action)) for action in row["actions_raw"]]
            count = len(encoded_actions)
            if count > max_actions: raise ValueError("action count %d exceeds %d" % (count, max_actions))
            actions[index, :count] = torch.tensor(encoded_actions, dtype=torch.float16)
            masks[index, :count] = True
        output = os.path.join(output_dir, "%s-%s.pt" % (split, stem))
        torch.save({"features": features, "actions": actions, "masks": masks, "targets": targets}, output)
        result.append({"path": os.path.basename(output), "split": split, "samples": n})
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--input-dir", default="artifacts/official_bc"); p.add_argument("--output-dir", default="artifacts/official_bc_tensors"); p.add_argument("--workers", type=int, default=8); p.add_argument("--max-actions", type=int, default=64); a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.input_dir, "*.parquet")))
    entries = []
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=16) as pool:
        for index, value in enumerate(pool.imap_unordered(convert, ((path, a.output_dir, a.max_actions) for path in paths)), 1):
            entries.extend(value)
            if index % 20 == 0: print("converted=%d/%d tensor_shards=%d" % (index, len(paths), len(entries)), flush=True)
    metadata = {"format": "torch-tensor-cache", "max_actions": a.max_actions, "shards": sorted(entries, key=lambda x: x["path"])}
    with open(os.path.join(a.output_dir, "tensor_metadata.json"), "w") as f: json.dump(metadata, f, indent=2)
    print("train=%d val=%d" % (sum(x["samples"] for x in entries if x["split"] == "train"), sum(x["samples"] for x in entries if x["split"] == "val")))
if __name__ == "__main__": main()
