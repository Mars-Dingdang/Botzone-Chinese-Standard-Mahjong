#!/usr/bin/env python3
"""Build fixed-shape PyTorch tensor shards from archival Parquet BC data."""
import argparse, glob, json, multiprocessing as mp, os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mahjong_agent.features import (deserialize_action, encode_action_v2,
                                    encode_observation_v2, expand_observation)
from mahjong_agent.features.token_encoder import (MAX_ACTION_TOKENS,
                                                  MAX_STATE_TOKENS, TOKEN_SIZE)

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
        features = torch.zeros(n, MAX_STATE_TOKENS, TOKEN_SIZE, dtype=torch.float16)
        feature_masks = torch.zeros(n, MAX_STATE_TOKENS, dtype=torch.bool)
        actions = torch.zeros(n, max_actions, MAX_ACTION_TOKENS, TOKEN_SIZE, dtype=torch.float16)
        action_token_masks = torch.zeros(n, max_actions, MAX_ACTION_TOKENS, dtype=torch.bool)
        masks = torch.zeros(n, max_actions, dtype=torch.bool)
        targets = torch.tensor([row["target"] for row in selected], dtype=torch.long)
        aux = torch.zeros(n, 4, dtype=torch.float32)
        fan_targets = torch.zeros(n, dtype=torch.long)
        belief = torch.zeros(n, 3, 34, dtype=torch.long)
        for index, row in enumerate(selected):
            observation = expand_observation(row["observation"])
            encoded_feature, encoded_feature_mask = encode_observation_v2(observation)
            features[index] = torch.tensor(encoded_feature, dtype=torch.float16)
            feature_masks[index] = torch.tensor(encoded_feature_mask, dtype=torch.bool)
            encoded_actions = [encode_action_v2(deserialize_action(action)) for action in row["actions_raw"]]
            count = len(encoded_actions)
            if count > max_actions: raise ValueError("action count %d exceeds %d" % (count, max_actions))
            actions[index, :count] = torch.tensor([item[0] for item in encoded_actions], dtype=torch.float16)
            action_token_masks[index, :count] = torch.tensor([item[1] for item in encoded_actions], dtype=torch.bool)
            masks[index, :count] = True
            labels = row.get("aux_labels") or {}
            aux[index] = torch.tensor([labels.get("win", 0), labels.get("deal_in", 0),
                                       labels.get("score", 0), labels.get("eight_fan", 0)])
            fan_targets[index] = int(labels.get("fan_bucket", 0))
            belief[index] = torch.tensor(row.get("belief_counts", [[0] * 34 for _ in range(3)]),
                                         dtype=torch.long)
        output = os.path.join(output_dir, "%s-%s.pt" % (split, stem))
        torch.save({"features": features, "feature_masks": feature_masks,
                    "actions": actions, "action_token_masks": action_token_masks,
                    "masks": masks, "targets": targets, "aux_labels": aux,
                    "fan_targets": fan_targets, "belief_targets": belief}, output)
        result.append({"path": os.path.basename(output), "split": split, "samples": n})
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--input-dir", default="artifacts/official_bc_v4"); p.add_argument("--output-dir", default="artifacts/official_bc_v4_tensors"); p.add_argument("--workers", type=int, default=8); p.add_argument("--max-actions", type=int, default=64); a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.input_dir, "*.parquet")))
    entries = []
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=16) as pool:
        from tqdm import tqdm
        work = ((path, a.output_dir, a.max_actions) for path in paths)
        for index, value in enumerate(tqdm(pool.imap_unordered(convert, work),
                                           total=len(paths), desc="tensor-cache",
                                           unit="shard"), 1):
            entries.extend(value)
            if index % 20 == 0: print("converted=%d/%d tensor_shards=%d" % (index, len(paths), len(entries)), flush=True)
    source_metadata = {}
    source_path = os.path.join(a.input_dir, "metadata.json")
    if os.path.exists(source_path):
        with open(source_path) as handle:
            source_metadata = json.load(handle)
    family_counts = source_metadata.get("families", {})
    metadata = {
        "format": "torch-tensor-cache", "cache_version": 2, "feature_version": 2,
        "legality_version": 5, "label_version": 2, "max_actions": a.max_actions,
        "samples": sum(item["samples"] for item in entries),
        "samples_by_split": {
            split: sum(item["samples"] for item in entries if item["split"] == split)
            for split in ("train", "val")
        },
        "families": family_counts,
        "source_version": source_metadata.get("version"),
        "shards": sorted(entries, key=lambda x: x["path"]),
    }
    with open(os.path.join(a.output_dir, "tensor_metadata.json"), "w") as f: json.dump(metadata, f, indent=2)
    print("train=%d val=%d" % (sum(x["samples"] for x in entries if x["split"] == "train"), sum(x["samples"] for x in entries if x["split"] == "val")))
if __name__ == "__main__": main()
