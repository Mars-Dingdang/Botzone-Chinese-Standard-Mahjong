import json
import glob
import os

from mahjong_agent.engine.actions import Action
from mahjong_agent.features import (deserialize_action, encode_action,
                                    encode_observation, expand_observation)


def record_to_json(record):
    legal = record["legal_actions"]
    action_key = record["action"].key()
    return {
        "features": encode_observation(record["observation"]),
        "actions": [encode_action(action) for action in legal],
        "target": next(i for i, action in enumerate(legal) if action.key() == action_key),
    }


def load_jsonl(path):
    records = []
    with open(path, "r") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def iter_jsonl_shards(path, pattern="*.jsonl", rank=0, world_size=1):
    paths = sorted(glob.glob(os.path.join(path, pattern))) if os.path.isdir(path) else [path]
    index = 0
    for shard in paths:
        with open(shard, "r") as handle:
            for line in handle:
                if index % world_size == rank:
                    yield json.loads(line)
                index += 1


def parquet_shard_plan(path, split, world_size):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    paths = sorted(glob.glob(os.path.join(path, "*.parquet")))
    weighted = []
    for shard in paths:
        column = pq.read_table(shard, columns=["split"])["split"]
        count = int(pc.sum(pc.cast(pc.equal(column, split), "int64")).as_py() or 0)
        weighted.append((count, shard))
    assignments = [[] for _ in range(world_size)]
    totals = [0] * world_size
    for count, shard in sorted(weighted, reverse=True):
        rank = min(range(world_size), key=lambda item: totals[item])
        assignments[rank].append(shard)
        totals[rank] += count
    return assignments, totals


def iter_parquet_shards(path, pattern="*.parquet", rank=0, world_size=1, split=None):
    import pyarrow.parquet as pq
    if split is not None and world_size > 1:
        paths = parquet_shard_plan(path, split, world_size)[0][rank]
    else:
        paths = sorted(glob.glob(os.path.join(path, pattern))) if os.path.isdir(path) else [path]
    for shard in paths:
        table = pq.read_table(shard)
        for record in table.to_pylist():
            if split is None or record.get("split") == split:
                record.pop("split", None)
                if "features" not in record:
                    record = {
                        "features": encode_observation(expand_observation(record["observation"])),
                        "actions": [encode_action(deserialize_action(action)) for action in record["actions_raw"]],
                        "target": record["target"],
                    }
                yield record


def iter_records(path, rank=0, world_size=1, split=None):
    if os.path.isdir(path) and glob.glob(os.path.join(path, "*.parquet")):
        return iter_parquet_shards(path, rank=rank, world_size=world_size, split=split)
    pattern = (split + "-*.jsonl") if split else "*.jsonl"
    return iter_jsonl_shards(path, pattern, rank, world_size)


def tensor_shard_plan(path, split, world_size, batch_size, seed=0, epoch=0):
    import random
    with open(os.path.join(path, "tensor_metadata.json")) as handle:
        entries = [item for item in json.load(handle)["shards"] if item["split"] == split]
    random.Random(seed + epoch).shuffle(entries)
    assignments = [[] for _ in range(world_size)]
    samples = [0] * world_size
    for item in sorted(entries, key=lambda x: x["samples"], reverse=True):
        rank = min(range(world_size), key=lambda value: samples[value])
        assignments[rank].append(item)
        samples[rank] += item["samples"]
    if world_size > 1:
        rotation = epoch % world_size
        assignments = [assignments[(rank + rotation) % world_size]
                       for rank in range(world_size)]
        samples = [samples[(rank + rotation) % world_size] for rank in range(world_size)]
    return assignments, [value // batch_size for value in samples]


def iter_tensor_batches(path, split, batch_size, rank=0, world_size=1, max_steps=0,
                        seed=0, epoch=0, shuffle=False):
    import torch
    assignments, _ = tensor_shard_plan(path, split, world_size, batch_size, seed, epoch)
    yielded = 0
    carry = None
    keys = None
    for item in assignments[rank]:
        payload = torch.load(os.path.join(path, item["path"]), map_location="cpu", weights_only=True)
        if keys is None:
            keys = tuple(key for key in (
                "features", "feature_masks", "actions", "action_token_masks",
                "masks", "targets", "aux_labels", "fan_targets", "belief_targets")
                if key in payload)
        current = tuple(payload[key] for key in keys)
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed + epoch * 1000003 + rank * 10007 + yielded)
            order = torch.randperm(current[0].size(0), generator=generator)
            current = tuple(value[order] for value in current)
        if carry is not None:
            current = tuple(torch.cat((left, right), dim=0) for left, right in zip(carry, current))
        full = current[0].size(0) // batch_size
        for index in range(full):
            start = index * batch_size
            yield tuple(value[start:start + batch_size] for value in current)
            yielded += 1
            if max_steps and yielded >= max_steps: return
        used = full * batch_size
        carry = tuple(value[used:] for value in current) if used < current[0].size(0) else None


def has_tensor_cache(path):
    return os.path.exists(os.path.join(path, "tensor_metadata.json"))


def collate_records(records, torch):
    if records and "feature_mask" in records[0]:
        return (
            torch.tensor([record["features"] for record in records], dtype=torch.float32),
            torch.tensor([record["feature_mask"] for record in records], dtype=torch.bool),
            torch.tensor([record["actions"] for record in records], dtype=torch.float32),
            torch.tensor([record["action_token_masks"] for record in records], dtype=torch.bool),
            torch.tensor([record["mask"] for record in records], dtype=torch.bool),
            torch.tensor([record["target"] for record in records], dtype=torch.long),
            torch.tensor([record.get("aux_labels", [0.0] * 4) for record in records], dtype=torch.float32),
            torch.tensor([record.get("fan_target", 0) for record in records], dtype=torch.long),
            torch.tensor([record.get("belief_targets", [[0] * 34] * 3) for record in records], dtype=torch.long),
        )
    max_actions = max(len(record["actions"]) for record in records)
    action_size = len(records[0]["actions"][0])
    features = []
    actions = []
    masks = []
    targets = []
    for record in records:
        count = len(record["actions"])
        features.append(record["features"])
        actions.append(record["actions"] + [[0.0] * action_size] * (max_actions - count))
        masks.append([1] * count + [0] * (max_actions - count))
        targets.append(record["target"])
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(masks, dtype=torch.bool),
        torch.tensor(targets, dtype=torch.long),
    )
