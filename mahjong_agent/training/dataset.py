import json

from mahjong_agent.engine.actions import Action
from mahjong_agent.features import encode_action, encode_observation


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


def collate_records(records, torch):
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
