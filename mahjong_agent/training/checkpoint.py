import json
import os
import subprocess
import time


def early_stopping_state(best, stale_epochs, value):
    improved = value > best
    return (value if improved else best), (0 if improved else stale_epochs + 1), improved


def _commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def save_checkpoint(path, model, optimizer=None, metadata=None, scheduler=None):
    import torch
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "metadata": dict(metadata or {}),
    }
    payload["metadata"].update({"created_at": time.time(), "commit": _commit()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path, _use_new_zipfile_serialization=False)
    with open(path + ".json", "w") as handle:
        json.dump(payload["metadata"], handle, indent=2, sort_keys=True)


def load_checkpoint(path, model, optimizer=None, map_location="cpu", scheduler=None):
    import torch
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload.get("metadata", {})
