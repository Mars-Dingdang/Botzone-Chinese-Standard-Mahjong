import json
import os
import subprocess
import time


def early_stopping_state(best, stale_epochs, value):
    # 返回更新后的最佳值、连续未提升轮数以及本轮是否提升。
    improved = value > best
    return (value if improved else best), (0 if improved else stale_epochs + 1), improved


def _commit():
    # 尝试记录当前 Git commit；非 Git 环境中使用 "unknown"。
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def save_checkpoint(path, model, optimizer=None, metadata=None, scheduler=None):
    # payload 保存可恢复训练的 state_dict；旁路 JSON 仅保存便于查看的 metadata。
    import torch
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "metadata": dict(metadata or {}),
    }
    payload["metadata"].update({
        "created_at": time.time(), "commit": _commit(),
        "feature_version": int(getattr(model, "feature_version", 1)),
        "architecture_version": int(getattr(model, "architecture_version", 1)),
        "model_class": model.__class__.__name__,
        "model_config": dict(getattr(model, "model_config", {})),
    })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path, _use_new_zipfile_serialization=False)
    with open(path + ".json", "w") as handle:
        json.dump(payload["metadata"], handle, indent=2, sort_keys=True)


def checkpoint_metadata(path, map_location="cpu"):
    # 只读取 checkpoint 内 metadata dict。
    import torch
    return torch.load(path, map_location=map_location).get("metadata", {})


def load_checkpoint(path, model, optimizer=None, map_location="cpu", scheduler=None,
                    allow_version_mismatch=False):
    # 默认严格校验特征和架构版本，避免形状兼容但语义不同的权重被误载。
    import torch
    payload = torch.load(path, map_location=map_location)
    metadata = payload.get("metadata", {})
    expected = int(getattr(model, "feature_version", 1))
    actual = int(metadata.get("feature_version", 1))
    if not allow_version_mismatch and expected != actual:
        raise ValueError("checkpoint feature_version=%d cannot load into model version=%d"
                         % (actual, expected))
    expected_architecture = int(getattr(model, "architecture_version", 1))
    actual_architecture = int(metadata.get("architecture_version", 1))
    if not allow_version_mismatch and expected_architecture != actual_architecture:
        raise ValueError(
            "checkpoint architecture_version=%d cannot load into model version=%d"
            % (actual_architecture, expected_architecture))
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    return metadata


def load_model_from_checkpoint(path, map_location="cpu"):
    # 先根据 metadata 构造正确模型，再加载权重；返回 (model, metadata)。
    from mahjong_agent.models import create_model
    metadata = checkpoint_metadata(path, map_location)
    model = create_model(metadata.get("feature_version", 1),
                         **metadata.get("model_config", {}))
    if hasattr(model, "belief_mode"):
        model.belief_mode = metadata.get("belief_mode", "aux")
    load_checkpoint(path, model, map_location=map_location)
    return model, metadata
