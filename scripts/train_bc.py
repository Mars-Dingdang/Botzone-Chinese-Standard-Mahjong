#!/usr/bin/env python3
"""Versioned BC trainer with weighted policy and auxiliary objectives."""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mahjong_agent.engine.actions import ActionType
from mahjong_agent.models import create_model
from mahjong_agent.training.checkpoint import (early_stopping_state,
                                               load_checkpoint, save_checkpoint)
from mahjong_agent.training.dataset import (has_tensor_cache,
                                            iter_tensor_batches,
                                            tensor_shard_plan)

DEFAULT_WEIGHTS = {
    "PASS": 1.0, "PLAY": 1.0, "CHI": 1.05, "PENG": 1.05,
    "GANG": 1.25, "BUGANG": 1.25, "HU": 1.5,
}


def _unpack(batch, device):
    if len(batch) != 9:
        raise ValueError("Feature V2 tensor cache required; rebuild with build_tensor_cache.py")
    batch = list(batch)
    state_tokens = int(batch[1].sum(1).max().item())
    max_actions = int(batch[4].sum(1).max().item())
    batch[0] = batch[0][:, :state_tokens]
    batch[1] = batch[1][:, :state_tokens]
    for index in (2, 3, 4):
        batch[index] = batch[index][:, :max_actions]
    values = [value.to(device) for value in batch]
    values[0] = values[0].float()
    values[2] = values[2].float()
    values[6] = values[6].float()
    return values


def _action_types(actions, targets):
    chosen = actions[torch.arange(len(targets), device=targets.device), targets, 0, 3]
    return torch.round(chosen - 1).long().clamp(0, len(ActionType) - 1)


def _weights(types, mode):
    base = torch.tensor([DEFAULT_WEIGHTS[kind.name] for kind in ActionType],
                        device=types.device)
    if mode == "none":
        return torch.ones_like(types, dtype=torch.float32)
    if mode == "inverse":
        counts = torch.bincount(types, minlength=len(ActionType)).float().clamp_min(1)
        inverse = (counts.sum() / counts).clamp(max=3.0)
        return inverse[types]
    return base[types]


def _loss(output, targets, types, action_mask, aux, fan_targets, belief_targets, args):
    log_probs = torch.nn.functional.log_softmax(output["logits"], dim=1)
    nll = -log_probs[torch.arange(len(targets), device=targets.device), targets]
    legal = action_mask.float()
    smooth = -(log_probs * legal).sum(1) / legal.sum(1).clamp_min(1.0)
    policy_each = (1.0 - args.label_smoothing) * nll + args.label_smoothing * smooth
    policy = (policy_each * _weights(types, args.weighting)).mean()
    binary = torch.nn.functional.binary_cross_entropy_with_logits(
        output["outcome"][:, [0, 1, 3]], aux[:, [0, 1, 3]])
    score = torch.nn.functional.mse_loss(output["outcome"][:, 2], aux[:, 2])
    fan = torch.nn.functional.cross_entropy(output["fan_logits"], fan_targets)
    belief = torch.zeros((), device=targets.device)
    belief_constraint = torch.zeros((), device=targets.device)
    if args.belief_mode != "none":
        belief = torch.nn.functional.cross_entropy(
            output["belief_logits"].reshape(-1, 5), belief_targets.reshape(-1))
        values = torch.arange(5, device=targets.device).float()
        expected = (output["belief_logits"].softmax(-1) * values).sum(-1)
        belief_constraint = torch.nn.functional.mse_loss(
            expected.sum(-1) / 14.0, belief_targets.float().sum(-1) / 14.0)
    total = policy + args.aux_coef * (binary + score + fan + belief + .01 * belief_constraint)
    return total, {"policy_loss": policy, "aux_binary_loss": binary,
                   "aux_score_loss": score, "aux_fan_loss": fan,
                   "belief_loss": belief, "belief_constraint_loss": belief_constraint}


def epoch(model, optimizer, data, device, scaler, train, args, desc):
    from tqdm import tqdm
    model.train(train)
    totals = Counter()
    type_total = Counter()
    type_correct = Counter()
    type_predicted = Counter()
    family_true_positive = Counter()
    iterator = tqdm(data, desc=desc, unit="batch", disable=not sys.stderr.isatty())
    context = torch.enable_grad if train else torch.no_grad
    for batch in iterator:
        features, feature_masks, actions, action_token_masks, masks, targets, aux, fan, belief = _unpack(batch, device)
        with context():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(features, actions, masks, feature_masks, action_token_masks)
                loss, parts = _loss(output, targets, _action_types(actions, targets),
                                    masks, aux, fan, belief, args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        predicted = output["logits"].argmax(1)
        matches = predicted == targets
        chosen_types = _action_types(actions, targets)
        predicted_types = _action_types(actions, predicted)
        top3 = output["logits"].topk(min(3, output["logits"].size(1)), dim=1).indices
        totals["samples"] += len(targets)
        totals["correct"] += int(matches.sum())
        totals["top3"] += int((top3 == targets.unsqueeze(1)).any(1).sum())
        totals["pass_vs_meld_correct"] += int(
            ((chosen_types == int(ActionType.PASS)) ==
             (predicted_types == int(ActionType.PASS))).sum())
        probabilities = output["outcome"][:, [0, 1, 3]].sigmoid()
        totals["aux_brier"] += float((probabilities - aux[:, [0, 1, 3]]).pow(2).sum())
        totals["nll"] += float(torch.nn.functional.cross_entropy(
            output["logits"], targets, reduction="sum"))
        totals["loss"] += float(loss) * len(targets)
        for name, value in parts.items():
            totals[name] += float(value) * len(targets)
        for kind in ActionType:
            selected = chosen_types == int(kind)
            predicted_selected = predicted_types == int(kind)
            type_total[kind.name] += int(selected.sum())
            type_correct[kind.name] += int((matches & selected).sum())
            type_predicted[kind.name] += int(predicted_selected.sum())
            family_true_positive[kind.name] += int((selected & predicted_selected).sum())
    samples = max(1, totals["samples"])
    result = {
        "loss": totals["loss"] / samples, "nll": totals["nll"] / samples,
        "accuracy": totals["correct"] / samples, "top3_accuracy": totals["top3"] / samples,
        "samples": totals["samples"],
    }
    result["accuracy_by_action"] = {
        name: type_correct[name] / float(max(1, type_total[name])) for name in type_total}
    result["precision_by_action"] = {
        name: family_true_positive[name] / float(max(1, type_predicted[name])) for name in type_total}
    result["recall_by_action"] = {
        name: family_true_positive[name] / float(max(1, type_total[name])) for name in type_total}
    result["macro_accuracy"] = sum(result["accuracy_by_action"].values()) / len(ActionType)
    result["pass_vs_meld_accuracy"] = totals["pass_vs_meld_correct"] / samples
    result["aux_brier"] = totals["aux_brier"] / (samples * 3.0)
    result["samples_by_action"] = dict(type_total)
    for name in ("policy_loss", "aux_binary_loss", "aux_score_loss",
                 "aux_fan_loss", "belief_loss", "belief_constraint_loss"):
        result[name] = totals[name] / samples
    return result


def log_jsonl(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/official_bc_full_v3_tensors")
    parser.add_argument("--output", default="artifacts/bc_model.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr-patience", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--weighting", choices=("fixed", "inverse", "none"), default="fixed")
    parser.add_argument("--aux-coef", type=float, default=0.1)
    parser.add_argument("--belief-mode", choices=("none", "aux", "actor"), default="aux")
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if not has_tensor_cache(args.data):
        raise ValueError("tensor cache not found: %s" % args.data)
    with open(os.path.join(args.data, "tensor_metadata.json")) as handle:
        cache_meta = json.load(handle)
    if int(cache_meta.get("feature_version", 1)) != 2:
        raise ValueError("Feature V2 tensor cache required")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if rank == 0:
        print(json.dumps({
            "event": "training_config", "batch_size_per_rank": args.batch_size,
            "effective_batch_size": args.batch_size * world,
            "world_size": world, "data": args.data, "belief_mode": args.belief_mode,
        }, sort_keys=True), flush=True)
    if distributed:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local)
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    model = create_model(2).to(device)
    model.belief_mode = args.belief_mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience)
    start = 0
    best = -1.0
    stale = 0
    if args.resume:
        metadata = load_checkpoint(args.resume, model, optimizer, scheduler=scheduler)
        start = int(metadata.get("epoch", 0))
        best = float(metadata.get("best_metric", -1))
        stale = int(metadata.get("stale_epochs", 0))
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=True)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch_index in range(start, args.epochs):
        _, rank_steps = tensor_shard_plan(args.data, "train", world, args.batch_size)
        steps = min(rank_steps)
        if args.max_steps:
            steps = min(steps, args.max_steps)
        train_data = iter_tensor_batches(args.data, "train", args.batch_size, rank, world, steps)
        train_metrics = epoch(model, optimizer, train_data, device, scaler, True, args,
                              "bc-train-%d" % (epoch_index + 1))
        if distributed:
            torch.distributed.barrier()
        stop = False
        if rank == 0:
            validation_model = model.module if distributed else model
            validation_data = iter_tensor_batches(
                args.data, "val", args.batch_size, max_steps=args.max_steps)
            val_metrics = epoch(validation_model, optimizer, validation_data, device, scaler,
                                False, args, "bc-val-%d" % (epoch_index + 1))
            selection_metric = (val_metrics["accuracy"] + val_metrics["macro_accuracy"]) / 2.0
            scheduler.step(selection_metric)
            best, stale, improved = early_stopping_state(best, stale, selection_metric)
            metadata = {
                "algorithm": "bc", "epoch": epoch_index + 1, "train": train_metrics,
                "val": val_metrics, "best_metric": best, "stale_epochs": stale,
                "label_smoothing": args.label_smoothing, "weighting": args.weighting,
                "aux_coef": args.aux_coef,
                "belief_mode": args.belief_mode,
                "batch_size_per_rank": args.batch_size,
                "effective_batch_size": args.batch_size * world,
                "seed": args.seed,
            }
            saved = model.module if distributed else model
            save_checkpoint(args.output, saved, optimizer, metadata, scheduler)
            save_checkpoint(args.output.replace(
                ".pt", ".epoch-%04d.pt" % (epoch_index + 1)),
                saved, optimizer, metadata, scheduler)
            if improved:
                save_checkpoint(args.output.replace(".pt", ".best.pt"), saved, optimizer,
                                metadata, scheduler)
            log_jsonl(args.metrics_jsonl, metadata)
            print(json.dumps(metadata, sort_keys=True), flush=True)
            stop = stale >= args.patience
        if distributed:
            control = torch.tensor([int(stop), optimizer.param_groups[0]["lr"]], device=device)
            torch.distributed.broadcast(control, 0)
            stop = bool(control[0].item())
            for group in optimizer.param_groups:
                group["lr"] = float(control[1].item())
            torch.distributed.barrier()
        if stop:
            break
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
