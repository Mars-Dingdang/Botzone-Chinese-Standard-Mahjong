#!/usr/bin/env python3
"""Action-conditioned, distributed BC trainer."""
import argparse
import json
import math
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


def _yaml(path):
    if not path:
        return {}
    import yaml
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def _parser():
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--config", default="configs/train/bc.yaml")
    early.add_argument("--model-config", default="configs/model/base.yaml")
    known, _ = early.parse_known_args()
    cfg = _yaml(known.config)
    model_cfg = _yaml(known.model_config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--model-config", default=known.model_config)
    parser.add_argument("--data", default="artifacts/official_bc_v4_tensors")
    parser.add_argument("--output", default="artifacts/bc_model.pt")
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 50))
    parser.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 512))
    parser.add_argument("--lr", type=float, default=cfg.get("learning_rate", 3e-4))
    parser.add_argument("--min-lr", type=float, default=cfg.get("min_learning_rate", 3e-5))
    parser.add_argument("--weight-decay", type=float, default=cfg.get("weight_decay", 0.01))
    parser.add_argument("--warmup-fraction", type=float, default=cfg.get("warmup_fraction", 0.05))
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--patience", type=int, default=cfg.get("patience", 8))
    parser.add_argument("--min-epochs", type=int, default=cfg.get("min_epochs", 20))
    parser.add_argument("--label-smoothing", type=float, default=cfg.get("label_smoothing", 0.01))
    parser.add_argument("--family-loss-coef", type=float, default=cfg.get("family_loss_coef", 0.2))
    parser.add_argument("--outcome-aux-coef", type=float, default=cfg.get("outcome_aux_coef", 0.05))
    parser.add_argument("--fan-aux-coef", type=float, default=cfg.get("fan_aux_coef", 0.05))
    parser.add_argument("--belief-aux-coef", type=float, default=cfg.get("belief_aux_coef", 0.02))
    parser.add_argument("--value-loss-coef", type=float, default=cfg.get("value_loss_coef", 0.1))
    parser.add_argument("--eight-fan-policy-weight", type=float,
                        default=cfg.get("eight_fan_policy_weight", 3.0))
    parser.add_argument("--win-pos-weight", type=float, default=cfg.get("win_pos_weight", 3.0))
    parser.add_argument("--deal-in-pos-weight", type=float,
                        default=cfg.get("deal_in_pos_weight", 3.0))
    parser.add_argument("--eight-fan-pos-weight", type=float,
                        default=cfg.get("eight_fan_pos_weight", 5.0))
    parser.add_argument("--belief-mode", choices=("none", "aux", "actor"),
                        default=cfg.get("belief_mode", "aux"))
    parser.add_argument("--disable-auxiliary-logit-fusion", action="store_true")
    parser.add_argument("--d-model", type=int, default=model_cfg.get("d_model", 256))
    parser.add_argument("--layers", type=int, default=model_cfg.get("layers", 6))
    parser.add_argument("--heads", type=int, default=model_cfg.get("heads", 8))
    parser.add_argument("--dropout", type=float, default=model_cfg.get("dropout", 0.1))
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _unpack(batch, device):
    if len(batch) != 9:
        raise ValueError("new Feature V2 tensor cache required")
    batch = list(batch)
    state_tokens = int(batch[1].sum(1).max().item())
    max_actions = int(batch[4].sum(1).max().item())
    batch[0], batch[1] = batch[0][:, :state_tokens], batch[1][:, :state_tokens]
    for index in (2, 3, 4):
        batch[index] = batch[index][:, :max_actions]
    values = [value.to(device) for value in batch]
    values[0], values[2], values[6] = values[0].float(), values[2].float(), values[6].float()
    return values


def _action_types(actions, targets):
    chosen = actions[torch.arange(len(targets), device=targets.device), targets, 0, 3]
    return torch.round(chosen - 1).long().clamp(0, len(ActionType) - 1)


def _loss(output, targets, types, action_mask, aux, fan_targets, belief_targets, args):
    rows = torch.arange(len(targets), device=targets.device)
    log_probs = torch.nn.functional.log_softmax(output["logits"], dim=1)
    nll = -log_probs[rows, targets]
    smooth = -(log_probs * action_mask.float()).sum(1) / action_mask.sum(1).clamp_min(1)
    sample_loss = (1.0 - args.label_smoothing) * nll + args.label_smoothing * smooth
    sample_weight = 1.0 + aux[:, 3] * (args.eight_fan_policy_weight - 1.0)
    policy = (sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
    family = torch.nn.functional.cross_entropy(output["family_scores"], types)
    chosen_outcome = output["action_outcome"][rows, targets]
    binary = torch.nn.functional.binary_cross_entropy_with_logits(
        chosen_outcome[:, [0, 1, 3]], aux[:, [0, 1, 3]],
        pos_weight=torch.tensor(
            [args.win_pos_weight, args.deal_in_pos_weight, args.eight_fan_pos_weight],
            device=targets.device))
    score = torch.nn.functional.mse_loss(chosen_outcome[:, 2], aux[:, 2])
    value = torch.nn.functional.mse_loss(output["value"], aux[:, 2])
    fan = torch.nn.functional.cross_entropy(output["action_fan_logits"][rows, targets], fan_targets)
    belief = torch.zeros((), device=targets.device)
    if args.belief_mode != "none":
        belief = torch.nn.functional.cross_entropy(
            output["belief_logits"].reshape(-1, 5), belief_targets.reshape(-1))
    total = (policy + args.family_loss_coef * family +
             args.outcome_aux_coef * (binary + score) +
             args.fan_aux_coef * fan + args.belief_aux_coef * belief +
             args.value_loss_coef * value)
    return total, {"policy_loss": policy, "family_loss": family,
                   "aux_binary_loss": binary, "aux_score_loss": score,
                   "aux_fan_loss": fan, "belief_loss": belief, "value_loss": value}


def _reduce(counter, device):
    if not torch.distributed.is_initialized():
        return counter
    keys = sorted(counter)
    values = torch.tensor([float(counter[key]) for key in keys], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(values)
    return Counter(dict(zip(keys, values.cpu().tolist())))


def epoch(model, optimizer, scheduler, data, device, scaler, train, args, desc,
          aggregate=True):
    from tqdm import tqdm
    model.train(train)
    totals, family_total, family_correct = Counter(), Counter(), Counter()
    iterator = tqdm(data, desc=desc, unit="batch", disable=not sys.stderr.isatty())
    context = torch.enable_grad if train else torch.no_grad
    for batch in iterator:
        features, feature_masks, actions, action_token_masks, masks, targets, aux, fan, belief = _unpack(batch, device)
        with context():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(features, actions, masks, feature_masks, action_token_masks)
                chosen_types = _action_types(actions, targets)
                loss, parts = _loss(output, targets, chosen_types, masks, aux, fan, belief, args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        predicted = output["logits"].argmax(1)
        predicted_types = _action_types(actions, predicted)
        matches, family_matches = predicted == targets, predicted_types == chosen_types
        phases = torch.round(features[:, 0, 3] - 1).long()
        claim = phases == 3
        samples = len(targets)
        totals.update(samples=samples, exact_correct=int(matches.sum()),
                      family_correct=int(family_matches.sum()),
                      tactical_correct=int((matches & family_matches).sum()),
                      tactical_total=int(family_matches.sum()),
                      play_correct=int((matches & (chosen_types == int(ActionType.PLAY))).sum()),
                      play_total=int((chosen_types == int(ActionType.PLAY)).sum()),
                      claim_correct=int((matches & claim).sum()),
                      claim_total=int(claim.sum()),
                      eight_fan_correct=int((matches & (aux[:, 3] > .5)).sum()),
                      eight_fan_total=int((aux[:, 3] > .5).sum()))
        totals["nll"] += float(torch.nn.functional.cross_entropy(
            output["logits"], targets, reduction="sum"))
        totals["loss"] += float(loss) * samples
        for name, value in parts.items():
            totals[name] += float(value) * samples
        for kind in ActionType:
            selected = chosen_types == int(kind)
            family_total[kind.name] += int(selected.sum())
            family_correct[kind.name] += int((selected & family_matches).sum())
    if aggregate:
        totals, family_total, family_correct = (
            _reduce(totals, device), _reduce(family_total, device), _reduce(family_correct, device))
    samples = max(1, totals["samples"])
    recalls = {kind.name: family_correct[kind.name] / max(1, family_total[kind.name])
               for kind in ActionType}
    result = {
        "loss": totals["loss"] / samples, "nll": totals["nll"] / samples,
        "exact_accuracy": totals["exact_correct"] / samples,
        "accuracy": totals["exact_correct"] / samples,
        "family_accuracy": totals["family_correct"] / samples,
        "tactical_accuracy_given_family": totals["tactical_correct"] / max(1, totals["tactical_total"]),
        "play_accuracy": totals["play_correct"] / max(1, totals["play_total"]),
        "claim_exact_accuracy": totals["claim_correct"] / max(1, totals["claim_total"]),
        "eight_fan_exact_accuracy": totals["eight_fan_correct"] / max(1, totals["eight_fan_total"]),
        "macro_family_recall": sum(recalls.values()) / len(ActionType),
        "family_recall": recalls, "samples": int(totals["samples"]),
        "samples_by_action": {key: int(value) for key, value in family_total.items()},
    }
    for name in ("policy_loss", "family_loss", "aux_binary_loss", "aux_score_loss",
                 "aux_fan_loss", "belief_loss", "value_loss"):
        result[name] = totals[name] / samples
    return result


def log_jsonl(path, payload):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    args = _parser().parse_args()
    torch.manual_seed(args.seed)
    if not has_tensor_cache(args.data):
        raise ValueError("tensor cache not found: %s" % args.data)
    with open(os.path.join(args.data, "tensor_metadata.json")) as handle:
        cache_meta = json.load(handle)
    if int(cache_meta.get("cache_version", 0)) < 2 or int(cache_meta.get("label_version", 0)) < 2:
        raise ValueError("obsolete tensor cache; rebuild with scripts/build_tensor_cache.py")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, local, world = (int(os.environ.get(name, default)) for name, default in
                          (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    if distributed:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local)
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    model_kwargs = {"d_model": args.d_model, "layers": args.layers, "heads": args.heads,
                    "dropout": args.dropout, "belief_mode": args.belief_mode,
                    "auxiliary_logit_fusion": not args.disable_auxiliary_logit_fusion}
    model = create_model(2, **model_kwargs).to(device)
    if args.init_checkpoint:
        if args.resume:
            raise ValueError("--init-checkpoint and --resume are mutually exclusive")
        init_meta = load_checkpoint(args.init_checkpoint, model)
        if args.belief_mode == "actor" and init_meta.get("belief_mode", "aux") != "actor":
            torch.nn.init.zeros_(model.belief_adapter.weight)
            torch.nn.init.zeros_(model.belief_adapter.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    _, rank_steps = tensor_shard_plan(args.data, "train", world, args.batch_size, args.seed, 0)
    steps_per_epoch = min(rank_steps)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_fraction))
    min_ratio = args.min_lr / args.lr

    def schedule(step):
        if step < warmup_steps:
            return max(1e-4, float(step + 1) / warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start, best, stale = 0, -1.0, 0
    if args.resume:
        metadata = load_checkpoint(args.resume, model, optimizer, scheduler=scheduler)
        start, best, stale = int(metadata.get("epoch", 0)), float(metadata.get("best_metric", -1)), int(metadata.get("stale_epochs", 0))
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=True)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    if rank == 0:
        print(json.dumps({"event": "training_config", "model": model_kwargs,
                          "effective_batch_size": args.batch_size * world,
                          "cache_version": cache_meta["cache_version"]}, sort_keys=True), flush=True)
    for epoch_index in range(start, args.epochs):
        _, rank_steps = tensor_shard_plan(
            args.data, "train", world, args.batch_size, args.seed, epoch_index)
        steps = min(rank_steps)
        if args.max_steps:
            steps = min(steps, args.max_steps)
        train_data = iter_tensor_batches(
            args.data, "train", args.batch_size, rank, world, steps,
            seed=args.seed, epoch=epoch_index, shuffle=True)
        train_metrics = epoch(model, optimizer, scheduler, train_data, device, scaler, True, args,
                              "bc-train-%d" % (epoch_index + 1))
        stop = False
        if rank == 0:
            validation_model = model.module if distributed else model
            validation_data = iter_tensor_batches(
                args.data, "val", args.batch_size, max_steps=args.max_steps,
                seed=args.seed, epoch=0, shuffle=False)
            val_metrics = epoch(validation_model, optimizer, scheduler, validation_data, device,
                                scaler, False, args, "bc-val-%d" % (epoch_index + 1),
                                aggregate=False)
            selection_metric = val_metrics["exact_accuracy"]
            best, stale, improved = early_stopping_state(best, stale, selection_metric)
            metadata = {"algorithm": "bc", "epoch": epoch_index + 1, "train": train_metrics,
                        "val": val_metrics, "best_metric": best, "stale_epochs": stale,
                        "model_config": model_kwargs, "training_config": vars(args),
                        "batch_size_per_rank": args.batch_size,
                        "effective_batch_size": args.batch_size * world,
                        "learning_rate": optimizer.param_groups[0]["lr"], "seed": args.seed}
            gate_owner = model.module if distributed else model
            metadata["auxiliary_logit_gate"] = gate_owner.auxiliary_logit_gate.detach().cpu().tolist()
            saved = model.module if distributed else model
            save_checkpoint(args.output, saved, optimizer, metadata, scheduler)
            epoch_path = args.output.replace(".pt", ".epoch-%04d.pt" % (epoch_index + 1))
            save_checkpoint(epoch_path, saved, optimizer, metadata, scheduler)
            if improved:
                save_checkpoint(args.output.replace(".pt", ".best.pt"), saved, optimizer,
                                metadata, scheduler)
            log_jsonl(args.metrics_jsonl, metadata)
            print(json.dumps(metadata, sort_keys=True), flush=True)
            stop = epoch_index + 1 >= args.min_epochs and stale >= args.patience
        if distributed:
            control = torch.tensor([int(stop)], device=device)
            torch.distributed.broadcast(control, 0)
            stop = bool(control.item())
        if stop:
            break
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
