"""Masked-action PPO utilities used by the single-machine trainer."""


def potential_shaped_rewards(potentials, terminal_reward, gamma=0.99, coefficient=0.02):
    """Apply potential shaping, including the terminal transition to Phi=0."""
    rewards = [0.0] * len(potentials)
    if not rewards:
        return rewards
    rewards[-1] = terminal_reward
    for index, current in enumerate(potentials):
        following = potentials[index + 1] if index + 1 < len(potentials) else 0.0
        rewards[index] += coefficient * (gamma * following - current)
    return rewards


def generalized_advantage_estimate(rewards, values, gamma=0.99, gae_lambda=0.95):
    """Return GAE advantages and value targets for one decision trajectory."""
    advantages = [0.0] * len(rewards)
    last = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        last = delta + gamma * gae_lambda * last
        advantages[index] = last
    return advantages, [advantage + value for advantage, value in zip(advantages, values)]


def ppo_update(model, optimizer, batch, clip_ratio=0.2, value_coef=0.5,
               entropy_coef=0.01, epochs=4, target_kl=0.02, bc_kl_coef=0.01,
               minibatch_size=1024):
    import torch
    model.train()
    # PPO ratios require deterministic old/new log probabilities. Keep gradients
    # enabled while disabling dropout that would otherwise create artificial KL.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    sample_count = batch["features"].size(0)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        shared_count = torch.tensor([sample_count], device=batch["features"].device)
        torch.distributed.all_reduce(shared_count, op=torch.distributed.ReduceOp.MIN)
        sample_count = int(shared_count.item())
        for key, value in list(batch.items()):
            if hasattr(value, "size") and value.size(0) >= sample_count:
                batch[key] = value[:sample_count]
    minibatch_size = min(max(1, minibatch_size), sample_count)
    totals = {}
    updates = 0
    completed_epochs = 0
    for _ in range(epochs):
        order = torch.randperm(sample_count, device=batch["features"].device)
        epoch_kl = 0.0
        epoch_updates = 0
        for start in range(0, sample_count, minibatch_size):
            index = order[start:start + minibatch_size]
            features = batch["features"][index]
            actions = batch["actions"][index]
            masks = batch["masks"][index]
            chosen = batch["chosen"][index]
            old_log_probs = batch["old_log_probs"][index]
            advantages = batch["advantages"][index]
            returns = batch["returns"][index]
            output = model(features, actions, masks)
            distribution = torch.distributions.Categorical(logits=output["logits"])
            log_probs = distribution.log_prob(chosen)
            ratio = (log_probs - old_log_probs).exp()
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = (output["value"] - returns).pow(2).mean()
            entropy = distribution.entropy().mean()
            approx_kl = (old_log_probs - log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
            bc_kl = torch.zeros((), device=features.device)
            if batch.get("reference_logits") is not None and bc_kl_coef:
                reference = torch.distributions.Categorical(logits=batch["reference_logits"][index])
                bc_kl = torch.distributions.kl_divergence(reference, distribution).mean()
            loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy
                    + bc_kl_coef * bc_kl + 0.0 * output["aux"].sum())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            returns_var = returns.var(unbiased=False)
            explained = 1.0 - (returns - output["value"]).var(unbiased=False) / (returns_var + 1e-8)
            values = {
                "loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
                "entropy": entropy, "approx_kl": approx_kl, "clip_fraction": clip_fraction,
                "explained_variance": explained, "bc_kl": bc_kl,
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
            updates += 1
            epoch_updates += 1
            epoch_kl += float(approx_kl.item())
        completed_epochs += 1
        should_stop = target_kl > 0 and epoch_kl / max(1, epoch_updates) > target_kl
        if distributed:
            stop_tensor = torch.tensor([int(should_stop)], device=batch["features"].device)
            torch.distributed.all_reduce(stop_tensor, op=torch.distributed.ReduceOp.MAX)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break
    metrics = dict((key, value / max(1, updates)) for key, value in totals.items())
    metrics.update({"epochs": completed_epochs, "minibatches": updates, "samples": sample_count})
    return metrics
