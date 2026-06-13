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
               minibatch_size=1024, aux_coef=0.05, deal_in_pos_weight=3.0,
               direct_deal_in_pos_weight=8.0):
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
            kwargs = {}
            if batch.get("feature_masks") is not None:
                kwargs["feature_mask"] = batch["feature_masks"][index]
                kwargs["action_token_mask"] = batch["action_token_masks"][index]
            output = model(features, actions, masks, **kwargs)
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
            aux_loss = torch.zeros((), device=features.device)
            if batch.get("aux_labels") is not None and "outcome" in output:
                labels = batch["aux_labels"][index]
                binary = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["outcome"][:, [0, 1, 3]], labels[:, [0, 1, 3]],
                    pos_weight=torch.tensor(
                        [1.0, deal_in_pos_weight, 1.0], device=features.device))
                score = torch.nn.functional.mse_loss(output["outcome"][:, 2], labels[:, 2])
                fan = torch.nn.functional.cross_entropy(
                    output["fan_logits"], batch["fan_targets"][index])
                belief = torch.nn.functional.cross_entropy(
                    output["belief_logits"].reshape(-1, 5),
                    batch["belief_targets"][index].reshape(-1))
                values = torch.arange(5, device=features.device).float()
                expected = (output["belief_logits"].softmax(-1) * values).sum(-1)
                constraint = torch.nn.functional.mse_loss(
                    expected.sum(-1) / 14.0,
                    batch["belief_targets"][index].float().sum(-1) / 14.0)
                chosen_outcome = output["action_outcome"][
                    torch.arange(len(index), device=features.device), chosen]
                action_labels = batch.get("action_aux_labels", batch["aux_labels"])[index]
                action_binary = torch.nn.functional.binary_cross_entropy_with_logits(
                    chosen_outcome[:, [0, 1, 3]], action_labels[:, [0, 1, 3]],
                    pos_weight=torch.tensor(
                        [1.0, direct_deal_in_pos_weight, 1.0], device=features.device))
                action_score = torch.nn.functional.mse_loss(
                    chosen_outcome[:, 2], action_labels[:, 2])
                action_fan = torch.nn.functional.cross_entropy(
                    output["action_fan_logits"][
                        torch.arange(len(index), device=features.device), chosen],
                    batch["fan_targets"][index])
                aux_loss = (binary + score + fan + belief + .01 * constraint +
                            action_binary + action_score + action_fan)
            loss = (policy_loss + value_coef * value_loss - entropy_coef * entropy
                    + bc_kl_coef * bc_kl + aux_coef * aux_loss)
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
                "aux_loss": aux_loss,
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
