"""Masked-action PPO utilities used by the single-machine trainer."""


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
               entropy_coef=0.01, epochs=4, target_kl=0.02):
    import torch
    features = batch["features"]
    actions = batch["actions"]
    masks = batch["masks"]
    chosen = batch["chosen"]
    old_log_probs = batch["old_log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    metrics = {}
    completed_epochs = 0
    for _ in range(epochs):
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
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        loss = loss + 0.0 * output["aux"].sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_epochs += 1
        returns_var = returns.var(unbiased=False)
        explained_variance = 1.0 - (returns - output["value"]).var(unbiased=False) / (returns_var + 1e-8)
        metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "explained_variance": float(explained_variance.item()),
            "epochs": completed_epochs,
        }
        if target_kl > 0 and approx_kl.item() > target_kl:
            break
    return metrics
