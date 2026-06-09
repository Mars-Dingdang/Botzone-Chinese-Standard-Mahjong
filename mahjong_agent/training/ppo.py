"""Compact masked-action PPO update used by the single-machine trainer."""


def ppo_update(model, optimizer, batch, clip_ratio=0.2, value_coef=0.5,
               entropy_coef=0.01, epochs=4):
    import torch
    features = batch["features"]
    actions = batch["actions"]
    masks = batch["masks"]
    chosen = batch["chosen"]
    old_log_probs = batch["old_log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    metrics = {}
    for _ in range(epochs):
        output = model(features, actions, masks)
        distribution = torch.distributions.Categorical(logits=output["logits"])
        log_probs = distribution.log_prob(chosen)
        ratio = (log_probs - old_log_probs).exp()
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = (output["value"] - returns).pow(2).mean()
        entropy = distribution.entropy().mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }
    return metrics
