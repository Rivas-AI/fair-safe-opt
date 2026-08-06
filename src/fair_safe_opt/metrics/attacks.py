import torch


def attack_success_rate(true_labels, baseline_preds, attacked_preds) -> float:
    """Rate at which the attack switched a correct prediction to an incorrect one."""
    baseline_correct = true_labels == baseline_preds
    amount_correct = baseline_correct.sum()
    attacked_wrong = true_labels != attacked_preds
    successful_attacks = (baseline_correct & attacked_wrong).float()
    rate = successful_attacks.sum() / amount_correct
    return rate.item()


def attack_success_by_group(
    true_labels: torch.Tensor,
    baseline_preds: torch.Tensor,
    attacked_preds: torch.Tensor,
    sensitive: torch.Tensor,
) -> list[float]:
    combined_rate = attack_success_rate(true_labels, baseline_preds, attacked_preds)

    mask = sensitive == 0

    gp0_rate = attack_success_rate(true_labels[mask], baseline_preds[mask], attacked_preds[mask])
    gp1_rate = attack_success_rate(true_labels[~mask], baseline_preds[~mask], attacked_preds[~mask])

    return [gp0_rate, gp1_rate, combined_rate]
