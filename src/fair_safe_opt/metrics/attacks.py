import torch

def attack_success_rate(
    true_labels: torch.Tensor, 
    baseline_preds: torch.Tensor, 
    attacked_preds: torch.Tensor
) -> float:
    '''Calculates the rate at which the adversarial attack switched a correct prediction to an incorrect one.'''
    correct_baseline_preds = (true_labels == baseline_preds).float()
    successful_attacks = (correct_baseline_preds != attacked_preds).float()
    
    return successful_attacks.mean()