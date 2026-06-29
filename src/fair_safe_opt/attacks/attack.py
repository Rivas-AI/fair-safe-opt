import torch

from fair_safe_opt.attacks.fgsm import fgsm_attack
from fair_safe_opt.attacks.pgd import pgd_attack

def attack_features(
    model: torch.nn.Module,
    attack_type: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    feature_mins: torch.Tensor,
    feature_maxes: torch.Tensor,
    eps: float, 
    steps: int | None = None,
    alpha: float | None = None,
    random_start: bool = False
) -> torch.Tensor: 
    '''Generates an adversarial attack of a given method at a given model.'''
    if attack_type == 'none':
        return features
    elif attack_type == 'pgd' or attack_type == 'fgsm':
        print(f"Generating adversarial features. attack_type: {attack_type}")
        
        if eps is None or eps < 0:
            raise ValueError(f"Incorrect 'eps' parameter value: {eps}")
        
        if attack_type == 'fgsm':
            features = fgsm_attack(model, features, labels, eps, mask, feature_mins, feature_maxes)
        elif attack_type == 'pgd':
            if alpha is None or alpha < 0:
                raise ValueError(f"Invalid 'alpha' param value {alpha}")
            features = pgd_attack(model, features, labels, eps, alpha, steps, mask, feature_mins, feature_maxes, random_start=random_start)
    else:
        raise ValueError(f"invalid attack attack_type {attack_type}")
    
    return features