import yaml
from pathlib import Path
import pandas as pd
import torch

from fair_safe_opt.data.adult import load_adult 
from fair_safe_opt.evaluation.evaluate import create_metric_table
from fair_safe_opt.metrics.attacks import attack_success_by_group
from fair_safe_opt.attacks.pgd import pgd_attack
from fair_safe_opt.reporting.robustness_tables import average_over_seeds, to_scores, split_by_group, values_as_deltas
from fair_safe_opt.models.build import build_model
from fair_safe_opt.utils.seeds import set_seed

_CONFIG_PATH = Path.cwd() / 'configs' / 'experiments' / 'training_configs.yaml'
_OUT_DIR = Path.cwd() / 'results' / 'robustness' / 'adversarial' / 'static_pgd'
Path.mkdir(_OUT_DIR, parents=True, exist_ok=True)
_PGD_SAVE_DIR = Path(f'data/pgd_testing_data')
Path.mkdir(_PGD_SAVE_DIR, parents=True, exist_ok=True)

with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def evaluate_single(model, split, architecture, epsilon, seed):
    alpha = 2 * epsilon / config['steps']
    adv_features = pgd_attack(model, split.x_test, split.y_test, continuous_mask=split.continuous_mask, feature_min=split.feature_min, feature_max=split.feature_max, epsilon=epsilon, steps=config['steps'], alpha=alpha, random_start=True)
    torch.save(adv_features, _PGD_SAVE_DIR / f'{architecture}_{seed}_{epsilon}')
    
    adv_preds, adv_probs = model.preds_probs(adv_features, config['threshold'])
    adv_metrics_df = create_metric_table(adv_preds, adv_probs, split.y_test, split.a_test)

    # Calculate and include attack success rates.
    base_preds, _ = model.preds_probs(split.x_test, config['threshold'])
    if epsilon == 0.0:
        adv_metrics_df['atk_success'] = 0.0
    else:
        adv_metrics_df['atk_success'] = attack_success_by_group(split.y_test, base_preds, adv_preds, split.a_test)

    adv_metrics_df.insert(0, 'seed', seed)
    adv_metrics_df.insert(0, 'epsilon', epsilon)
    adv_metrics_df.insert(0, 'architecture', architecture)
    
    return adv_metrics_df


def evaluation_sweep(architectures, epsilons, seeds, *, model_save_dir: Path = Path.cwd(), as_deltas: bool = False):
    '''
    Sweeps across all parameters, creating a table with all metrics for all data and returning it divided by group. 
    
    Returns: a dictionary with each table corresponding to 'combined', 'group_0', 'group_1'
    '''
    full_df = pd.DataFrame()

    for architecture in architectures:
        for epsilon in epsilons:
            for seed in seeds:
                set_seed(seed)
                split = load_adult(random_state=seed)
                
                print(f"\n--- arch={architecture}  evaluation epsilon={epsilon} seed={seed}---")
                model = build_model(architecture, split, hidden_dim=config['hidden_dim'], dropout=config['dropout'])
                checkpoint = model_save_dir / f"{architecture}_eps{config['training_epsilon']}_{seed}" / "best_model.pt"
                
                try:
                    model.load_state_dict(torch.load(checkpoint))
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"No trained static-adversarial model at {checkpoint}. "
                        f"Run train_static_adversarial.py first."
                        )
                    
                results_df = evaluate_single(model, split, architecture, epsilon, seed)
                full_df = pd.concat([full_df, results_df], axis=0)

    averaged_df = average_over_seeds(full_df)
    if as_deltas:
        delta_df = values_as_deltas(averaged_df)
        scores_df = to_scores(delta_df)
    else:
        scores_df = to_scores(averaged_df)
    
    group_tables = split_by_group(scores_df)
    return group_tables
