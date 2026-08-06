'''
train_static_adversarial
----------
Create ten sets of PGD adversarial examples, train model on that data, and then evaluate based on a hidden 11th set
'''
import torch
import yaml
from fair_safe_opt.training.adversarial import pgd_attack
from fair_safe_opt.data.adult import load_adult, AdultSplit
from fair_safe_opt.models.build import build_model
from fair_safe_opt.training.erm import train_erm
from fair_safe_opt.utils.seeds import set_seed
from fair_safe_opt.training.training_history import save_training_history
from fair_safe_opt.evaluation.plot import plot_history
from fair_safe_opt.utils.paths import TRAINING_CONFIG_PATH, baseline_run_dir, pgd_training_data_path, static_adversarial_results_dir, static_adversarial_run_dir

from pathlib import Path

_TRAINING_EPS = 0.3
_NUM_PGD_EXAMPLES = 10
_CONFIG_PATH = TRAINING_CONFIG_PATH
_OUT_DIR = static_adversarial_results_dir()
Path.mkdir(_OUT_DIR, parents=True, exist_ok=True)

with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

def make_pgd_dataset(model, split, epsilon, steps, alpha):
    """
    Create PGD adversarial examples for the given model and test data.
    """
    aggregated_training_features = [split.x_train]
    aggregated_val_features = [split.x_val]
    
    for i in range(_NUM_PGD_EXAMPLES):
        adv_training_features = pgd_attack(model, split.x_train, split.y_train, continuous_mask=split.continuous_mask, feature_min=split.feature_min, feature_max=split.feature_max, epsilon=epsilon, steps=steps, alpha=alpha, random_start=True)
        aggregated_training_features.append(adv_training_features)
        
        adv_val_features = pgd_attack(model, split.x_val, split.y_val, continuous_mask=split.continuous_mask, feature_min=split.feature_min, feature_max=split.feature_max, epsilon=epsilon, steps=steps, alpha=alpha, random_start=True)
        aggregated_val_features.append(adv_val_features)
        
    aggregated_training_features = torch.cat(aggregated_training_features, dim=0)
    aggregated_val_features = torch.cat(aggregated_val_features)
    
    return aggregated_training_features, aggregated_val_features

def run_single(architecture, epsilon, seed):
    baseline_path = baseline_run_dir(architecture, seed) / 'best_model.pt'
    pgd_data_path = pgd_training_data_path(architecture, epsilon, seed)
    pgd_data_path.parent.mkdir(parents=True, exist_ok=True)
    save_dir = static_adversarial_run_dir(architecture, epsilon, seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    split = load_adult(random_state=seed)
    
    # Make PGD dataset for training
    if pgd_data_path.exists():
        print(f"Loading PGD dataset for {architecture} with epsilon={epsilon} and seed={seed}.")
        pgd_train_tensor, pgd_val_tensor = torch.load(pgd_data_path)
    else:
        print(f"Creating PGD dataset for {architecture} with epsilon={epsilon} and seed={seed}.")
        model = build_model(architecture, split, config['hidden_dim'], config['dropout'])
        model.load_state_dict(torch.load(baseline_path))
        
        steps = config['steps']
        alpha = 2 * (epsilon / steps)
        pgd_train_tensor, pgd_val_tensor = make_pgd_dataset(model, split, epsilon, steps, alpha)
        torch.save((pgd_train_tensor, pgd_val_tensor), pgd_data_path)
        
    if Path(save_dir / 'best_model.pt').exists():
        print(f"Loading model for {architecture} with epsilon={epsilon} and seed={seed}.")
        adv_model = build_model(architecture, split, hidden_dim=config['hidden_dim'], dropout=config['dropout'])
        adv_model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    else:
        print(f"Training model for {architecture} with epsilon={epsilon} and seed={seed}.")
        adv_model = build_model(architecture, split, hidden_dim=config['hidden_dim'], dropout=config['dropout'])
        adv_model.train()
        adv_model.to(config['device'])
        history = train_erm(adv_model, pgd_train_tensor, split.y_train.repeat(_NUM_PGD_EXAMPLES+1), pgd_val_tensor, split.y_val.repeat(_NUM_PGD_EXAMPLES+1), epochs=config['epochs'], learning_rate=config['learning_rate'], out_dir=save_dir)
        save_training_history(history, save_dir / 'training_history.json')
        plot_history(history, save_dir)
        adv_model.to('cpu')

def main():
    for architecture in ['logistic', 'small_mlp']:
        for seed in [0, 1, 2]:
            run_single(architecture, _TRAINING_EPS, seed)
            
if __name__ == '__main__':
    main()