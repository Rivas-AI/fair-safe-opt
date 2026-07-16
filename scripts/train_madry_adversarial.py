'''
train_adversarial.py
----------------------------------
Train (or load cached) PGD-adversarially trained models for all architectures and seeds.
'''
import torch
import yaml
from pathlib import Path

from fair_safe_opt.utils.seeds import set_seed
from fair_safe_opt.utils.devices import set_device
from fair_safe_opt.data.adult import load_adult, AdultSplit
from fair_safe_opt.models.build import build_model
from fair_safe_opt.training.adversarial import train_pgd_madry

_CONFIG_PATH = Path.cwd() / 'configs' / 'experiments' / 'training_configs.yaml'

with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

_TRAINING_EPS = config.get('training_epsilon', 0.3)


def save_config_snapshot(seed, architecture, hidden_dim, dropout, training_epsilon, outfile):
    config_snapshot = {
        'seed': seed,
        'architecture': architecture,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'training_epsilon': training_epsilon,
    }
    with open(outfile, 'w') as f:
        yaml.dump(config_snapshot, f)


def get_or_train_adversarial(seed: int, architecture: str, training_epsilon: float) -> tuple[torch.nn.Module, AdultSplit]:
    save_dir = Path(f"trained_models/adversarial/madry_pgd/{architecture}_eps{training_epsilon}_{seed}")
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = set_device(config['device'])

    split = load_adult(random_state=seed)
    model = build_model(architecture, split, config['hidden_dim'], config['dropout'])

    checkpoint = save_dir / "best_model.pt"
    if checkpoint.exists():
        print(f"Loading existing model from {checkpoint}")
        model.load_state_dict(torch.load(checkpoint))
    else:
        save_config_snapshot(seed, architecture, config['hidden_dim'], config['dropout'], training_epsilon, save_dir / 'config.yaml')
        model.to(device)
        x_train, y_train, x_val, y_val = split.x_train.to(device), split.y_train.to(device), split.x_val.to(device), split.y_val.to(device)
        train_pgd_madry(model, x_train, y_train, x_val, y_val, epsilon=training_epsilon, out_dir=save_dir, continuous_mask=split.continuous_mask, epochs=config['epochs'], patience=config['patience'], learning_rate=config['learning_rate'])
        model.to('cpu')

    return model, split


def main():
    for architecture in ['logistic', 'small_mlp']:
        for seed in config['seeds']:
            print(f"\n--- arch={architecture} training epsilon={_TRAINING_EPS} seed={seed} ---")
            get_or_train_adversarial(seed, architecture, _TRAINING_EPS)


if __name__ == "__main__":
    main()
