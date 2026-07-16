'''
train_baselines.py
----------------------------------
Train (or load cached) baseline ERM models for all architectures and seeds.
'''
import torch
import yaml
from pathlib import Path

from fair_safe_opt.utils.seeds import set_seed
from fair_safe_opt.utils.devices import set_device
from fair_safe_opt.data.adult import load_adult, AdultSplit
from fair_safe_opt.training.training_history import save_training_history
from fair_safe_opt.models.build import build_model
from fair_safe_opt.training.erm import train_erm

_CONFIG_PATH = Path.cwd() / 'configs' / 'experiments' / 'training_configs.yaml'

with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def save_config_snapshot(seed, architecture, hidden_dim, dropout, outfile):
    config_snapshot = {
        'seed': seed,
        'architecture': architecture,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
    }
    with open(outfile, 'w') as f:
        yaml.dump(config_snapshot, f)


def get_or_train_baseline(seed: int, architecture: str) -> tuple[torch.nn.Module, AdultSplit]:
    save_dir = Path(f"trained_models/baseline/{architecture}_eps0.0_{seed}")
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
        save_config_snapshot(seed, architecture, config['hidden_dim'], config['dropout'], save_dir / 'config.yaml')
        model.to(device)
        x_train, y_train, x_val, y_val = split.x_train.to(device), split.y_train.to(device), split.x_val.to(device), split.y_val.to(device)
        history = train_erm(model, x_train, y_train, x_val, y_val, out_dir=save_dir, epochs=config['epochs'], patience=config['patience'], learning_rate=config['learning_rate'])
        save_training_history(history, f'{save_dir}/training_history.json')
        model.to('cpu')

    return model, split


def main():
    for architecture in config['architecture_sweep']:
        for seed in config['seed_sweep']:
            print(f"\n--- arch={architecture} seed={seed} ---")
            get_or_train_baseline(seed, architecture)


if __name__ == "__main__":
    main()
