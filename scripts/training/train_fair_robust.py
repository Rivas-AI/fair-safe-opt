"""
train_adversarial.py
----------------------------------
Train (or load cached) PGD-adversarially trained models for all architectures and seeds.
"""

import torch
import yaml
import shutil

from fair_safe_opt.utils.seeds import set_seed
from fair_safe_opt.utils.devices import set_device
from fair_safe_opt.data.adult import load_adult, AdultSplit
from fair_safe_opt.models.build import build_model
from fair_safe_opt.training.fair_adversarial import fairness_constrained_pgd
from fair_safe_opt.training.training_history import save_training_history
from fair_safe_opt.utils.paths import TRAINING_CONFIG_PATH, fair_adversarial_run_dir

_CONFIG_PATH = TRAINING_CONFIG_PATH

with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# Model Params
_TRAINING_EPS = config["training_epsilon"]
_FAIRNESS_METRIC = config["fairness_metric"]
_PENALTY = config["initial_penalty"]
_MULTIPLIER = config["initial_multiplier"]
_DUAL_LR = config["dual_lr"]


def train_model(
    architecture: str, training_epsilon: float, tolerance: float, seed: int, metric: str
) -> tuple[torch.nn.Module, AdultSplit]:
    save_dir = fair_adversarial_run_dir(metric, tolerance, architecture, training_epsilon, seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = set_device(config["device"])

    split = load_adult(random_state=seed)
    model = build_model(architecture, split, config["hidden_dim"], config["dropout"])

    checkpoint = save_dir / "best_model.pt"
    if checkpoint.exists():
        print(f"Loading existing model from {checkpoint}")
        model.load_state_dict(torch.load(checkpoint))
    else:
        shutil.copy(_CONFIG_PATH, save_dir)
        model.to(device)
        x_train, y_train, a_train, x_val, y_val, a_val = (
            split.x_train.to(device),
            split.y_train.to(device),
            split.a_train.to(device),
            split.x_val.to(device),
            split.y_val.to(device),
            split.a_val.to(device),
        )
        history = fairness_constrained_pgd(
            model,
            x_train,
            y_train,
            a_train,
            x_val,
            y_val,
            a_val,
            epsilon=training_epsilon,
            out_dir=save_dir,
            continuous_mask=split.continuous_mask,
            epochs=config["epochs"],
            patience=config["patience"],
            learning_rate=config["learning_rate"],
            fairness_metric=metric,
            tolerance=tolerance,
            multiplier=_MULTIPLIER,
            dual_lr=_DUAL_LR,
        )
        save_training_history(history, save_dir / "training_history.json")
        model.to("cpu")

    return model, split


def main():
    for architecture in config["architecture_sweep"]:
        for tolerance in config["tolerance_sweep"]:
            for seed in config["seed_sweep"]:
                train_model(architecture, _TRAINING_EPS, tolerance, seed, _FAIRNESS_METRIC)


if __name__ == "__main__":
    main()
