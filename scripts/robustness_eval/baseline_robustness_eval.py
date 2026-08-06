"""
baseline_robustness_eval.py
----------------------------------
This is code to create an evaluation table for the baseline trained model's robustness against adversarial attacks.
"""

import yaml
from pathlib import Path
from fair_safe_opt.evaluation.robustness import evaluation_sweep
from fair_safe_opt.utils.paths import config_path, baseline_robustness_dir, baseline_root

with open(config_path()) as f:
    config = yaml.safe_load(f)


def main():
    saved_model_loc = baseline_root()
    out_dir = baseline_robustness_dir()
    Path.mkdir(out_dir, exist_ok=True, parents=True)

    metrics_dict = evaluation_sweep(
        config["architecture_sweep"],
        config["epsilon_sweep"],
        config["seed_sweep"],
        training_style="baseline",
        model_save_dir=saved_model_loc,
        as_deltas=False,
        attack_type=config["attack_type"],
    )

    for name, table in metrics_dict.items():
        print(name)
        print(table.head())
        table.to_csv(Path(out_dir / f"{config['attack_type']}_{name}.csv"), index=False)


if __name__ == "__main__":
    main()
