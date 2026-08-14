"""
adversarial_robustness_eval.py
----------------------------------
This is code to create an evaluation table for the adversarially trained model's robustness against adversarial attacks.
"""

import yaml
from pathlib import Path
from fair_safe_opt.evaluation.robustness import evaluation_sweep
from fair_safe_opt.utils.paths import (
    TRAINING_CONFIG_PATH,
    fair_adversarial_results_dir,
    fair_adversarial_root,
)


_CONFIG_PATH = TRAINING_CONFIG_PATH
with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

_FAIRNESS_METRIC = config["fairness_metric"]
_TOLERANCE = config["tolerance"]


def main():
    saved_model_loc = fair_adversarial_root(_FAIRNESS_METRIC, _TOLERANCE)
    out_dir = fair_adversarial_results_dir(_FAIRNESS_METRIC)

    Path.mkdir(out_dir, exist_ok=True, parents=True)

    metrics_dict = evaluation_sweep(
        config["architecture_sweep"],
        config["epsilon_sweep"],
        config["seed_sweep"],
        training_style="dp_fair",
        model_save_dir=saved_model_loc,
        training_epsilon=config["training_epsilon"],
        tolerance=_TOLERANCE,
    )

    for name, table in metrics_dict.items():
        print(name)
        print(table.head(8))
        table.to_csv(Path(out_dir / f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
