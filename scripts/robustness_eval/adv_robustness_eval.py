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
    madry_adversarial_results_dir,
    madry_adversarial_root,
    static_adversarial_results_dir,
    static_adversarial_root,
)


_TRAINING_STYLE = "madry"

_CONFIG_PATH = TRAINING_CONFIG_PATH
with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def main():
    if _TRAINING_STYLE == "static":
        saved_model_loc = static_adversarial_root()
        out_dir = static_adversarial_results_dir()
    elif _TRAINING_STYLE == "madry":
        saved_model_loc = madry_adversarial_root()
        out_dir = madry_adversarial_results_dir()
    else:
        raise ValueError("training style must be static or madry.")

    Path.mkdir(out_dir, exist_ok=True, parents=True)

    metrics_dict = evaluation_sweep(
        config["architecture_sweep"],
        config["epsilon_sweep"],
        config["seed_sweep"],
        training_style=_TRAINING_STYLE,
        model_save_dir=saved_model_loc,
        training_epsilon=config["training_epsilon"],
        as_deltas=False,
    )

    for name, table in metrics_dict.items():
        print(name)
        print(table.head(8))
        table.to_csv(Path(out_dir / f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
