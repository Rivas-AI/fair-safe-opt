'''
fair_adv_tolerance_sweep_eval.py
----------------------------------
This is code to create evaluation tables comparing the fairness-constrained adversarially trained
models across the tolerance (tau) sweep. Each row is tagged with the tolerance it was trained at.
'''
import yaml
from pathlib import Path
import pandas as pd
from fair_safe_opt.evaluation.robustness import tolerance_sweep_evaluation
from fair_safe_opt.utils.paths import TRAINING_CONFIG_PATH, fair_adversarial_sweep_results_dir


_CONFIG_PATH = TRAINING_CONFIG_PATH
with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

_FAIRNESS_METRIC = config['fairness_metric']
_TOLERANCE_SWEEP = config['tolerance_sweep']


def main():
    out_dir = fair_adversarial_sweep_results_dir(_FAIRNESS_METRIC)

    Path.mkdir(out_dir, exist_ok=True, parents=True)

    metrics_dict = tolerance_sweep_evaluation(config['architecture_sweep'], config['epsilon_sweep'], config['seed_sweep'], _TOLERANCE_SWEEP, _FAIRNESS_METRIC, training_style='dp_fair', training_epsilon=config['training_epsilon'], as_deltas=False)

    for name, table in metrics_dict.items():
        table = table.sort_values(['architecture', 'tolerance'])
        print(name)
        print(table.head(20))
        table.to_csv(Path(out_dir / f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
