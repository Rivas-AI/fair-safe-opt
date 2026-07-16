'''
adversarial_robustness_eval.py
----------------------------------
This is code to create an evaluation table for the adversarially trained model's robustness against adversarial attacks.
'''
import yaml
from pathlib import Path
import pandas as pd
from fair_safe_opt.evaluation.robustness import evaluation_sweep


_TRAINING_STYLE = 'madry'

_CONFIG_PATH = Path.cwd() / 'configs' / 'experiments' / 'training_configs.yaml'
with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def main():
    if _TRAINING_STYLE == 'static':
        saved_model_loc = Path('trained_models/adversarial/static_pgd')
        out_dir = Path('results/robustness/adversarial/static_pgd')
    elif _TRAINING_STYLE == 'madry':
        saved_model_loc = Path('trained_models/adversarial/madry_pgd')
        out_dir = Path('results/robustness/adversarial/madry_pgd')
    else:
        raise ValueError("training style must be static or madry.")
        
    Path.mkdir(out_dir, exist_ok=True, parents=True)
    
    metrics_dict = evaluation_sweep(config['architecture_sweep'], config['epsilon_sweep'], config['seed_sweep'], model_save_dir=saved_model_loc, training_epsilon=config['training_epsilon'], as_deltas=True)
    
    for name, table in metrics_dict.items():
        print(name)
        print(table.head())
        table.to_csv(Path(out_dir / f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
