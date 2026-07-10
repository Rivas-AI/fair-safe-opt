import torch
import yaml
from pathlib import Path
import pandas as pd
from fair_safe_opt.utils.directories import create_report_dir
from fair_safe_opt.utils.seeds import set_seed
from fair_safe_opt.utils.devices import set_device

from fair_safe_opt.data.adult import load_adult

from fair_safe_opt.models.build import build_model

from fair_safe_opt.training.adversarial import train_pgd_adversarial
from fair_safe_opt.evaluation.evaluate import save_evaluation_plots, create_metric_table
from fair_safe_opt.metrics.attacks import attack_success_by_group
from fair_safe_opt.evaluation.plot import plot_history
from fair_safe_opt.attacks.attack import attack_features
from fair_safe_opt.reporting.console_table import print_delta_table

_CONFIG_PATH = Path.cwd() / 'configs' / 'experiments' / 'training_configs.yaml'
with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

_METRICS = ['balanced_accuracy', 'auc', 'advantaged_tpr']
_GROUP = 'group_0'


def run_single(seed, architecture, epsilon):
    set_seed(seed)
    device = set_device(config['device'])

    out_dir = create_report_dir(architecture, config['method'], seed)

    split = load_adult(random_state=seed)

    model = build_model(architecture, split, config['hidden_dim'], config['dropout'])

    model.to(device)
    x_train, y_train, x_val, y_val = split.x_train.to(device), split.y_train.to(device), split.x_val.to(device), split.y_val.to(device)

    history = train_pgd_adversarial(model, config['method'], x_train, y_train, x_val, y_val, out_dir, epochs=config['epochs'], patience=config['patience'], learning_rate=config['learning_rate'], device=config['device'])

    model.to('cpu')

    plot_history(history, out_dir)

    # Evaluate baselines
    base_preds, base_probs = model.preds_probs(split.x_test, config['threshold'])
    baseline_metrics_df = create_metric_table(base_preds, base_probs, split.y_test, split.a_test)
    save_evaluation_plots(base_preds, base_probs, split.y_test, split.a_test, out_dir, prefix='baseline')

    # Evaluate adversarial impact
    alpha = 2 * epsilon / config['steps']
    adv_features = attack_features(model, config['attack'], split.x_test, split.y_test, split.continuous_mask, split.feature_min, split.feature_max, epsilon, config['steps'], alpha, random_start=True)
    adv_preds, adv_probs = model.preds_probs(adv_features, config['threshold'])
    adv_metrics_df = create_metric_table(adv_preds, adv_probs, split.y_test, split.a_test)
    save_evaluation_plots(adv_preds, adv_probs, split.y_test, split.a_test, out_dir, prefix='attacked')

    delta_df = adv_metrics_df - baseline_metrics_df
    atk_success_rates = attack_success_by_group(split.y_test, base_preds, adv_preds, split.a_test)

    delta_df['atk_success'] = atk_success_rates
    return baseline_metrics_df, delta_df


def main():
    all_results = []
    keys = []
    baseline_rows = []
    delta_rows = []
    seen_baselines = set()

    for seed in config['seeds']:
        for architecture in config['architectures']:
            for epsilon in config['epsilons']:
                print(f"\n--- seed={seed}  arch={architecture}  epsilon={epsilon} ---")
                baseline_df, delta_df = run_single(seed, architecture, epsilon)
                all_results.append(delta_df)
                keys.append((seed, architecture, epsilon))

                if (seed, architecture) not in seen_baselines:
                    seen_baselines.add((seed, architecture))
                    c = baseline_df.loc[_GROUP]
                    baseline_rows.append({'architecture': architecture, **{m: c[m] for m in _METRICS}})

                c = delta_df.loc[_GROUP]
                delta_rows.append({'architecture': architecture, 'epsilon': epsilon, **{m: c[m] for m in _METRICS}, 'atk_success': c['atk_success']})

    combined = pd.concat(all_results, keys=keys, names=['seed', 'architecture', 'epsilon'])
    combined.to_csv('combined_results.csv')

    print_delta_table(baseline_rows, delta_rows, config['architectures'], _METRICS)


if __name__ == "__main__":
    main()
