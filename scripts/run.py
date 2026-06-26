import torch
import yaml
from pathlib import Path
import pandas as pd
from fair_safe_opt.utils.directories import create_report_dir
from fair_safe_opt.utils.seeds import set_seed
from fair_safe_opt.utils.devices import set_device

from fair_safe_opt.data.adult import load_adult

from fair_safe_opt.models.build import build_model

from fair_safe_opt.training.train import train_model

from fair_safe_opt.evaluation.evaluate import save_evaluation_plots, create_metric_table
from fair_safe_opt.metrics.attacks import attack_success_by_group
from fair_safe_opt.evaluation.plot import plot_history
from fair_safe_opt.attacks.attack import attack_features

_CONFIG_PATH = Path.cwd() / 'configs' / 'experiments' / 'training_configs.yaml'
with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SEEDS = [0, 1, 2]
ARCHITECTURES = ['logistic', 'small_mlp']
EPSILONS = [0.1, 0.2, 0.3]


_ARCH_LABELS = {'logistic': 'Logistic', 'small_mlp': 'Small MLP'}
_METRICS = ['balanced_accuracy', 'auc', 'demographic_parity_difference', 'equal_opportunity_difference']
_COL_HEADERS = ['bal_acc', 'auc', 'DPD', 'Eop']


def run_single(seed, architecture, epsilon):
    set_seed(seed)
    device = set_device(config['device'])

    out_dir = create_report_dir(architecture, config['method'], seed)

    split = load_adult(random_state=seed)

    model = build_model(architecture, split, config['hidden_dim'], config['dropout'])

    model.to(device)
    x_train, y_train, x_val, y_val = split.x_train.to(device), split.y_train.to(device), split.x_val.to(device), split.y_val.to(device)

    history = train_model(model, config['method'], x_train, y_train, x_val, y_val, out_dir, epochs=config['epochs'], patience=config['patience'], learning_rate=config['learning_rate'], device=config['device'])

    model.to('cpu')

    plot_history(history, out_dir)

    # Evaluate baselines
    base_preds, base_probs = model.preds_probs(split.x_test, config['threshold'])
    baseline_metrics_df = create_metric_table(base_preds, base_probs, split.y_test, split.a_test)
    save_evaluation_plots(base_preds, base_probs, split.y_test, split.a_test, out_dir, prefix='baseline')

    # Evaluate adversarial impact
    adv_features = attack_features(model, config['attack'], split.x_test, split.y_test, split.continuous_mask, split.feature_min, split.feature_max, epsilon, config['steps'], config['alpha'])
    adv_preds, adv_probs = model.preds_probs(adv_features, config['threshold'])
    adv_metrics_df = create_metric_table(adv_preds, adv_probs, split.y_test, split.a_test)
    save_evaluation_plots(adv_preds, adv_probs, split.y_test, split.a_test, out_dir, prefix='attacked')

    delta_df = adv_metrics_df - baseline_metrics_df
    atk_success_rates = attack_success_by_group(split.y_test, base_preds, adv_preds, split.a_test)

    delta_df['atk_success'] = atk_success_rates
    return baseline_metrics_df, delta_df


def _fmt_val(v):
    return f"{v * 100:.2f}"


def _fmt_delta(v):
    s = f"{v * 100:.2f}"
    return f"(+{s})" if v >= 0 else f"({s})"


def _print_table(baseline_rows, delta_rows):
    baseline_df = pd.DataFrame(baseline_rows).groupby('architecture')[_METRICS].mean()
    delta_df = (pd.DataFrame(delta_rows)
                .groupby(['architecture', 'epsilon'])[_METRICS]
                .mean()
                .reset_index())

    col_w = 12
    header = f"{'Model':<14} {'Epsilon':>8}" + "".join(f"{h:>{col_w}}" for h in _COL_HEADERS)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for arch in ARCHITECTURES:
        b = baseline_df.loc[arch]
        vals = "".join(f"{_fmt_val(b[m]):>{col_w}}" for m in _METRICS)
        print(f"{_ARCH_LABELS[arch]:<14} {'0':>8}{vals}")

        arch_deltas = delta_df[delta_df['architecture'] == arch].sort_values('epsilon')
        for _, row in arch_deltas.iterrows():
            vals = "".join(f"{_fmt_delta(row[m]):>{col_w}}" for m in _METRICS)
            print(f"{'':14} {row['epsilon']:>8.2f}{vals}")

    print("=" * len(header))


def main():
    all_results = []
    keys = []
    baseline_rows = []
    delta_rows = []
    seen_baselines = set()

    for seed in SEEDS:
        for architecture in ARCHITECTURES:
            for epsilon in EPSILONS:
                print(f"\n--- seed={seed}  arch={architecture}  epsilon={epsilon} ---")
                baseline_df, delta_df = run_single(seed, architecture, epsilon)
                all_results.append(delta_df)
                keys.append((seed, architecture, epsilon))

                if (seed, architecture) not in seen_baselines:
                    seen_baselines.add((seed, architecture))
                    c = baseline_df.loc['combined']
                    baseline_rows.append({'architecture': architecture, **{m: c[m] for m in _METRICS}})

                c = delta_df.loc['combined']
                delta_rows.append({'architecture': architecture, 'epsilon': epsilon, **{m: c[m] for m in _METRICS}})

    combined = pd.concat(all_results, keys=keys, names=['seed', 'architecture', 'epsilon'])
    combined.to_csv('combined_results.csv')

    _print_table(baseline_rows, delta_rows)


if __name__ == "__main__":
    main()
