'''
baseline_robustness_eval.py
----------------------------------
This is code to create an evaluation table for the baseline model's robustness against adversarial attacks.
In later weeks, we will judge our improvement by how much robustness improves from the baseline. 
'''
import torch
import yaml
from pathlib import Path
import pandas as pd
from fair_safe_opt.utils.directories import create_report_dir
from fair_safe_opt.utils.seeds import set_seed
from fair_safe_opt.utils.devices import set_device

from fair_safe_opt.data.adult import load_adult

from fair_safe_opt.models.build import build_model

from fair_safe_opt.training.erm import train_erm
from fair_safe_opt.evaluation.evaluate import save_evaluation_plots, create_metric_table
from fair_safe_opt.metrics.attacks import attack_success_by_group
from fair_safe_opt.evaluation.plot import plot_history
from fair_safe_opt.attacks.pgd import pgd_attack
from fair_safe_opt.reporting.console_table import print_delta_table

_CONFIG_PATH = Path.cwd() / 'configs' / 'experiments' / 'training_configs.yaml'
with open(_CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def run_single(seed, architecture, epsilon):
    set_seed(seed)
    device = set_device(config['device'])

    out_dir = create_report_dir(architecture, 'erm', seed)

    split = load_adult(random_state=seed)
    model = build_model(architecture, split, config['hidden_dim'], config['dropout'])

    model.to(device)
    x_train, y_train, x_val, y_val = split.x_train.to(device), split.y_train.to(device), split.x_val.to(device), split.y_val.to(device)

    history = train_erm(model, x_train, y_train, x_val, y_val, out_dir=out_dir, epochs=config['epochs'], patience=config['patience'], learning_rate=config['learning_rate'])
    model.to('cpu')

    # Evaluate adversarial impact
    alpha = 2 * epsilon / config['steps']
    adv_features = pgd_attack(model, split.x_test, split.y_test, continuous_mask=split.continuous_mask, feature_min=split.feature_min, feature_max=split.feature_max, epsilon=epsilon, steps=config['steps'], alpha=alpha, random_start=True)
    adv_preds, adv_probs = model.preds_probs(adv_features, config['threshold'])
    adv_metrics_df = create_metric_table(adv_preds, adv_probs, split.y_test, split.a_test)
    # save_evaluation_plots(adv_preds, adv_probs, split.y_test, split.a_test, out_dir, prefix='attacked')

    
    # Calculate and include attack success rates. 
    base_preds, _ = model.preds_probs(split.x_test, config['threshold'])
    if epsilon == 0.0:
        adv_metrics_df['atk_success'] = 0.0
    else:
        adv_metrics_df['atk_success'] = attack_success_by_group(split.y_test, base_preds, adv_preds, split.a_test)

    adv_metrics_df.insert(0, 'seed', seed)
    adv_metrics_df.insert(0, 'epsilon', epsilon)
    adv_metrics_df.insert(0, 'architecture', architecture)
    return adv_metrics_df


def create_baseline_robustness_table():
    full_df = pd.DataFrame()

    for architecture in ['logistic', 'small_mlp']:
        for epsilon in config['epsilons']:
            for seed in config['seeds']:
                print(f"\n--- arch={architecture}  epsilon={epsilon} seed={seed}---")
                results_df = run_single(seed, architecture, epsilon)
                full_df = pd.concat([full_df, results_df], axis=0)

    full_df.to_csv("results/baseline_robustness_table.csv", index=False)


def decimals_to_scores(v):
    return float(f"{v * 100:.2f}")


def _fmt_delta(v):
    s = f"{v * 100:.2f}"
    return f"(+{s})" if v >= 0 else f"({s})"


def values_as_deltas(df):
    baseline_row = pd.Series()
    
    for idx, row in df.iterrows():
        if row['epsilon'] == 0.0:
            baseline_row = row
        else:
            for col in df.columns:
                if col not in ['seed', 'architecture', 'epsilon', 'group']:
                    row[col] = row[col] - baseline_row[col]
        
        df.iloc[idx] = row
        
    return df


def main():
    make_new_table = False
    if make_new_table:
        create_baseline_robustness_table()
        
    full_df = pd.read_csv("results/baseline_robustness_table.csv")
    
    # average based on seed
    full_w_seed_averages = full_df.groupby(['architecture', 'epsilon', 'group']).mean().reset_index().drop('seed', axis=1)
    
    # Change values to scores
    numeric_cols = full_w_seed_averages.select_dtypes(include=['float64']).columns
    metric_cols = [col for col in numeric_cols if col not in ['epsilon', 'seed']]
    full_avg_scores = full_w_seed_averages.copy()
    full_avg_scores[metric_cols] = full_w_seed_averages[metric_cols].map(decimals_to_scores)
    
    # Separate by group
    combined_df = full_avg_scores[full_avg_scores['group'] == 'combined'].copy()
    combined_df = combined_df.drop('group', axis=1, inplace=False).reset_index(drop=True)
    combined_df.to_csv("results/baseline_robustness_table_combined.csv", index=False)
    group0_df = full_avg_scores[full_avg_scores['group'] == 'group_0'].copy()
    group0_df = group0_df.drop(['group', 'demographic_parity_difference', 'equal_opportunity_difference'], axis=1, inplace=False).reset_index(drop=True)
    group0_df.to_csv("results/baseline_robustness_table_group_0.csv", index=False)
    group1_df = full_avg_scores[full_avg_scores['group'] == 'group_1'].copy()
    group1_df = group1_df.drop(['group', 'demographic_parity_difference', 'equal_opportunity_difference'], axis=1, inplace=False).reset_index(drop=True)
    group1_df.to_csv("results/baseline_robustness_table_group_1.csv", index=False)

    combined_delta = values_as_deltas(combined_df)
    print(combined_delta)
    print()
    group0_delta = values_as_deltas(group0_df)
    print(group0_delta)
    print()
    group1_delta = values_as_deltas(group1_df)
    print(group1_delta)
    print()

if __name__ == "__main__":
    main()
