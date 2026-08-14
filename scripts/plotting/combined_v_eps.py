import matplotlib.pyplot as plt
import pandas as pd
import yaml

from fair_safe_opt.evaluation.plot import metric_vs_eps_plot
from fair_safe_opt.reporting.robustness_tables import seed_averages_and_stddev
from fair_safe_opt.utils.paths import (
    baseline_robustness_dir,
    config_path,
    fair_adversarial_sweep_results_dir,
    madry_adversarial_robustness_dir,
    results_tables_dir,
)

MODELS = ["baseline", "madry", "fair-adv"]
TAU = 0.015
METRICS = ["balanced_accuracy", "demographic_parity_difference"]
SAVE_NAME = "act3_dp_acc"

with open(config_path()) as f:
    config = yaml.safe_load(f)

baseline_table_path = baseline_robustness_dir() / "pgd_combined.csv"
madry_table_path = madry_adversarial_robustness_dir() / "combined.csv"
fair_adv_table_path = fair_adversarial_sweep_results_dir(config["fairness_metric"]) / "combined.csv"

architectures = config["architecture_sweep"]
n = len(architectures)

path_dict = {
    "baseline": baseline_table_path,
    "madry": madry_table_path,
    "fair-adv": fair_adv_table_path,
}

dfs = {}
for model in MODELS:
    df = pd.read_csv(path_dict[model])

    if model == "fair-adv" and TAU is not None:
        df = df[df["tolerance"] == TAU]
        df.drop(columns=["tolerance"], inplace=True)

    dfs[model] = seed_averages_and_stddev(df)

fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=False)
axes = axes[0]

for metric in METRICS:
    metric_vs_eps_plot(dfs, metric, axes)

plt.savefig(results_tables_dir() / f"{SAVE_NAME}.png", dpi=150, bbox_inches="tight")
