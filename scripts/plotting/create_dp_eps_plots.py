import pandas as pd
import yaml
import matplotlib.pyplot as plt

from fair_safe_opt.utils.paths import (
    baseline_robustness_dir,
    madry_adversarial_robustness_dir,
    results_tables_dir,
    config_path,
)
from fair_safe_opt.evaluation.plot import metric_vs_eps_plot
from fair_safe_opt.reporting.robustness_tables import seed_averages_and_stddev

with open(config_path()) as f:
    config = yaml.safe_load(f)

baseline_table_path = baseline_robustness_dir() / "pgd_combined.csv"
madry_table_path = madry_adversarial_robustness_dir() / "combined.csv"

architectures = config["architecture_sweep"]
n = len(architectures)

baseline_df = pd.read_csv(baseline_table_path)
baseline_df = seed_averages_and_stddev(baseline_df)

fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=False)
axes = axes[0]
metric_vs_eps_plot({"baseline": baseline_df}, "demographic_parity_difference", axes)
fig.savefig(baseline_robustness_dir() / "dp_vs_eps.png", dpi=150, bbox_inches="tight")
plt.close()

madry_df = pd.read_csv(madry_table_path)
madry_df = seed_averages_and_stddev(madry_df)

fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=False)
axes = axes[0]
metric_vs_eps_plot({"baseline": madry_df}, "demographic_parity_difference", axes)
fig.savefig(madry_adversarial_robustness_dir() / "dp_vs_eps.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot them against each other
fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=False)
axes = axes[0]
metric_vs_eps_plot(
    {"Baseline": baseline_df, "Madry": madry_df},
    "demographic_parity_difference",
    axes,
)
fig.savefig(results_tables_dir() / "dpd_baseline_vs_madry.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Plot group PPRates to see why Madry is lower DPD
baseline_gp1 = seed_averages_and_stddev(pd.read_csv(baseline_robustness_dir() / "pgd_group_1.csv"))
baseline_gp0 = seed_averages_and_stddev(pd.read_csv(baseline_robustness_dir() / "pgd_group_0.csv"))

madry_gp1 = seed_averages_and_stddev(
    pd.read_csv(madry_adversarial_robustness_dir() / "group_1.csv")
)
madry_gp0 = seed_averages_and_stddev(
    pd.read_csv(madry_adversarial_robustness_dir() / "group_0.csv")
)

fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=False)
axes = axes[0]
metric_vs_eps_plot(
    {
        "Baseline Group 0": baseline_gp0,
        "Baseline Group 1": baseline_gp1,
        "Madry Group 0": madry_gp0,
        "Madry Group 1": madry_gp1,
    },
    "positive_pred_rate",
    axes,
)
fig.savefig(results_tables_dir() / "group_pprs.png", dpi=150, bbox_inches="tight")
plt.close(fig)
