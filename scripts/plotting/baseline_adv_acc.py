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
madry_df = pd.read_csv(madry_table_path)
madry_df = seed_averages_and_stddev(madry_df)

fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=False)
axes = axes[0]
metric_vs_eps_plot({"baseline": baseline_df, "Adversarial": madry_df}, "balanced_accuracy", axes)
fig.savefig(results_tables_dir() / "act2acc.png", dpi=150, bbox_inches="tight")
plt.close()
