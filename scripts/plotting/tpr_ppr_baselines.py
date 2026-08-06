import pandas as pd
import yaml
import matplotlib.pyplot as plt

from fair_safe_opt.utils.paths import baseline_robustness_dir, config_path
from fair_safe_opt.evaluation.plot import metric_vs_eps_plot
from fair_safe_opt.reporting.robustness_tables import seed_averages_and_stddev

with open(config_path()) as f:
    config = yaml.safe_load(f)

group0_table_path = baseline_robustness_dir() / "pgd_group_0.csv"
group1_table_path = baseline_robustness_dir() / "pgd_group_1.csv"

architectures = config["architecture_sweep"]
n = len(architectures)

group0_df = pd.read_csv(group0_table_path)
group0_df = seed_averages_and_stddev(group0_df)

group1_df = pd.read_csv(group1_table_path)
group1_df = seed_averages_and_stddev(group1_df)

fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=False)
axes = axes[0]
metric_vs_eps_plot(
    {"Group 0 Advantaged TPR": group0_df, "Group 1 AdvantagedTPR": group1_df},
    "advantaged_tpr",
    axes,
)
metric_vs_eps_plot({"Group 0 PPR": group0_df, "Group 1 PPR": group1_df}, "positive_pred_rate", axes)

handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    if ax.get_legend() is not None:
        ax.get_legend().remove()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)

fig.savefig(baseline_robustness_dir() / "act1_per_group.png", dpi=150, bbox_inches="tight")
plt.close()
