import pandas as pd
import yaml
import matplotlib.pyplot as plt

from fair_safe_opt.utils.paths import baseline_robustness_dir, config_path
from fair_safe_opt.evaluation.plot import metric_vs_eps_plot
from fair_safe_opt.reporting.robustness_tables import seed_averages_and_stddev

with open(config_path()) as f:
    config = yaml.safe_load(f)

baseline_table_path = baseline_robustness_dir() / "pgd_combined.csv"

architectures = config["architecture_sweep"]
n = len(architectures)

baseline_df = pd.read_csv(baseline_table_path)
baseline_df = seed_averages_and_stddev(baseline_df)

fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=False)
axes = axes[0]
metric_vs_eps_plot({"DPD": baseline_df}, "demographic_parity_difference", axes)
metric_vs_eps_plot({"EOpp": baseline_df}, "equal_opportunity_difference", axes)
fig.savefig(baseline_robustness_dir() / "dp_eopp_baselines.png", dpi=150, bbox_inches="tight")
plt.close()
