import yaml
import pandas as pd
import matplotlib.pyplot as plt

from fair_safe_opt.utils.paths import baseline_robustness_dir, config_path, results_tables_dir
from fair_safe_opt.reporting.robustness_tables import seed_averages_and_stddev
from fair_safe_opt.evaluation.plot import metric_vs_eps_plot

with open(config_path()) as f:
    config = yaml.safe_load(f)

plot_metric = 'balanced_accuracy'
architectures = config['architecture_sweep']

pgd_path = baseline_robustness_dir() / 'pgd_combined.csv'
fgsm_path = baseline_robustness_dir() / 'fgsm_combined.csv'

pgd_df = seed_averages_and_stddev(pd.read_csv(pgd_path))
fgsm_df = seed_averages_and_stddev(pd.read_csv(fgsm_path))

n = len(architectures)

fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=True)
metric_vs_eps_plot({'FGSM Attacked': fgsm_df, 'PGD Attacked': pgd_df}, plot_metric, axes)
fig.savefig(results_tables_dir() / 'fgsm_vs_pgd.png', dpi=150, bbox_inches="tight")
plt.close()


