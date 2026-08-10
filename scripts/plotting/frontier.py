import pandas as pd
import yaml
import matplotlib.pyplot as plt

from fair_safe_opt.utils.paths import (
    madry_adversarial_robustness_dir,
    fair_adversarial_sweep_results_dir,
    results_tables_dir,
    config_path,
)
from fair_safe_opt.reporting.robustness_tables import seed_averages_and_stddev

EPSILON = 0.0
X_METRIC = 'demographic_parity_difference'
Y_METRIC = 'balanced_accuracy'


def fair_adv_frontier(table_path, epsilon):
    """One seed-averaged point per tolerance, at the given epsilon.

    seed_averages_and_stddev groups on (architecture, epsilon) only, so tolerance
    would be averaged away if left in place. Instead we split by tolerance, average
    each slice on its own, then re-attach the tolerance value to the result.
    """
    df = pd.read_csv(table_path)

    frames = []
    for tau in sorted(df['tolerance'].unique()):
        sub = df[df['tolerance'] == tau].drop(columns=['tolerance'])
        sub = seed_averages_and_stddev(sub)
        sub = sub[sub['epsilon'] == epsilon].copy()
        sub['tolerance'] = tau
        frames.append(sub)

    return pd.concat(frames, ignore_index=True)


def madry_anchor(table_path, epsilon):
    """Single seed-averaged point per architecture at the given epsilon."""
    df = pd.read_csv(table_path)
    df = seed_averages_and_stddev(df)
    return df[df['epsilon'] == epsilon].copy()


def frontier_plot(fair_df, madry_df, architectures, axes):
    """Fairness-accuracy frontier, one panel per architecture, onto `axes`."""
    for ax, architecture in zip(axes, architectures):
        fair = fair_df[fair_df['architecture'] == architecture].sort_values('tolerance')
        ax.errorbar(
            fair[X_METRIC],
            fair[Y_METRIC],
            xerr=fair[f"{X_METRIC}_std"],
            yerr=fair[f"{Y_METRIC}_std"],
            marker='o',
            capsize=3,
            label='fair-adv frontier',
        )
        for _, row in fair.iterrows():
            ax.annotate(
                f"τ={row['tolerance']}",
                (row[X_METRIC], row[Y_METRIC]),
                textcoords='offset points',
                xytext=(6, -6),
                ha='left',
                va='top',
                fontsize=8,
            )

        madry = madry_df[madry_df['architecture'] == architecture]
        ax.errorbar(
            madry[X_METRIC],
            madry[Y_METRIC],
            xerr=madry[f"{X_METRIC}_std"],
            yerr=madry[f"{Y_METRIC}_std"],
            marker='D',
            color='crimson',
            capsize=3,
            linestyle='none',
            label='Madry (unconstrained)',
        )

        ax.set_title(architecture)
        ax.set_xlabel(X_METRIC)

    axes[0].set_ylabel(Y_METRIC)


def main():
    with open(config_path()) as f:
        config = yaml.safe_load(f)

    architectures = config['architecture_sweep']
    fair_adv_table_path = (
        fair_adversarial_sweep_results_dir(config['fairness_metric']) / "combined.csv"
    )
    madry_table_path = madry_adversarial_robustness_dir() / "combined.csv"

    fair_df = fair_adv_frontier(fair_adv_table_path, EPSILON)
    madry_df = madry_anchor(madry_table_path, EPSILON)

    n = len(architectures)
    fig, axes = plt.subplots(1, n, sharey=True, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    frontier_plot(fair_df, madry_df, architectures, axes)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

    save_path = results_tables_dir() / f"act3_frontier_eps{EPSILON}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


if __name__ == '__main__':
    main()
