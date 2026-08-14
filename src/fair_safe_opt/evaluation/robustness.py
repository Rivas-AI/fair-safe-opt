import yaml
from pathlib import Path
import pandas as pd
import torch

from fair_safe_opt.data.adult import load_adult
from fair_safe_opt.evaluation.evaluate import create_metric_table
from fair_safe_opt.metrics.attacks import attack_success_by_group
from fair_safe_opt.attacks.pgd import pgd_attack
from fair_safe_opt.attacks.fgsm import fgsm_attack
from fair_safe_opt.reporting.robustness_tables import (
    seed_averages_and_stddev,
    split_by_group,
    to_scores,
    values_as_deltas,
)
from fair_safe_opt.models.build import build_model
from fair_safe_opt.utils.seeds import set_seed
from fair_safe_opt.utils.paths import (
    TRAINING_CONFIG_PATH,
    fair_adversarial_root,
    pgd_cache_path,
    run_dir_name,
    fgsm_cache_path,
)

with open(TRAINING_CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def evaluate_single(
    model, split, architecture, epsilon, seed, training_style, attack_type="pgd", tolerance=None
):
    if attack_type == "pgd":
        alpha = 2 * epsilon / config["steps"]
        pgd_attacked_features_path = pgd_cache_path(
            training_style, architecture, seed, epsilon, tolerance
        )
        pgd_attacked_features_path.parent.mkdir(parents=True, exist_ok=True)

        if pgd_attacked_features_path.exists():
            adv_features = torch.load(pgd_attacked_features_path)
        else:
            adv_features = pgd_attack(
                model,
                split.x_test,
                split.y_test,
                continuous_mask=split.continuous_mask,
                feature_min=split.feature_min,
                feature_max=split.feature_max,
                epsilon=epsilon,
                steps=config["steps"],
                alpha=alpha,
                random_start=True,
            )
            torch.save(adv_features, pgd_attacked_features_path)

    elif attack_type == "fgsm":
        fgsm_attacked_features_path = fgsm_cache_path(training_style, architecture, epsilon, seed)
        fgsm_attacked_features_path.parent.mkdir(parents=True, exist_ok=True)
        if fgsm_attacked_features_path.exists():
            adv_features = torch.load(fgsm_attacked_features_path)
        else:
            adv_features = fgsm_attack(
                model,
                split.x_test,
                split.y_test,
                continuous_mask=split.continuous_mask,
                feature_min=split.feature_min,
                feature_max=split.feature_max,
                epsilon=epsilon,
            )
            torch.save(adv_features, fgsm_attacked_features_path)

    else:
        raise ValueError(f"unknown attack_type: {attack_type}")

    adv_preds, adv_probs = model.preds_probs(adv_features, config["threshold"])
    adv_metrics_df = create_metric_table(adv_preds, adv_probs, split.y_test, split.a_test)

    # Calculate and include attack success rates.
    base_preds, _ = model.preds_probs(split.x_test, config["threshold"])
    if epsilon == 0.0:
        adv_metrics_df["atk_success"] = 0.0
    else:
        adv_metrics_df["atk_success"] = attack_success_by_group(
            split.y_test, base_preds, adv_preds, split.a_test
        )

    adv_metrics_df = adv_metrics_df.round(4)
    adv_metrics_df.insert(0, "seed", seed)
    adv_metrics_df.insert(0, "epsilon", epsilon)
    adv_metrics_df.insert(0, "architecture", architecture)

    return adv_metrics_df


def evaluation_sweep(
    architectures,
    epsilons,
    seeds,
    training_style,
    *,
    attack_type: str = "pgd",
    model_save_dir: Path,
    training_epsilon: float = 0.0,
    tolerance: float | None = None,
):
    """
    Sweeps across all parameters, creating a table with all metrics for all data and returning it divided by group.

    model_save_dir is a checkpoint tree root from utils.paths; for fair-adversarial models
    that root is already tolerance-scoped. Tolerance is still needed here to key the attack cache.

    Returns: a dictionary with each table corresponding to 'combined', 'group_0', 'group_1'
    """
    full_df = pd.DataFrame()

    for architecture in architectures:
        for epsilon in epsilons:
            for seed in seeds:
                set_seed(seed)
                split = load_adult(random_state=seed)

                print(f"\n--- arch={architecture}  evaluation epsilon={epsilon} seed={seed}---")
                model = build_model(
                    architecture, split, hidden_dim=config["hidden_dim"], dropout=config["dropout"]
                )
                checkpoint = (
                    model_save_dir
                    / run_dir_name(architecture, training_epsilon, seed)
                    / "best_model.pt"
                )

                try:
                    model.load_state_dict(torch.load(checkpoint))
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"No trained static-adversarial model at {checkpoint}. "
                        f"Run train_static_adversarial.py first."
                    )

                results_df = evaluate_single(
                    model,
                    split,
                    architecture,
                    epsilon,
                    seed,
                    training_style,
                    attack_type=attack_type,
                    tolerance=tolerance,
                )
                full_df = pd.concat([full_df, results_df], axis=0)

    return split_by_group(full_df)


def summarize_evaluation_sweep(
    raw_tables: dict[str, pd.DataFrame], *, as_deltas: bool = False
) -> dict[str, pd.DataFrame]:
    """Summarize raw per-seed evaluation tables without changing their persisted schema.

    Each group is averaged over seeds, converted from decimal metrics to
    percentage-point scores, and optionally expressed as deltas from the
    architecture's epsilon-zero row. Tolerance sweeps are summarized separately
    per tolerance so distinct fairness constraints are never averaged together.
    """

    def summarize_table(table: pd.DataFrame) -> pd.DataFrame:
        summarized = to_scores(seed_averages_and_stddev(table))
        return values_as_deltas(summarized) if as_deltas else summarized

    summarized_tables = {}
    for name, table in raw_tables.items():
        if "tolerance" not in table.columns:
            summarized_tables[name] = summarize_table(table)
            continue

        tolerance_frames = []
        for tolerance in sorted(table["tolerance"].unique()):
            frame = summarize_table(
                table[table["tolerance"] == tolerance].drop(columns="tolerance")
            )
            frame.insert(0, "tolerance", tolerance)
            tolerance_frames.append(frame)
        summarized_tables[name] = pd.concat(tolerance_frames, ignore_index=True)

    return summarized_tables


def tolerance_sweep_evaluation(
    architectures: list[str],
    epsilons: list[float],
    seeds: list[int],
    tolerances: list[float],
    metric: str,
    training_style: str,
    *,
    training_epsilon: float = 0.0,
    attack_type="pgd",
) -> dict[str, pd.DataFrame]:
    """
    Runs evaluation_sweep once per tolerance and stacks the per-group tables, tagging every row with the tolerance that produced it.

    Takes the fairness metric rather than a checkpoint root, since each tolerance reads from a different one.

    Returns: a dictionary with each table corresponding to 'combined', 'group_0', 'group_1'
    """
    sweep_tables: dict[str, list[pd.DataFrame]] = {}

    for tolerance in tolerances:
        print(f"\n=== tolerance={tolerance} ===")
        group_tables = evaluation_sweep(
            architectures,
            epsilons,
            seeds,
            training_style,
            model_save_dir=fair_adversarial_root(metric, tolerance),
            training_epsilon=training_epsilon,
            tolerance=tolerance,
            attack_type=attack_type,
        )

        for name, table in group_tables.items():
            table.insert(0, "tolerance", tolerance)
            sweep_tables.setdefault(name, []).append(table)

    return {
        name: pd.concat(tables, axis=0).reset_index(drop=True)
        for name, tables in sweep_tables.items()
    }
