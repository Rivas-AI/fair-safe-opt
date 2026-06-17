from fair_safe_opt.evaluation.evaluate import Evaluation
from pathlib import Path


def create_report(
    architecture: str, method: str, seed: int, eval: Evaluation, out_dir: Path
) -> None:
    output_file = out_dir / "report.md"
    with open(output_file, "w") as f:
        print("# Full Run Report", file=f)
        print("", file=f)

        print("## Model Specifications", file=f)
        print(f" - Architecture: {architecture}", file=f)
        print(f" - Seed: {seed}", file=f)
        print(f" - Method: {method}", file=f)
        print("", file=f)

        print("## Training History", file=f)
        print("", file=f)
        print(f"![Training Curves]({eval.hist_plot_name})", file=f)
        print("", file=f)

        # Combined Metrics
        print("## Metrics", file=f)
        print("", file=f)
        print("### Total Combined Metrics", file=f)
        print("| Metric | Value |", file=f)
        print("|----------|----------|", file=f)
        print(f"| Accuracy | {eval.total_acc:.4f} |", file=f)
        print(f"| ROC AUC | {eval.total_auc:.4f}", file=f)
        print(f"| Demographic Parity Difference | {eval.dem_parity_diff:.4f} |", file=f)
        print(f"| Equal Opportunity Difference | {eval.equal_opp_diff:.4f} |", file=f)
        print("", file=f)

        print("## Confusion Matrix & ROC Curve", file=f)
        print("", file=f)
        print(f"![Confusion Matrix]({eval.total_roc_conf_name})", file=f)

        # Group 0 (Men) Metrics
        print("### Group 0 (Men) Metrics", file=f)
        print("| Metric | Value |", file=f)
        print("|----------|----------|", file=f)
        print(f"| Accuracy | {eval.group0_acc:.4f} |", file=f)
        print(f"| ROC AUC | {eval.group0_auc:.4f}", file=f)
        print("", file=f)

        print("## Confusion Matrix & ROC Curve", file=f)
        print("", file=f)
        print(f"![Confusion Matrix]({eval.group0_roc_conf_name})", file=f)

        # Group 1 (Women) Metrics
        print("### Group 1 (Women) Metrics", file=f)
        print("| Metric | Value |", file=f)
        print("|----------|----------|", file=f)
        print(f"| Accuracy | {eval.group1_acc:.4f} |", file=f)
        print(f"| ROC AUC | {eval.group1_auc:.4f}", file=f)
        print("", file=f)

        print("## Confusion Matrix & ROC Curve", file=f)
        print("", file=f)
        print(f"![Confusion Matrix]({eval.group1_roc_conf_name})", file=f)
