import pandas as pd

_ARCH_LABELS = {"logistic": "Logistic", "small_mlp": "Small MLP"}
_COL_HEADERS = ["bal_acc", "auc", "advantaged_tpr", "atk_succ"]


def _fmt_val(v):
    return f"{v * 100:.2f}"


def _fmt_delta(v):
    s = f"{v * 100:.2f}"
    return f"(+{s})" if v >= 0 else f"({s})"


def print_delta_table(baseline_rows, delta_rows, architectures, metrics):
    baseline_df = pd.DataFrame(baseline_rows).groupby("architecture")[metrics].mean()
    delta_df = (
        pd.DataFrame(delta_rows)
        .groupby(["architecture", "epsilon"])[metrics + ["atk_success"]]
        .mean()
        .reset_index()
    )

    col_w = 16
    header = f"{'Model':<14} {'Epsilon':>8}" + "".join(f"{h:>{col_w}}" for h in _COL_HEADERS)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for arch in architectures:
        b = baseline_df.loc[arch]
        vals = "".join(f"{_fmt_val(b[m]):>{col_w}}" for m in metrics)
        print(f"{_ARCH_LABELS[arch]:<14} {'0':>8}{vals}{' ' * col_w}")

        arch_deltas = delta_df[delta_df["architecture"] == arch].sort_values("epsilon")
        for _, row in arch_deltas.iterrows():
            vals = "".join(f"{_fmt_delta(row[m]):>{col_w}}" for m in metrics)
            atk = f"{_fmt_val(row['atk_success']):>{col_w}}"
            print(f"{'':14} {row['epsilon']:>8.2f}{vals}{atk}")

    print("=" * len(header))
