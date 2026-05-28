#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python - <<'PY'
from fair_safe_opt.config import load_config
from fair_safe_opt.data.synthetic import make_synthetic_tabular
from fair_safe_opt.evaluation import evaluate_binary_classifier
from fair_safe_opt.models import LogisticRegression
from fair_safe_opt.training import train_erm
from fair_safe_opt.utils import set_seed

cfg = load_config("configs/smoke/synthetic_logreg_erm.yaml")
set_seed(int(cfg["seed"]))
data_cfg = cfg["data"]
train_cfg = cfg["training"]
dataset = make_synthetic_tabular(
    n_samples=int(data_cfg["n_samples"]),
    n_continuous=int(data_cfg["n_continuous"]),
    n_categorical=int(data_cfg["n_categorical"]),
    seed=int(cfg["seed"]),
)
model = LogisticRegression(dataset.x.shape[1])
history = train_erm(
    model,
    dataset.x,
    dataset.y,
    epochs=int(train_cfg["epochs"]),
    batch_size=int(data_cfg["batch_size"]),
    learning_rate=float(train_cfg["learning_rate"]),
)
metrics = evaluate_binary_classifier(model, dataset.x, dataset.y, dataset.sensitive)
print({"final_loss": history.losses[-1], **metrics})
PY
