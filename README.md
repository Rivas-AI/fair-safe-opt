# fair-safe-opt

Summer 2026 research repository for a bounded pilot study on optimization
methods for fair and adversarially robust machine learning.

The initial scope is intentionally narrow:

- Dataset: Adult Income / UCI Adult via `sklearn.datasets.fetch_openml(data_id=1590, as_frame=True)`.
- Sensitive feature: `sex`.
- Target: binary income classification with positive class `>50K`.
- Models: PyTorch logistic regression and a small feed-forward neural network.
- Training modes: ERM, adversarial training, and fairness-aware adversarial training.
- Attacks: FGSM and PGD on standardized continuous features only.
- Metrics: accuracy, demographic parity difference, and equal opportunity difference.

This repository should not contain raw data, cached datasets, credentials,
trained model artifacts, or large generated results.

## Setup

Use a dedicated project environment. Do not install project dependencies into
Conda `base`.

```bash
conda create -n fair-safe-opt python=3.12 -y
conda run -n fair-safe-opt python -m pip install --upgrade pip
conda run -n fair-safe-opt python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
conda run -n fair-safe-opt python -m pip install -e ".[dev]"
```

The CPU path is the required baseline. CUDA is not assumed, and the project
does not configure CUDA, ROCm, or GPU-specific dependencies.

## Git Workflow

`main` is the normal working branch for this project. Jeremy and the PI may
commit directly to `main`.

Daily work sequence:

```bash
git switch main
git pull --ff-only origin main

conda activate fair-safe-opt

# edit files and run the task for the day

pytest -q
ruff check .
ruff format --check .

git add <specific-files>
git commit -m "Brief description of completed work"
git push origin main
```

Do not use `git add .` for ordinary weekly work; stage only intended files. Do
not commit data files, generated checkpoints, large result outputs, or
environment folders. Never force-push to `main`.

If `git pull --ff-only` fails, stop and ask the PI before changing history. Do
not use feature branches or pull-request requirements unless the PI later
changes this rule. Since two people may use `main`, only one person should edit
the same research module at a time unless they coordinate first. Major changes
to the dataset, sensitive attribute, attack model, fairness metric, or training
objective require a short decision record under `docs/decisions/`.

## Checks

```bash
make test
make smoke
```

Automated tests use synthetic data only and must work without network access.

## Adult Data

Adult is loaded only when explicitly requested by code that calls
`fair_safe_opt.data.adult.load_adult`. Tests and smoke checks avoid downloading
the dataset.

The current Adult loader is preliminary scaffolding for later review, not a
validated experiment pipeline.

## Preliminary Modules

The initialization scaffold includes small, testable code for synthetic data,
logistic regression, fairness metrics, continuous-feature attack masks, and ERM
smoke training. FGSM, PGD, the Adult loader, adversarial training,
fairness-penalized adversarial loss, the MLP, and evaluation helpers are present
as preliminary scaffolding unless and until later tasks add precise tests and
experiment documentation.

## Week 1 Start

Use the templates in `docs/weekly/`, `docs/meetings/`, and `docs/reading/` for
the first week. They are lightweight records for reading, definitions, blockers,
commits, and meeting decisions.

## Reproducibility Notes

Use fixed seeds in configs and scripts. Keep final tables and figures traceable
to CSV outputs, but do not commit large results or trained checkpoints.
