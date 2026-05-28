# Repository Instructions

This repository supports a Summer 2026 master's-student research project on
fairness-aware adversarial training for tabular binary classification.

Do not add claims, benchmark numbers, or fabricated experiment outputs. Code,
tests, and documentation should support later experiments without implying that
results have already been run.

Keep scope bounded to Adult Income, `sex` as the primary sensitive attribute,
PyTorch logistic regression and small MLP models, FGSM/PGD attacks, and
demographic parity/equal opportunity metrics unless the PI approves changes.

Tests must run offline with synthetic data. Raw data, cached data, credentials,
trained models, and large results must not be committed.

## Git Workflow

`main` is the normal working branch. Jeremy and the PI may commit directly to
`main`.

Before starting a work session, run:

```bash
git switch main
git pull --ff-only origin main
```

After making a small coherent change, run the required tests, commit, and push
to `main`. Never force-push to `main`. Do not require feature branches or pull
requests unless the PI later changes this rule.

Major changes to the dataset, sensitive attribute, attack model, fairness
metric, or training objective require a short decision record under
`docs/decisions/`.
