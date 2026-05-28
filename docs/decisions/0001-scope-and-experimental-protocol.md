# Decision 0001: Scope and Experimental Protocol

## Status

Accepted for initial scaffold.

## Decision

The project is limited to a pilot study on Adult Income tabular binary
classification.

The dataset is UCI Adult, retrieved later through:

```python
sklearn.datasets.fetch_openml(data_id=1590, as_frame=True)
```

The target is income `>50K`. The primary sensitive attribute is `sex`.

`sex` is retained for group metric computation and fairness-surrogate
computation. The primary model-input protocol excludes `sex` as a predictive
feature. Any later ablation that includes `sex` as a predictive feature requires
PI approval and a new decision record.

Splits occur before fitting imputers, encoders, or scalers. Preprocessing is fit
on training data only and then applied to validation and test splits.

Required models are PyTorch logistic regression and a small PyTorch MLP.
Required attacks are FGSM and PGD. Attacks may perturb standardized continuous
predictive features only and must never alter one-hot categorical entries.
Required report metrics are clean accuracy, robust accuracy, demographic parity
difference, and equal opportunity difference.

Fairlearn may be used only as an evaluation cross-check for hard-label group
fairness metrics. Training-time fairness surrogates are implemented locally.

The selected group-disparity metrics are diagnostic measurements for the study.
They do not by themselves prove that a model is fair, appropriate for
deployment, or fit for use.

## Consequences

The scaffold prioritizes deterministic CPU smoke checks, synthetic offline
tests, explicit attack masks, and traceable experiment configuration. Broader
datasets, attacks, metrics, and deployment claims remain out of scope until PI
approval.

No second dataset, changed sensitive attribute, altered attack protocol, new
fairness metric, or changed training objective should be added without a PI
decision record.

The project uses a simple direct-to-`main` workflow. Jeremy and the PI may
commit approved work directly to `main`; feature branches and pull requests are
not required unless the PI later changes this rule. Contributors should update
from GitHub with `git pull --ff-only origin main` before starting work and must
never force-push to `main`.
