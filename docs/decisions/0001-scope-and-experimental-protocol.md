# Decision 0001: Scope and Experimental Protocol

## Status

Accepted for initial scaffold.

## Decision

The project is limited to a pilot study on Adult Income tabular binary
classification using `sex` as the primary sensitive attribute. Required models
are PyTorch logistic regression and a small PyTorch MLP. Required attacks are
FGSM and PGD, restricted to standardized continuous features only. Required
fairness metrics are demographic parity difference and equal opportunity
difference.

Fairlearn may be used only as an evaluation cross-check for hard-label group
fairness metrics. Training-time fairness surrogates are implemented locally.

## Consequences

The scaffold prioritizes deterministic CPU smoke checks, synthetic offline
tests, explicit attack masks, and traceable experiment configuration. Broader
datasets, attacks, metrics, and deployment claims remain out of scope until PI
approval.

The project uses a simple direct-to-`main` workflow. Jeremy and the PI may
commit approved work directly to `main`; feature branches and pull requests are
not required unless the PI later changes this rule. Contributors should update
from GitHub with `git pull --ff-only origin main` before starting work and must
never force-push to `main`.
