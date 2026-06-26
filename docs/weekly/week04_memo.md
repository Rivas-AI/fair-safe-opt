# Week 4 Summary

## Dates

- Start: Monday, 6/22
- End: Friday, 6/26

## Hours Completed

- Planned: 40
- Completed: 40

## Code Or Test Changes

- ### configs/experiments/
  - *training_configs.yaml*:
    - Created a config file that centralizes all training and attack parameters (architecture, method, seed, epochs, patience, hidden_dim, dropout, learning_rate, attack type, epsilon, steps).
    - Updated throughout the week as experiments progressed; scripts now read directly from this file.

- ### Makefile
  - Replaced `baseline` and `attack` targets with `run` (runs `scripts/run.py`) and `play` (runs `scripts/baseline_v_attack.py`).

- ### scripts/
  - *run.py*:
    - Created a new unified run script that loads all parameters from the configs file and runs training, evaluation, and attack evaluation across multiple seeds, architectures, and epsilons.
  - *generate_attack_report.py*:
    - Deleted; functionality folded into `run.py`.
  - *train_and_generate_report.py*:
    - Deleted; functionality folded into `run.py`.

- ### src/fair_safe_opt
  - **attacks/**
    - *attack.py*
      - Created `attack_features` factory function that dispatches to FGSM or PGD based on the `attack_type` config parameter, or returns features unchanged for `'none'`.
  - **data/**
    - *adult.py*
      - Switched `AdultSplit` inputs to tensors, removing unnecessary conversion boilerplate from call sites.
  - **evaluation/**
    - *evaluate.py*
      - Replaced per-group and combined accuracy with balanced accuracy.
      - Streamlined existing evaluation code.
    - *plot.py*
      - Replaced accuracy labels with balanced accuracy.
      - Streamlined existing plot code.
  - **metrics/**
    - *attacks.py*
      - Created `attack_success_rate` and `attack_success_by_group` functions measuring the rate at which an adversarial attack flips a correct prediction to an incorrect one, broken down by demographic group.
    - *classification.py*
      - Replaced accuracy with balanced accuracy.
    - *fairness.py*
      - Replaced accuracy with balanced accuracy.
  - **models/**
    - *build.py*
      - Created `build_model` factory function that instantiates the correct model architecture (logistic or small MLP) based on config parameters.
    - *logistic.py, mlp.py*
      - Added method to return both predictions and probabilities directly from logits.
  - **reporting/**
    - *report.py*
      - Updated to report balanced accuracy in place of accuracy.
  - **training/**
    - *erm.py*
      - Extracted seeding logic into `utils/seeds.py`.
    - *train.py*
      - Added optional `device` parameter; updated to dispatch to the correct training method from config.
  - **utils/**
    - *devices.py*
      - Extracted `set_device` into its own module.
    - *seeds.py*
      - Created to hold `set_seed`; added `torch.use_deterministic_algorithms(True)` for stronger reproducibility guarantees.

## Results Summary

This week I implemented and validated the FGSM and PGD adversarial attacks for tabular data, restricting perturbations to the standardized continuous features and holding the one-hot categorical features fixed via a boolean mask. I confirmed attack correctness on three fronts: the epsilon-ball is enforced in clamp_continuous_features, gradients flow correctly to the input, and categorical entries are provably unchanged after attack. I also built per-group and combined attack-success metrics.


I then evaluated the baseline ERM logistic and small MLP models under FGSM across seeds 0, 1, and 2, sweeping epsilon to find a range that produces graded degradation in standardized (z-score) units rather than model collapse. All metric values below are seed-averaged and scaled by 100 (percentage points). The top row for each model is the clean baseline (epsilon = 0); subsequent rows report the change from that baseline.

```bash
=======================================================================
Model           Epsilon     bal_acc         auc         DPD         Eop
=======================================================================
Logistic              0       72.14       89.39       17.98       23.08
                   0.10     (-3.09)     (-3.79)     (+0.26)     (-0.52)
                   0.20     (-6.19)     (-8.33)     (+0.71)     (-2.34)
                   0.30     (-9.33)    (-13.52)     (+1.25)     (-3.58)
Small MLP             0       78.10       90.86       19.28        9.76
                   0.10     (-4.05)     (-4.47)     (+1.51)     (-0.88)
                   0.20     (-9.17)    (-10.16)     (+4.14)     (-0.62)
                   0.30    (-14.59)    (-17.04)     (+6.26)     (-1.39)
=======================================================================
```

Two trends stand out. First, the attack degrades both accuracy and AUC smoothly with epsilon, confirming a healthy graded attack rather than the all-or-nothing collapse seen when epsilon was initially set too high. Second, the small MLP is consistently more attackable than logistic regression — at epsilon = 0.30 it loses 14.59 balanced-accuracy points versus the logistic model's 9.33 — which is consistent with higher-capacity models having more exploitable curvature.


The fairness picture is the more interesting one and warrants caution before any strong claim. The demographic parity difference grows under attack for both models (positive deltas), and the effect is far larger for the MLP (+6.26 at epsilon = 0.30) than for logistic regression (+1.25). In other words, the attack widens the gap in positive-prediction rates between groups. Equal opportunity moves in the opposite direction (gaps narrow slightly), so the two fairness metrics disagree — a divergence worth investigating rather than reporting as a clean finding.

## Commit Hashes

- 2a2ee5b began utilizing separate configs file.
- **e870ddd updated run scripts**
- 377a555 Added attack success rate metrics.
- 7b69049 Extracted set_seed into its own function
- **907b5e9 Added factory for training and updated ERM.**
- 5b1d909 added method to return predictions and probabilities based on logits.
- c266363 Added balanced accuracy in place of accuracy.
- b3d27f9 updated script to use new libraries
- 0bd646d Switched AdultSplit inputs to tensors to reduce unnecessary boilerplate.
- 0758b9c Streamlined existing evaluation code
- 2182a84 Added helper functions to allow switching based on configs
- 462f753 update configs
- **38c5f66 Created config file for experiments**
- 0223584 Replaced accuracy with balanced accuracy
- e695fd6 Extracted seeding code into separate file.
- b40ba04 Added line to enhance reproducibility.

## Next Actions

- Confirm the DPD-widening-under-attack effect holds before treating it as a result: rerun with additional seeds and check whether the MLP's larger fairness degradation is stable or seed-driven.
- Investigate why DPD and EOpp move in opposite directions under attack — likely tied to the baseline class imbalance within the female group, where positive predictions are sparse.
- Run PGD across the same epsilon grid and confirm the expected sanity check (PGD robust accuracy ≤ FGSM robust accuracy); if they match, revisit alpha and step count.
- Add per-group breakdowns to the table (currently combined only) so the fairness trends can be read directly rather than inferred from the gap metrics.
- Confirm with Dr. Rivas the target epsilon range and comparison venue for standardized tabular epsilon, so the reported values align with the literature.

