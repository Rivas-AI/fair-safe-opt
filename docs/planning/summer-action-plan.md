# 10-Week Summer Research Action Plan

**Project:** Optimization Methods for Fair and Adversarially Robust Machine Learning  
**Student:** Jeremy Frank  
**Faculty sponsor:** Dr. Pablo Rivas  
**Period:** June 1, 2026 to August 7, 2026  
**Workload:** 300 total hours, planned as 10 weeks × 30 hours/week  
**End goal:** preliminary results, reproducible code, and a paper draft

---

## 1. Purpose of this plan

This plan is a concrete weekly work plan. The main purpose is to produce usable preliminary results for a future NSF resubmission while also giving him a clear, bounded student research experience.

The project will focus on the classical fairness and robustness problem. The main research product will be a small but well-tested study of fairness-constrained adversarial training for tabular classification.

Each week has a concrete result, a strict definition of done, and two short meetings with Dr. Rivas.

---

## 2. Project thesis

The summer paper should make one careful claim:

> Fairness constraints can be added to adversarial training for tabular binary classification using practical constrained or penalized optimization methods, and doing so lets us measure the tradeoff among clean accuracy, robust accuracy, and group fairness gaps.

The project should not claim that it solves fair robust learning. The stronger and safer claim is that it gives a clear pilot framework, shows early empirical tradeoffs, and creates evidence that the larger NSF project is concrete and testable.

---

## 3. Required final products

By Friday, August 7, 2026, you should deliver the following:

1. **Reproducible GitHub repository**
   - Clean source code.
   - Clear README.
   - Fixed environment file.
   - Scripts to run all final experiments.
   - Results stored in a clear folder structure.
   - No hidden local-only files needed to reproduce the paper tables.

2. **Workshop paper draft**
   - Target length: 4 to 6 pages, plus references and appendix if allowed.
   - Format: LaTeX.
   - Draft should be ready for your revision, not just notes.
   - The paper should include at least one full experimental table and one tradeoff plot.

3. **Preliminary NSF evidence package**
   - One-page summary for proposal reuse.
   - Main figures in PDF and PNG.
   - Main tables in CSV and LaTeX table format.
   - Short statement of what the results prove and what they do not prove.

4. **Final technical memo**
   - 3 to 5 pages.
   - Plain account of method, experiments, problems, and next steps.
   - Written so that anyone can reproduce the research results.

---

## 4. Non-negotiable scope limits

These limits protect the project from becoming too broad.

### Required scope

- Primary dataset: **Adult Income**.
- Primary protected attribute: **sex**.
- Optional protected attribute: **race**, only if the primary pipeline is stable.
- Required model 1: **logistic regression** in PyTorch.
- Required model 2: **small feed-forward neural network**.
- Required attacks: **FGSM** and **PGD** on continuous tabular features only.
- Required fairness metrics:
  - Demographic parity difference.
  - Equal opportunity difference.
- Required training methods:
  - Standard empirical risk minimization.
  - Adversarial training.
  - Fairness-constrained or fairness-penalized adversarial training.

### Optional scope

- Second dataset: **COMPAS**.
- Optional fairness metric: equalized odds difference.
- Optional solver variant: augmented Lagrangian.
- Optional result: Pareto-style plot across fairness tolerance values.

### Out of scope

- Quantum fairness.
- Multimodal datasets.
- Image datasets.
- Large pretrained models.
- A full interior-point method implementation.
- Three or more datasets.
- A new fairness metric.
- A new adversarial attack.
- Claims about real-world deployment.

---

## 5. Research questions for the workshop paper

The paper should answer these questions:

**RQ1.** Does adversarial training change group fairness gaps compared with standard training on Adult Income?

**RQ2.** Can a fairness-constrained or fairness-penalized adversarial training method reduce demographic parity or equal opportunity gaps under clean and adversarial evaluation?

**RQ3.** What is the cost, in clean accuracy and robust accuracy, when the fairness tolerance is tightened?

These questions are specific enough for a workshop paper and map directly to the NSF repair goal: define the fairness goal, define the robustness setting, define the method, and evaluate progress.

---

## 6. Technical formulation

### 6.1 Data and notation

Let the data be

\[
D = \{(x_i, y_i, a_i)\}_{i=1}^n,
\]

where:

- \(x_i\) is the feature vector.
- \(y_i \in \{0,1\}\) is the class label.
- \(a_i \in \{0,1\}\) is the protected group label.
- \(f_\theta(x)\) is a binary classifier returning a logit.
- \(p_\theta(x) = \sigma(f_\theta(x))\) is the predicted probability for class 1.

For Adult Income:

- \(y=1\): income greater than 50K.
- Primary \(a\): sex, encoded as binary according to the dataset field.
- Protected attribute handling must be stated clearly: either included as a feature, excluded as a feature, or tested both ways. The required baseline should exclude it from the model input but keep it for fairness metrics. A secondary ablation can include it.

### 6.2 Adversarial perturbation model

Since Adult is tabular data with categorical and continuous features, adversarial attacks must not perturb one-hot categorical entries. The required perturbation set is:

\[
\Delta = \{\delta : \|\delta_S\|_\infty \le \epsilon,\; \delta_C = 0\},
\]

where:

- \(S\) is the set of standardized continuous feature indices.
- \(C\) is the set of categorical feature indices.
- Perturbed continuous features are clipped to valid standardized train-set ranges.
- Categorical features are held fixed.

This avoids the common mistake of creating invalid tabular samples by changing one-hot categories with gradient noise.

### 6.3 Robust loss

For each training example, adversarial training uses:

\[
\max_{\delta \in \Delta} \ell(f_\theta(x_i + \delta), y_i).
\]

FGSM uses one step:

\[
\delta = \epsilon \cdot \mathrm{sign}(\nabla_x \ell(f_\theta(x),y)).
\]

PGD uses repeated projected steps:

\[
\delta_{t+1} = \Pi_\Delta\left(\delta_t + \alpha \cdot \mathrm{sign}(\nabla_x \ell(f_\theta(x+\delta_t),y))\right).
\]

### 6.4 Fairness metrics for training

The project should use probability-based smooth surrogates during training and hard-label metrics for final evaluation.

**Demographic parity surrogate:**

\[
G_{DP}(\theta) =
\left(\mathbb{E}[p_\theta(x) \mid a=0] -
\mathbb{E}[p_\theta(x) \mid a=1]\right)^2.
\]

**Equal opportunity surrogate:**

\[
G_{EOpp}(\theta) =
\left(\mathbb{E}[p_\theta(x) \mid a=0, y=1] -
\mathbb{E}[p_\theta(x) \mid a=1, y=1]\right)^2.
\]

The final report should also compute hard-label versions:

- Demographic parity difference:
  \(|P(\hat{Y}=1\mid A=0)-P(\hat{Y}=1\mid A=1)|\)
- Equal opportunity difference:
  \(|TPR_{A=0}-TPR_{A=1}|\)

### 6.5 Main constrained objective

The clean target formulation is:

\[
\min_\theta \frac{1}{n}\sum_{i=1}^{n}
\max_{\delta_i \in \Delta}
\ell(f_\theta(x_i+\delta_i),y_i)
\quad
\text{subject to}\quad
G_j(\theta) \le \tau_j,
\]

where \(G_j\) is a fairness gap surrogate and \(\tau_j\) is a chosen tolerance.

### 6.6 Practical solver

The required implementation should use a practical Lagrangian method:

\[
\mathcal{L}(\theta,\lambda) =
L_{adv}(\theta) + \sum_j \lambda_j(G_j(\theta)-\tau_j)
+ \frac{\rho}{2}\sum_j \max(0,G_j(\theta)-\tau_j)^2.
\]

Training loop:

1. Generate adversarial examples for the current batch.
2. Compute adversarial cross-entropy loss.
3. Compute fairness surrogate on the batch.
4. Update \(\theta\) using Adam.
5. At the end of each epoch, compute full-train fairness gap.
6. Update \(\lambda_j \leftarrow \max(0, \lambda_j + \eta_\lambda(G_j-\tau_j))\).
7. Log loss, fairness gap, clean accuracy, and robust accuracy.

Required fallback if this is unstable:

\[
L(\theta)=L_{adv}(\theta)+\lambda G_j(\theta)
\]

with a fixed sweep over \(\lambda \in \{0.1,1,10\}\). This fallback is still useful for a workshop paper if it is reported honestly as a penalty-based approximation.

---

## 7. Books and readings: bounded use

The books are useful, but you should not read them front to back.

### Numerical Optimization by Nocedal and Wright

Use as the main math reference.

Read only selected material on:

- Optimality conditions.
- Constrained optimization.
- Penalty methods.
- Augmented Lagrangian methods.
- Practical stopping criteria.

Expected output: short notes connecting KKT conditions and Lagrangian relaxation to the project objective.

### Linear Programming by Dantzig

Use only for intuition.

Read selected material on:

- Constraints.
- Slack variables.
- Dual variables.
- Feasibility.

Expected output: half-page explanation of why fairness can be seen as a constraint rather than only a regularizer.

### Primal-Dual Interior-Point Methods by Wright

Use as optional support.

Read selected introductory material on:

- Barrier ideas.
- Primal-dual updates.
- Complementarity.

Expected output: a short note explaining why a full interior-point method is not the implementation target for this summer.

---

## 8. Standard weekly meeting structure

There will be two short meetings every week.

### Meeting 1: Monday 11:00–11:20 AM

Purpose: set the week’s work and remove ambiguity before you code.

Jeremy must send by Monday 9:00 AM:

- A 5-bullet plan for the week.
- A list of files he plans to modify.
- One blocker or risk, if any.
- The previous week’s unfinished items, if any.

Meeting agenda:

1. Confirm this week’s definition of done.
2. Confirm what is required and what is optional.
3. Confirm code branch name and output folder.
4. Identify one risk to reduce before Thursday.

### Meeting 2: Thursday 3:00–3:20 PM

Purpose: inspect the actual artifact before the week ends.

Jeremy must send by Thursday 12:00 PM:

- GitHub link or commit hash.
- Results file path.
- Short memo path.
- One screenshot, table, or terminal output proving progress.

Meeting agenda:

1. Demo code or results.
2. Check if the week’s deliverable passes the done test.
3. Decide whether Friday is for cleanup, repair, or extension.
4. Assign final Friday push.

### Friday rule

Every Friday by 5:00 PM, Jeremy must post:

- A weekly memo in `reports/weekXX_memo.md`.
- Updated code in GitHub.
- Any result files in `outputs/weekXX/`.
- A short issue list for the next week.

---

## 9. Repository structure

Use this structure from Week 1.

```text
fair-robust-constraints/
  README.md
  requirements.txt
  environment.yml
  .gitignore
  configs/
    adult_erm_logreg.yaml
    adult_erm_mlp.yaml
    adult_adv_mlp.yaml
    adult_fair_adv_dp.yaml
    adult_fair_adv_eopp.yaml
  data/
    raw/                 # not committed if license or size is an issue
    processed/           # only committed if allowed
  src/
    __init__.py
    data.py
    models.py
    fairness.py
    attacks.py
    losses.py
    train.py
    evaluate.py
    utils.py
  scripts/
    run_week05_baselines.sh
    run_week06_adversarial.sh
    run_week08_final_sweep.sh
    make_tables.py
    make_plots.py
  outputs/
    week01/
    week02/
    week03/
    week04/
    week05/
    week06/
    week07/
    week08/
    week09/
    week10/
  reports/
    week01_memo.md
    week02_memo.md
    week03_memo.md
    week04_memo.md
    week05_memo.md
    week06_memo.md
    week07_memo.md
    week08_memo.md
    week09_memo.md
    week10_final_memo.md
  paper/
    main.tex
    refs.bib
    sections/
      intro.tex
      related_work.tex
      method.tex
      experiments.tex
      results.tex
      limitations.tex
  tests/
    test_fairness.py
    test_attacks.py
    test_data.py
```

---

## 10. Weekly plan

## Week 1: June 1–5 — Scope lock, repo setup, and reading map

### Main goal

Set up the project so that there is no confusion about scope, tools, datasets, or final outputs.

### Target by Friday

A working repository, a one-page project contract, a reading matrix, and a paper skeleton.

### Work plan, 30 hours

**Reading, 6 hours**

Read and summarize:

- Goodfellow et al. on FGSM.
- Madry et al. on PGD adversarial training.
- Hardt et al. on equality of opportunity.
- Agarwal et al. on reductions for fair classification.
- Xu et al. or a similar paper on robustness and fairness tradeoffs.

Jeremy should not write long summaries. For each paper, he should produce:

- Problem.
- Method.
- Fairness or robustness metric.
- What can be reused.
- What does not fit this summer.

**Coding, 8 hours**

- Create GitHub repository.
- Create Python environment.
- Add README with project goal.
- Add repo folders listed above.
- Create empty modules in `src/`.
- Add a simple `pytest` test that runs.
- Add `paper/main.tex` with section placeholders.

**Writing, 6 hours**

Create:

- `reports/week01_memo.md`.
- `reports/project_contract.md`.
- `paper/sections/intro.tex` with 3 to 5 rough paragraphs.

The project contract must state:

- Required dataset.
- Required metrics.
- Required models.
- Required attacks.
- Required methods.
- Items that are out of scope.

**Planning and meetings, 2 hours**

- Monday meeting: confirm scope and repo setup.
- Thursday meeting: repo demo and reading matrix check.

**Buffer and cleanup, 8 hours**

Use for environment problems, GitHub issues, and setting up data access.

### Deliverables

- GitHub repo link.
- `reports/project_contract.md`.
- `reports/week01_memo.md`.
- `reports/reading_matrix.csv`.
- `paper/main.tex` and section files.

### Done test

The week is done only if Dr. Rivas can clone the repo, install the environment, run one test, and see the paper skeleton.

---

## Week 2: June 8–12 — Dataset pipeline and fairness metrics

### Main goal

Build a clean Adult Income data pipeline and implement fairness metrics correctly.

### Target by Friday

A processed Adult dataset, documented protected attribute handling, and tested fairness metric functions.

### Work plan, 30 hours

**Reading, 4 hours**

Read selected material on:

- Demographic parity.
- Equal opportunity.
- Equalized odds, as optional background.
- Train/test leakage in ML preprocessing.

Optimization reading:

- Nocedal and Wright: selected sections on constrained problems and optimality conditions.

**Coding, 14 hours**

Implement `src/data.py`:

- Download or load Adult.
- Clean missing values.
- Define label.
- Define protected attribute.
- Split train/validation/test.
- Fit preprocessing on train only.
- Standardize continuous features.
- One-hot encode categorical features.
- Track continuous feature indices and categorical feature indices.
- Save processed tensors or arrays.

Implement `src/fairness.py`:

- Hard demographic parity difference.
- Hard equal opportunity difference.
- Smooth demographic parity surrogate.
- Smooth equal opportunity surrogate.
- Group counts and warning if a group is too small.

Write tests:

- Fairness metrics return zero on balanced toy data.
- Fairness metrics return expected nonzero value on toy data.
- No division by zero when one group has no positive examples; function should return `nan` with warning or skip cleanly.

**Writing, 5 hours**

Write `reports/week02_memo.md`:

- Dataset fields used.
- Protected attribute used.
- Preprocessing choices.
- Fairness metric definitions.
- Known limits.

Update `paper/sections/experiments.tex` with dataset description.

**Meetings, 2 hours**

- Monday meeting: approve protected attribute handling.
- Thursday meeting: inspect dataset stats and fairness metric tests.

**Buffer, 5 hours**

Use for data loading problems and test fixes.

### Deliverables

- `src/data.py`.
- `src/fairness.py`.
- `tests/test_fairness.py`.
- `tests/test_data.py`.
- `outputs/week02/adult_dataset_summary.csv`.
- `reports/week02_memo.md`.

### Done test

The week is done only if Jeremy can run one command that prints:

- Train/validation/test sizes.
- Group counts.
- Positive label rate by group.
- Baseline fairness gaps from a dummy classifier.

---

## Week 3: June 15–19 — Baseline models and clean evaluation

### Main goal

Train standard non-adversarial models and create clean reference points.

### Target by Friday

A reproducible ERM baseline for logistic regression and a small MLP.

### Work plan, 30 hours

**Reading, 3 hours**

Read selected material on:

- Logistic regression as cross-entropy minimization.
- Basic PyTorch training loop structure.
- Model evaluation protocol for fairness studies.

Optimization reading:

- Nocedal and Wright: selected material on gradient methods and stopping criteria.

**Coding, 15 hours**

Implement `src/models.py`:

- Logistic regression.
- Small MLP: input layer, one hidden layer, ReLU, dropout optional, output logit.

Implement `src/train.py`:

- Config loading.
- Seed setting.
- Training loop.
- Validation loop.
- Checkpoint saving.
- CSV logging.

Implement `src/evaluate.py`:

- Accuracy.
- Balanced accuracy, optional.
- AUC, optional.
- DP difference.
- EOpp difference.
- Per-group accuracy.

Run experiments:

- Logistic regression, seeds 0, 1, 2.
- MLP, seeds 0, 1, 2.

**Writing, 5 hours**

Write:

- `reports/week03_memo.md` with clean baseline results.
- First draft of `paper/sections/method.tex` model subsection.
- First draft of `paper/sections/results.tex` baseline subsection.

**Meetings, 2 hours**

- Monday meeting: confirm model simplicity and expected logs.
- Thursday meeting: inspect first baseline table.

**Buffer, 5 hours**

Use for training bugs, seed variance, and config cleanup.

### Deliverables

- `src/models.py`.
- `src/train.py`.
- `src/evaluate.py`.
- `configs/adult_erm_logreg.yaml`.
- `configs/adult_erm_mlp.yaml`.
- `outputs/week03/baseline_results.csv`.
- `reports/week03_memo.md`.

### Done test

The week is done only if the repo can reproduce a table with clean accuracy, DP difference, and EOpp difference for both models across three seeds.

### Sanity checks

- If Adult clean accuracy is below 0.75 for the MLP, debug preprocessing or labels.
- If fairness gaps are exactly zero for all trained models, debug metric code.
- If train accuracy is high but test accuracy is poor, check leakage, scaling, or split errors.

---

## Week 4: June 22–26 — FGSM and PGD attacks for tabular data

### Main goal

Implement valid adversarial attacks for tabular continuous features and test them.

### Target by Friday

FGSM and PGD code that attacks continuous features only, with validation tests and robust accuracy results for ERM models.

### Work plan, 30 hours

**Reading, 4 hours**

Read:

- FGSM paper sections on attack definition.
- PGD adversarial training paper sections on first-order adversary.
- One short source on adversarial attacks for tabular data, focusing on why categorical features should not be perturbed directly.

Optimization reading:

- Dantzig selected material on constraints and slack variables, only enough to connect constraints to fairness tolerance.

**Coding, 15 hours**

Implement `src/attacks.py`:

- `fgsm_attack(model, x, y, epsilon, continuous_idx, bounds)`.
- `pgd_attack(model, x, y, epsilon, alpha, steps, continuous_idx, bounds)`.
- Clipping to continuous feature bounds.
- No changes to categorical features.
- Optional random PGD start.

Add tests:

- Categorical columns unchanged after attack.
- Continuous perturbation max norm is within epsilon.
- Attack increases or does not reduce loss on a fixed batch.
- Perturbed samples remain within stored bounds.

Run experiments:

- Evaluate Week 3 ERM models under FGSM.
- Evaluate Week 3 ERM models under PGD.
- Try epsilon values `{0.01, 0.03, 0.05}` in standardized continuous space.

**Writing, 4 hours**

Write:

- `reports/week04_memo.md`.
- Attack subsection in `paper/sections/method.tex`.
- Experimental setup text explaining tabular attack limits.

**Meetings, 2 hours**

- Monday meeting: confirm attack validity rules.
- Thursday meeting: demo attack tests and robust accuracy table.

**Buffer, 5 hours**

Use for gradient bugs and bounds handling.

### Deliverables

- `src/attacks.py`.
- `tests/test_attacks.py`.
- `outputs/week04/erm_under_attack.csv`.
- `reports/week04_memo.md`.

### Done test

The week is done only if categorical entries are proven unchanged under attack and PGD robust accuracy is reported for the Week 3 MLP.

### Sanity checks

- Robust accuracy should usually be less than or close to clean accuracy. If robust accuracy is higher, check attack code.
- If attack has no effect at all, check gradients, feature scaling, and epsilon.
- If attack destroys every prediction at epsilon 0.01, check scaling and clipping.

---

## Week 5: June 29–July 3 — Adversarial training baseline

### Main goal

Train models with adversarial training but without fairness constraints.

### Target by Friday

A standard adversarial training baseline with clean accuracy, robust accuracy, and fairness gaps.

### Work plan, 30 hours

**Reading, 3 hours**

Read:

- Madry-style adversarial training method.
- One paper or section on robustness-accuracy tradeoff.
- One short source on robust fairness tradeoffs.

**Coding, 15 hours**

Modify `src/train.py`:

- Add adversarial training option.
- Generate PGD examples during training.
- Log clean and adversarial losses.
- Save configs and checkpoints.

Run experiments:

- MLP standard ERM, seeds 0, 1, 2.
- MLP adversarial training with selected epsilon, seeds 0, 1, 2.
- Optional logistic adversarial training if time allows.

Evaluation:

- Clean test accuracy.
- FGSM robust accuracy.
- PGD robust accuracy.
- DP difference on clean test set.
- EOpp difference on clean test set.
- DP and EOpp on PGD-attacked test set if stable.

**Writing, 5 hours**

Write:

- `reports/week05_memo.md`.
- First result table comparing ERM and adversarial training.
- Paper paragraph answering RQ1 in early form.

**Meetings, 2 hours**

- Monday meeting: confirm epsilon choice from Week 4.
- Thursday meeting: inspect adversarial training table and decide if Week 6 can start constrained method.

**Buffer, 5 hours**

Use for slow training, bad epsilon choice, or unstable model.

### Deliverables

- `configs/adult_adv_mlp.yaml`.
- `outputs/week05/adv_training_results.csv`.
- `reports/week05_memo.md`.
- Updated `paper/sections/results.tex`.

### Done test

The week is done only if there is a clean comparison between ERM and adversarial training across three seeds.

### Decision gate

If adversarial training is not working by Friday:

- Freeze model design.
- Use a smaller PGD step count.
- Use one epsilon value only.
- Do not add a second dataset.

---

## Week 6: July 6–10 — Fairness-constrained training, version 1

### Main goal

Implement the first fair adversarial training method using demographic parity.

### Target by Friday

A working constrained or penalty-based fair adversarial training run for demographic parity.

### Work plan, 30 hours

**Reading, 4 hours**

Read selected material on:

- Lagrangian relaxation.
- Penalty methods.
- Augmented Lagrangian idea.
- Agarwal reductions at a high level, focusing on constraints rather than full implementation.

Use Nocedal and Wright as the main reference. Wright’s primal-dual book is optional this week only if the implementation is already stable.

**Coding, 15 hours**

Implement `src/losses.py` or extend `src/train.py`:

- Fairness surrogate computation inside training.
- DP surrogate constraint.
- Lagrangian term.
- Nonnegative lambda update after each epoch.
- CSV logging for lambda, constraint value, tau, clean loss, adversarial loss.

Run small experiments:

- One seed only at first.
- One tau value first.
- Short epoch count first.
- Then full run if stable.

Required fallback:

If lambda updates are unstable, implement fixed penalty:

- `loss = adv_loss + lambda_dp * dp_surrogate`.
- Sweep `lambda_dp = {0.1, 1, 10}`.

**Writing, 4 hours**

Write:

- `reports/week06_memo.md`.
- Method subsection for fairness constraint.
- Note explaining whether the result is true constrained training or penalty approximation.

**Meetings, 2 hours**

- Monday meeting: confirm DP as first constraint and choose tau.
- Thursday meeting: inspect logs for lambda, DP surrogate, and model accuracy.

**Buffer, 5 hours**

Use for numerical issues and metric bugs.

### Deliverables

- `configs/adult_fair_adv_dp.yaml`.
- `outputs/week06/dp_fair_adv_debug.csv`.
- `outputs/week06/lambda_logs.csv`.
- `reports/week06_memo.md`.

### Done test

The week is done only if at least one fair adversarial training run completes and logs both fairness and accuracy through training.

### Sanity checks

- If lambda grows without bound, tau may be too strict or fairness estimate may be noisy.
- If fairness improves but accuracy collapses, reduce lambda or loosen tau.
- If fairness does not change at all, check whether the fairness loss has gradients.
- If batches lack both groups, use stratified batches or compute fairness on larger batches.

---

## Week 7: July 13–17 — Fairness-constrained training, version 2 and equal opportunity

### Main goal

Refine the fair adversarial method and add equal opportunity if demographic parity is stable.

### Target by Friday

A fair adversarial training comparison for DP and, if feasible, EOpp.

### Work plan, 30 hours

**Reading, 3 hours**

Read:

- Hardt et al. equality of opportunity sections.
- Short notes on threshold-dependent versus probability-based fairness metrics.
- Nocedal and Wright selected material on penalty parameter behavior.

**Coding, 15 hours**

Tasks:

- Clean up DP training code.
- Add EOpp surrogate.
- Add config switch: `fairness_metric: dp | eopp`.
- Add tolerance sweep: tau values such as `{0.01, 0.025, 0.05}` for squared or unsquared form, clearly documented.
- Add final evaluation script that computes hard-label metrics.

Run experiments:

- DP fair adversarial training, seeds 0, 1, 2.
- EOpp fair adversarial training, one seed first.
- If EOpp is stable, run seeds 0, 1, 2.

**Writing, 5 hours**

Write:

- `reports/week07_memo.md`.
- Methods subsection contrasting DP and EOpp.
- Results notes for tradeoff plot.

**Meetings, 2 hours**

- Monday meeting: decide whether EOpp is required or optional based on Week 6.
- Thursday meeting: choose final experiments for Week 8.

**Buffer, 5 hours**

Use for EOpp instability and tolerance sweeps.

### Deliverables

- `configs/adult_fair_adv_eopp.yaml`.
- `outputs/week07/fair_adv_dp_results.csv`.
- `outputs/week07/fair_adv_eopp_results.csv`, if stable.
- `reports/week07_memo.md`.

### Done test

The week is done only if there is a clear answer to this question: does the fairness-aware training reduce the selected fairness gap compared with adversarial training alone?

### Decision gate

If EOpp is unstable or slow:

- Drop EOpp as a training constraint.
- Keep EOpp as an evaluation metric.
- Focus the paper on DP-constrained adversarial training.

This is acceptable for a workshop paper.

---

## Week 8: July 20–24 — Final experiment sweep and tables

### Main goal

Run final experiments and produce the paper’s main result table and main plot.

### Target by Friday

Final Adult results across methods, metrics, and seeds.

### Work plan, 30 hours

**Reading, 2 hours**

Read only as needed for writing. No new major technical reading unless required to explain a result.

**Coding and experiments, 17 hours**

Run final required experiment matrix:

| Method | Model | Dataset | Seeds | Metrics |
|---|---|---|---:|---|
| ERM | MLP | Adult | 0,1,2 | clean acc, robust acc, DP, EOpp |
| PGD adversarial training | MLP | Adult | 0,1,2 | clean acc, robust acc, DP, EOpp |
| DP fair adversarial training | MLP | Adult | 0,1,2 | clean acc, robust acc, DP, EOpp |
| EOpp fair adversarial training | MLP | Adult | 0,1,2 if stable | clean acc, robust acc, DP, EOpp |

Generate:

- Main result table.
- Tradeoff plot: fairness gap vs clean accuracy.
- Tradeoff plot: fairness gap vs robust accuracy.
- Optional plot: tau or lambda vs fairness gap.

Optional second dataset:

Only start COMPAS if all Adult final runs are complete by Wednesday noon. If started, run only:

- ERM MLP.
- PGD adversarial MLP.
- DP fair adversarial MLP.
- One or three seeds depending on time.

**Writing, 6 hours**

Write:

- `paper/sections/results.tex` with final Adult results.
- `paper/sections/limitations.tex` first draft.
- `reports/week08_memo.md`.

**Meetings, 2 hours**

- Monday meeting: lock final experiment matrix.
- Thursday meeting: review tables and choose paper claim.

**Buffer, 3 hours**

Use for reruns and plot cleanup.

### Deliverables

- `outputs/week08/final_adult_results.csv`.
- `outputs/week08/main_results_table.tex`.
- `outputs/week08/fairness_accuracy_tradeoff.pdf`.
- `outputs/week08/fairness_robustness_tradeoff.pdf`.
- `reports/week08_memo.md`.

### Done test

The week is done only if the paper has one complete table and one complete plot generated from scripts, not manually typed.

### Main paper claim decision

By Friday, decide one of these claims:

1. Strong claim: fairness-aware adversarial training reduces DP gap with modest cost in robust accuracy.
2. Mixed claim: fairness-aware adversarial training can reduce DP gap, but the cost varies by tolerance and seed.
3. Negative but useful claim: fairness constraints are unstable under adversarial training in this setup, showing the need for better constrained optimization methods.

All three can support a paper if written honestly.

---

## Week 9: July 27–31 — Workshop paper full draft

### Main goal

Turn the code and results into a complete workshop paper draft.

### Target by Friday

A full paper draft with no missing major section.

### Work plan, 30 hours

**Writing, 16 hours**

Complete the paper:

- Abstract.
- Introduction.
- Related work.
- Problem setup.
- Method.
- Experiments.
- Results.
- Limitations.
- Conclusion.

Recommended title:

**Fairness-Constrained Adversarial Training for Tabular Classification: A Pilot Study**

Paper structure:

1. Introduction
   - Motivation.
   - Why fairness and robustness must be tested together.
   - What this pilot does.
   - Contributions.

2. Related Work
   - Fairness in classification.
   - Adversarial training.
   - Fairness and robustness tradeoffs.
   - Constrained optimization for fair ML.

3. Problem Setup
   - Dataset.
   - Protected attribute.
   - Perturbation set.
   - Fairness metrics.

4. Method
   - ERM.
   - PGD adversarial training.
   - Fairness-constrained or penalty-based adversarial training.

5. Experiments
   - Dataset and preprocessing.
   - Models.
   - Attack settings.
   - Metrics.
   - Seeds.

6. Results
   - Main table.
   - Tradeoff plot.
   - RQ1, RQ2, RQ3 answers.

7. Limitations
   - Tabular attacks are restricted to continuous features.
   - Adult is only one dataset unless COMPAS is added.
   - Constraint estimates can be noisy.
   - This is a pilot, not a full theory of fair robustness.

8. Conclusion
   - What was learned.
   - Why it supports the larger NSF proposal.

**Coding, 6 hours**

- Make scripts deterministic.
- Add `make_tables.py`.
- Add `make_plots.py`.
- Add README section: reproduce paper results.
- Remove unused code.

**Validation, 4 hours**

- Run final scripts from a clean terminal.
- Confirm table values match paper.
- Confirm figures match paper.
- Confirm random seeds are written in configs.

**Meetings, 2 hours**

- Monday meeting: paper outline approval.
- Thursday meeting: full draft review and claim check.

**Buffer, 2 hours**

Use for figure quality and missing references.

### Deliverables

- `paper/main.tex` complete draft.
- `paper/refs.bib`.
- `outputs/week09/paper_draft.pdf`.
- `reports/week09_memo.md`.
- Updated README reproduction section.

### Done test

The week is done only if Dr. Rivas can read a full PDF draft with all main tables and figures included.

---

## Week 10: August 3–7 — Final package, NSF evidence memo, and handoff

### Main goal

Finish the summer with a usable research package, not scattered files.

### Target by Friday

A workshop paper draft, reproducible repo, final memo, and NSF-ready preliminary-results summary.

### Work plan, 30 hours

**Writing, 12 hours**

Finish:

- Workshop paper draft.
- Final technical memo.
- One-page NSF preliminary-results summary.
- README.

The NSF summary should include:

- Research question.
- Dataset and method.
- Main table.
- Main plot.
- Key finding.
- Why this addresses prior critique.
- What the next NSF-scale work should do.

**Coding and cleanup, 8 hours**

- Clean configs.
- Remove broken notebooks.
- Move all final outputs to `outputs/final/`.
- Add a command list for reproduction.
- Run tests.
- Tag repo version: `summer-2026-final`.

**Reproduction check, 4 hours**

Jeremy should do a fresh run:

1. Clone repo into a new folder.
2. Create environment.
3. Run tests.
4. Run one short experiment.
5. Run final table script from stored final CSVs.
6. Build paper PDF.

**Meetings, 2 hours**

- Monday meeting: final package checklist.
- Thursday meeting: final handoff review.

**Buffer, 4 hours**

Use for final corrections.

### Deliverables

- `outputs/final/final_adult_results.csv`.
- `outputs/final/main_results_table.tex`.
- `outputs/final/main_figures/`.
- `paper/final_draft.pdf`.
- `reports/week10_final_memo.md`.
- `reports/nsf_preliminary_results_summary.md`.
- Tagged GitHub repo.

### Done test

The week is done only if the final package can be handed to another student or reviewer and they can understand what was done, what worked, what failed, and how to rerun the key results.

---

## 11. Weekly memo template

Jeremy should use this exact template each Friday.

```markdown
# Week XX Memo

## 1. Work completed
- 
- 
- 

## 2. Main result this week
One paragraph.

## 3. Files changed
- 
- 
- 

## 4. Experiments run
| Experiment | Config | Seed | Status | Output path |
|---|---|---:|---|---|

## 5. Key numbers
| Method | Clean Acc | PGD Robust Acc | DP Gap | EOpp Gap |
|---|---:|---:|---:|---:|

## 6. What failed or did not work
- 

## 7. What I need from Dr. Rivas
- 

## 8. Plan for next week
- 
- 
- 
```

---

## 12. Technical validation checklist

Jeremy should check these before claiming any result.

### Data validation

- Train/validation/test split is fixed and saved.
- Preprocessing is fit on train only.
- Label definition is correct.
- Protected attribute is not accidentally overwritten.
- Group counts are printed and saved.
- Continuous and categorical feature indices are saved.

### Attack validation

- Categorical features are unchanged by FGSM and PGD.
- Continuous features stay within epsilon.
- Continuous features stay within valid bounds.
- Loss on attacked batch is usually higher than clean loss.
- PGD attack is at least as strong as FGSM in most cases.

### Fairness validation

- Fairness functions pass toy tests.
- Group denominators are nonzero.
- Probability-based training metric and hard-label evaluation metric are both reported.
- DP and EOpp are not mixed.
- All metrics are computed on the same split.

### Training validation

- Random seeds are fixed.
- Config files are saved.
- Model checkpoints are saved only when useful.
- Logs include epoch, loss, metric, lambda, tau, and seed.
- Final table uses test data only once per final run.

### Paper validation

- Every table comes from a script.
- Every figure comes from a script.
- Every numeric claim can be traced to a CSV.
- Limitations are stated clearly.
- No claim says the method is general beyond the tested setting.

---

## 13. Risk plan

### Risk 1: Jeremy spends too much time reading

Control:

- Reading is capped each week.
- Every reading task must produce a small artifact.
- No week is allowed to have reading as the main deliverable after Week 2.

### Risk 2: adversarial attacks are invalid for tabular data

Control:

- Perturb continuous features only.
- Freeze categorical one-hot features.
- Report this clearly as a design choice and limitation.

### Risk 3: constrained training is unstable

Control:

- Start with DP only.
- Use full-train fairness estimate for lambda update after each epoch.
- Keep fixed-penalty fallback.
- Write honestly if the method is penalty-based rather than truly constrained.

### Risk 4: results are negative

Control:

Negative results can still support a paper if the experiment is clean. The paper can argue that simple fairness constraints do not reliably improve fairness under adversarial training, which motivates more serious constrained methods in the NSF resubmission.

### Risk 5: student work is hard to assess

Control:

- Every week has a GitHub commit.
- Every week has a memo.
- Every meeting inspects an artifact.
- Claims are not accepted without code or output files.

---

## 14. Grading rubric for weekly progress

Use this simple score each Friday.

| Score | Meaning | Action |
|---:|---|---|
| 0 | No usable artifact | Reduce scope immediately |
| 1 | Artifact exists but does not run or is not clear | Friday repair required |
| 2 | Artifact runs but lacks tests or explanation | Fix before next week |
| 3 | Artifact runs, is documented, and supports next step | Good progress |
| 4 | Artifact runs, is tested, documented, and paper-ready | Excellent progress |

Minimum acceptable score each week: **3**.

If any week scores below 3, the next week starts with repair, not new scope.

---

## 15. Expected final paper contribution

The final workshop paper should claim the following type of contribution:

1. A practical formulation of fairness-constrained adversarial training for tabular binary classification.
2. A reproducible implementation using Adult Income with valid tabular perturbation limits.
3. An empirical study comparing ERM, PGD adversarial training, and fairness-aware adversarial training.
4. Evidence about the clean accuracy, robust accuracy, and fairness tradeoff.

This is concrete and useful. It directly supports a stronger future NSF proposal with a tested protocol, working code, and early results.

---

## 16. Final success criteria

The summer is successful if all five conditions hold:

1. The repo reproduces the final table and figure.
2. The paper draft has a clear method and at least one real result table.
3. The method is stated honestly as constrained, augmented Lagrangian, or penalty-based.
4. The limits of tabular adversarial perturbations are clear.
5. The NSF summary can be inserted into a future proposal as preliminary evidence.

The summer is not successful if Jeremy only produces reading notes, partial code, or unverified results.
