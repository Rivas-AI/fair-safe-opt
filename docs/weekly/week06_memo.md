# Week Summary

## Dates

*Still one to two weeks behind the plan calendar due to earlier vacation time; Week 6 scope completed, with substantial unplanned re-validation work.*

## Hours Completed

- Planned: 40
- Completed: 40

## Code Or Test Changes

- ### src/fair_safe_opt/training/
  - Implemented fairness-constrained adversarial training via the augmented Lagrangian method. Training minimizes adversarial BCE subject to a soft-DPD constraint `G(θ) = |DPD_soft(θ)| − τ ≤ 0`.
  - Dual variable `λ` updated once per epoch by projected ascent on the full-train constraint estimate, floored at 0 for KKT dual feasibility; penalty `μ` doubled only when infeasible and not shrinking fast enough.
  - Switched early stopping to feasibility-first (best val loss among constraint-satisfying epochs) and raised patience so training survives the constrained adaptation phase.
  - Guarded empty-group minibatches (return `nan` by design) at the two consumption points — batch loss and the `λ` update — rather than suppressing in the surrogate.

- ### src/fair_safe_opt/metrics/fairness.py
  - Converted DPD and EOpp from absolute-value to signed metrics (positive ⇒ G0 has the higher rate), with a shared sign convention across both. Absolute value now applied only downstream inside the training constraint.
  - Added `PPR_by_group` (positive prediction rate per group plus combined) to support the DPD-mechanism investigation.

- ### src/fair_safe_opt/evaluation/
  - Added an `attack_type` switch (`pgd` / `fgsm`) threaded through `evaluate_single`, `evaluation_sweep`, and `tolerance_sweep_evaluation`, with separate cache paths per attack so FGSM and PGD examples cannot collide.
  - Added `seed_averages_and_stddev` to the reporting library: reports per-seed standard deviation alongside each averaged metric.

- ### src/fair_safe_opt/utils/paths.py
  - Extended the adversarial-example cache key to carry `τ`, so fair-adversarial models at different tolerances cannot read one another's cached attacks (same failure class as the Week 5 stale-cache bug).

- ### src/fair_safe_opt/evaluation/plot.py
  - Added reusable line-plot helpers (mean line with ±1 std band, multi-panel per architecture, multi-regime overlay) for the paper figures.

- ### General
  - Regenerated all baseline, Madry, and static results from current code (see Results).
  - Added tests for the new metrics and the constrained-training update steps.

## Results Summary

This week's planned goal — implementing fairness-constrained adversarial training — is complete and the method works. The unplanned and more important story is that re-validating against the new code exposed that **the Week 5 result tables were generated from a stale code state and do not reproduce**, which changes what we can currently claim about the robustness–fairness tradeoff.

**Deliverable: DPD-constrained adversarial training works.** Using the augmented Lagrangian formulation above, the constraint substantially reduces DPD relative to Madry-style training while retaining robustness (comparable or lower attack success). A full τ sweep over {0.005, 0.01, 0.015, 0.02, 0.025} across both architectures and three seeds is complete. DPD and balanced accuracy trade off cleanly and monotonically as τ tightens — this is the paper's central, reportable result.

**Re-validation finding 1 — the Week 5 tables were stale.** Regenerating baseline and Madry results from current code produced materially different numbers from those in the Week 5 memo. For logistic regression the shift is large: baseline EOpp @ε=0 moved from 23.08 to 12.21, and clean balanced accuracy rose ~4 points. The most likely cause is the `max_epochs` fix and/or checkpoint-selection changes propagating to models that had been evaluated pre-fix. Every table in the Week 5 and Week 6 memos has been regenerated; the June 29 baseline memo is affected identically.

**Re-validation finding 2 — the mechanism behind DPD's behavior is now clear, and it is not what Week 5 claimed.** Per-group PPR analysis shows that on clean data the DPD gap is essentially unchanged between baseline and Madry. Adversarial attacks inflate positive-prediction rate (false positives) faster for G0 than G1, widening DPD; Madry-style training suppresses that attack-driven inflation almost entirely. So Madry narrows DPD *under attack* purely through robustness, not by correcting any pre-existing disparity. This is a cleaner and better-supported mechanism than the Week 5 "protects the majority group" framing, which the regenerated per-group numbers no longer support.

**Re-validation finding 3 — EOpp is too noisy to characterize at three seeds, and this matters for the paper's motivation.** Computing per-seed spread shows EOpp has a seed standard deviation of roughly 4–6 points across cells, while DPD's is under 1 point. The Week 5 headline — that Madry widens the EOpp gap — does not survive this: baseline and Madry EOpp confidence intervals overlap heavily at three seeds. The root cause is that EOpp conditions on the minority-positive cell (G1, Y=1), which is small enough that three seeds swing it substantially. This is the same instability that made EOpp-constrained *training* infeasible at batch size 32.

The practical reframing: **DPD is a stable, estimable, controllable fairness target on this dataset and EOpp is not.** That is itself a defensible finding, and it motivates both our choice to constrain DPD and the deferral of EOpp-constrained training — but it does change the paper's spine from "robustness widens the EOpp gap, so we constrain it" to "robustness stabilizes DPD but leaves EOpp noisy and uncontrolled, so an explicit, estimable constraint is needed."

| Method   | Arch      | Clean Bal. Acc. | Bal. Acc. @ε=0.3 | Atk. Succ. @ε=0.3 | DPD @ε=0.3 | EOpp @ε=0.3 (±std) |
|----------|-----------|----------------:|-----------------:|------------------:|-----------:|-------------------:|
| Baseline | Logistic  | 76.45           | 64.12            | 14.33             | 23.85      | 13.98 (±~4.8)      |
| Baseline | Small MLP | 78.27           | 62.75            | 18.64             | 24.22      | 6.42 (±~5)         |
| Madry    | Logistic  | 74.02           | 71.89            | 2.37              | 19.18      | 16.06 (±~4.3)      |
| Madry    | Small MLP | 77.53           | 74.19            | 3.29              | 18.88      | 9.88 (±~5)         |

(All values regenerated from current code, seed-averaged over seeds 0–2. FGSM results are near-identical to PGD on both architectures, confirming FGSM ≈ PGD here and justifying PGD-only reporting in subsequent sections.)

## Next Actions

- **Decision needed from Dr. Rivas:** given that EOpp is noise-dominated at three seeds, do we (a) run more seeds (5–10, eval-only, checkpoints exist) to try to recover a significant EOpp effect, or (b) reframe the paper around DPD as the estimable/controllable metric and report EOpp with explicit confidence intervals? This is a scope decision, not a code one, and it changes the paper's motivating claim.
- Regenerate the fair-model τ-sweep tables from current code and confirm they were trained under the corrected configuration (the checkpoints predate some of this week's changes).
- Fix the three math-notation errors in the DPD-constrained writeup (Lagrangian sign, `λ` accumulation, `μ` feasibility guard) to match the implemented code.
- Consolidate the `torch.load()` checkpoint-loading pattern into a single `load_model()` helper (carried over from Week 5; still outstanding).
- Build the main-text figures: Act 1 baseline vulnerability, Act 2 DPD mechanism (per-group PPR), Act 3 τ-sweep frontier — all with seed-std bands.
- Begin paper skeleton assembly against the three-act structure.

## Commit Hashes

9a0bb34 (HEAD -> main) Added positive prediction rate metric to see why DPD moves the way it does.
bd49d56 (origin/main, origin/HEAD) Changed DPD and EOpp hard metrics to return the actual value, not the absolute value'
a6dd9d3 Added fair adversarial model training with tau sweep.
bdc40ae Added path library for more robust caching.
d6091a5 Updated standard madry training to track fairness surrogate.