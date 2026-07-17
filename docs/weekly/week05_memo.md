# Week Summary

## Dates

*Shifted due to vacation time*

## Hours Completed

- Planned: 40
- Completed: 40

## Code Or Test Changes

- ### Makefile
  - Updated build/run targets to reflect the new script structure (`Updated make`).
  - Deleted extra/obsolete scripts; consolidated early scripts and smoke checks.

- ### scripts/ and top-level training entry points
  - Created two independent scripts for training Madry-style and static-style adversarially trained models, replacing a single combined script.
  - `train_static_adversarial.py`: generates 10 random-start PGD copies per training/validation sample against a frozen baseline model, concatenates with clean data, and trains a fresh model via standard ERM.
  - Madry-style adaptive training script: generates PGD examples per-batch against the current model weights during training (true min-max adversarial training).

- ### src/fair_safe_opt/reporting/
  - Created `robustness_tables.py`, a shared library extracted from duplicated table-building code, containing `average_over_seeds`, `to_scores`, `split_by_group`, and `values_as_deltas`.
  - Moved table metric functions into this library so static, Madry, and baseline evaluations all use identical, tested table-construction logic.

- ### src/fair_safe_opt/evaluation/
  - Extracted robustness evaluation into a library (`evaluation_sweep`) with a `_TRAINING_STYLE` switch (`static` vs. `madry`) so one evaluation script drives both comparisons.
  - Added combined-group row to the TPR-by-group metric; added group column to output dataframes (previously an index) for cleaner downstream table code.

- ### src/fair_safe_opt/training/
  - Added `TrainingHistory` saving (`save_training_history` / load), and fixed file paths so history is written per (architecture, seed) run.
  - Extracted the `TrainingHistory` class into its own file, removing a duplicated definition.
  - Removed a confusing/unused attack-dispatch factory from the training folder; updated `run.py` accordingly.
  - Made hyperparameters keyword arguments throughout training calls for clarity at call sites.

- ### src/fair_safe_opt/attacks/
  - Added a random-start switch for PGD (`random_start` parameter), used by the static-training data generation pipeline.

- ### General cleanup
  - Removed an unused model, vestigial code, and stray print statements.
  - Renamed `save_dir` to `out_dir` for consistency across training functions.
  - Added tests.

## Results Summary

This week's main goal was the Week 5 deliverable: comparing standard ERM, static PGD-augmented training, and Madry-style adaptive adversarial training under identical PGD evaluation. Two real bugs were found and fixed along the way, and the headline result held up through both fixes.

**Bug 1 — undertrained models.** `TrainingHistory` inspection showed both architectures' train/val loss still dropping steeply at the `max_epochs` cap (10 epochs), with no sign of convergence. Raising the epoch budget was necessary before any robustness result from these models could be trusted.

**Bug 2 — stale attack cache.** The adversarial-example cache in the evaluation pipeline was keyed only by `architecture_seed_epsilon`, with no reference to which trained model generated the cached attack. This caused evaluation to silently reuse adversarial examples generated against a different model (e.g., an earlier eval run), producing artificially flat, unrealistic robustness numbers for whichever architecture had a stale file on disk. Fixed by clearing the cache and confirming the fix produces graded, monotonic degradation consistent with a real attack.

**Finding: static PGD augmentation catastrophically fails, independent of both bugs.** After both fixes, the static-trained models are still far more attackable than the undefended baseline (e.g., small MLP attack success reaches 93.61% at ε=0.3, vs. 18.01% for the baseline). Static-model clean accuracy also exceeds the dataset's known ~84–85% ceiling on the augmented val set, which points to label leaking: PGD's perturbation direction is approximately label-dependent, and with 10 of 11 training copies carrying that perturbation, the fresh model learns the shortcut instead of the underlying task. Because this persists after fixing undertraining and the caching artifact, I'm treating it as a genuine property of static augmentation rather than a remaining bug.

**Madry-style adaptive training works and is genuinely robust**, evaluated with the same fixed pipeline:

| Method    | Arch      | Clean Bal. Acc. | Bal. Acc. @ε=0.3 | Atk. Succ. @ε=0.3 | DPD @ε=0.3 | EOpp @ε=0.3 |
|-----------|-----------|-----------------:|-------------------:|--------------------:|-----------:|------------:|
| Baseline  | Logistic  | 72.14            | 62.81               | 9.23                | 19.23      | 19.50       |
| Baseline  | Small MLP | 78.10            | 63.50               | 18.01               | 25.52      | 8.37        |
| Static    | Logistic  | 69.56            | 29.58               | 46.80                | 9.97       | 2.48        |
| Static    | Small MLP | 78.61            | 4.45                | 93.61                | 17.10      | 1.17        |
| Madry     | Logistic  | 70.08            | 65.98               | 3.86                 | 16.97      | 20.81       |
| Madry     | Small MLP | 77.09             | 74.38               | 2.74                 | 19.58      | 11.70       |

Madry training substantially outperforms the baseline in both accuracy retention and attack resistance at every ε. However, the fairness picture is mixed: DPD improves relative to baseline, but EOpp gets worse, and per-group breakdowns show the robustness gain is not evenly distributed — Group 0 (male, the majority class) gets roughly 2–3x the accuracy/AUC/TPR improvement that Group 1 gets. Standard adversarial training appears to disproportionately protect whichever group was already more vulnerable at baseline, which happens to be the majority group here, widening the Equal Opportunity gap even as aggregate metrics improve.

## Commit Hashes

- 125d93c Added random start switch for PGD
- efdab03 Cleaned up training folder. Removed confusing factory. Extracted training history class into its own file to remove redundancy.
- d2ea41b Added code to print a table of metrics for each group
- d8bbc02 Updated run.py in accordance with removed factory
- 95ca9ee Removed pointless and confusing attack factory
- 25e71d8 Changed save_dir to out_dir for consistency.
- b9f52dd Made hyperparameters keyword arguments for clarity in calling.
- a85ee6a Removed print statement
- a01bbd6 Removed vestigial code.
- d177518 Removed unused model
- 49fed43 Added group column in dataframe instead of as index
- 0f1fb48 Added combined group to TPR_by_group metric
- 8a98e12 Added tests
- 666f42b Added baseline robustness evaluation code
- faaface Updated makefile
- 3f14e40 Deleted extra scripts
- 4b9f1f8 Moved table metric functions into their own library
- fe6fbb5 Consolidated early scripts and smoke checks
- **f6afb02 Extracted robustness eval into a library and added a switch for madry or static models.**
- **d02fc66 Created two independent scripts for training madry-style and static-style**
- 0f58c9f created robust evaluation library
- e0913af Added helper functions for making clean robustness tables
- c87fa88 Updated name
- d56f137 Added train history saving to examine training behavior
- a1f737d Added training history saving and proper filepaths.
- 57c3bd2 Updated make
- **71c788f Fixed caching bug**

## Next Actions

- Begin Week 6 scope (fairness-constrained/penalized adversarial training) immediately — the plan's calendar puts this two weeks behind due to time off, and it is the paper's core contribution.
- Decide whether to invest further time in static-training mitigation (e.g., generating perturbations from predicted rather than true labels) or document the label-leaking failure as a final negative result and move on; leaning toward the latter given schedule pressure.
- Consolidate the recurring `torch.load()` / state-dict checkpoint-loading pattern into a single `load_model()` helper in `models/build.py` to prevent the class of bug just fixed from recurring elsewhere.
- Confirm with Dr. Rivas whether the per-group Madry disparity (majority group favored) is worth a deeper look now or is better framed as motivation for the fairness-constrained method in Week 6/7.