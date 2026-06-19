# Week 1 Summary

## Dates

- Start: Monday, 6/15
- End: Friday, 6/19

## Hours Completed

- Planned: 40
- Completed: 40

## Code Or Test Changes

- ### tests/
  - *test_baseline_logistic.py, test_baseline_mlp.py, test_smoke_training.py*:
    - Updated tests to work with new validation set code.

- ### scripts/
  - *train_and_write_report.py*: 
    - Replaces the two baseline scripts. Alter parameters such as architecture, seed, training method, and then run with ```make report```. Will train a model of the specified architecture and generate a report with graphs and metrics in ```results/YYYY.MM.DD/{architecture}_seed{seed}_{method}_{HHMM}```

- ### src/fair_safe_opt
  - **training/**
    - *erm.py*
      -  Added validation loop and early stopping to training.
  - **evaluation/**
    - *evaluate.py*
      - Created a new datatype ```Evaluation``` to be returned, which includes:
        - Names of per-group and combined ROC and confusion matrix plots for easy retrieval.
        - Demographic Parity and Equal Opportunity differences.
        - Per-group and combined accuracy and AUC curve scores. 
  - **metrics/**
    - *fairness.py*
      - Added per-group accuracy function. 
  - **reporting/**
    - *report.py*
      - Created a function which takes in an ```Evaluation``` object and outputs a full run report .md file.

- ### paper/
  - *BaselineReport.pdf*
    - Added a report of baseline metrics for both mlp and logistic models on seeds 0, 1, and 2.

## Results Summary

This week, I've set up a reproducible model evaluation pipeline which takes in training parameters, trains a model, evaluates the model, and prints a report. I then ran this pipeline on each for seeds 0, 1, and 2 to establish baseline results for us to compare to moving forward. I've created a file detailing these results

## Commit Hashes

- 92ff066 Added epochs and patience to reporting
- **f3342a5 Updated model eval pipeline with heavy restructuring to disentangle**
- e5c1c08 Got evaluation code working and report format correct
- 35b59f4 Updated seed code
- f0ad914 Added artifact output to evaluation pipeline
- 84dd431 Ruff check and format
- 5a5afd2 Ruff check and format
- acfe93d Begin week 3 memo
- 142d358 Added tqdm progress bar to training
- 453ca25 Added more robust testing and implemented libraries into MLP script
- bb6ea34 Transferred evaluation code to a reusable library and updated script
- 8459107 Added per-group accuracy and output it in the model analysis
- 6f1deaa Finished week 2 memo
- dba822b Updated testing to match new validation-inclusive Adult split
- **77c7fe6 Added validation and early stopping**
- ef92454 Added CUDA and MPS functionality for long-term scalability
- a38e506 Updated tests
- 3540366 Updated prediction function name for clarity
- d8b109c Updated week 2 memo
- 76854de Updated week 2 memo

## Next Actions
