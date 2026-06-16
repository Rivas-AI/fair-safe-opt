# Week 1 Summary

## Dates

- Start: Monday, 6/15
- End: Friday, 6/20

## Hours Completed

- Planned: 40
- Completed: 

## Assigned Readings

## Code Or Test Changes

- ### tests/
  - *test_baseline_logistic.py, test_baseline_mlp.py, test_smoke_training.py*:
    - Updated tests to work with new validation set code.

- ### scripts/
  - *run_baseline_logistic.py & baseline_mlp.py*: 
    - Transfer evaluation code to library usage

- ### src/fair_safe_opt
  - **training/**
    - *erm.py*
      -  Added validation loop and early stopping to training.
  - **evaluation/**
    - *evaluate.py*
      - Moved set_seed, plot training history, and print metrics functions here for reproducibility. 
  - **metrics/**
    - *fairness.py*
      - Added per-group accuracy function. 

## Logistic Regression Baseline

from ```make baseline-logistic```

Total Test Accuracy: 0.8484

Confusion Matrix:
[[6272  531]
 [ 840 1402]]

Demographic Parity Difference:
0.18703436851501465

Equal Opportunity Difference:
0.003864884376525879


## MLP Baseline 

Total Test Accuracy: 0.8399

Confusion Matrix:
[[6288  515]
 [ 933 1309]]

Demographic Parity Difference:
0.14557848870754242

Equal Opportunity Difference:
0.09484440088272095

## Commit Hashes

- ef92454 Added cuda and mps functionalityfor long-term scalability
- a38e506 Updated tests
- 3540366 Updated prediction function name for clarity.
- d8b109c Updated week 2 memo
- 76854de Updated week 2 memo
- 30c4439 Added minimally tested baseline mlp code.
- 2922c25 Fixed make
- de48c04 Added error throwing for malformed input
- 6c51562 Added make baseline-mlp
- bc8b7f1 Updated testing. Added baseline mlp testing.
- 72a9d35 Updated week02mmemo with day's work and updated function names to enhance clarity.
- fb2aa1b Updated makefile for clarity. Fixed bug in inspect_adult causing nan behavior.
- ed04a8c Add script to run a simple logistic regression and print metrics.
- 10c1f2d added malformed input error throwing
- 0371842 Added test suite for logistic regression model.
- f4f0742 Corrected split-inspect filepath
- 0c246a3 Add logistic and split-inspect makes
- a0429da Added predict function

## Next Actions

- Add validation loop and early stopping.
- Add random seed
- Add better logging and checkpoints
- output per-group accuracy
- transfer model analysis into a library
- separate loss/accuracy graphs