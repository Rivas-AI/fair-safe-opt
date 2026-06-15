# Week 1 Summary

## Dates

- Start: Monday, 6/8
- End: Friday, 6/12

## Hours Completed

- Planned: 30
- Completed: 

## Assigned Readings

- Hardt et al. on Fairness

## Code Or Test Changes

- ### tests/
  - *test_imports.py*:
    - Added monkeypatch tests to check general shape, continuous mask functionality, and 
    - correct sensitive/label attribute extraction.
  - *test_baseline_logistic.py*:
    - Added logistic regression function testing using fake adult income dataset. 
    - Checks shapes, predictions, training history output. 
  - *test_baseline_mlp.py*:
    - Add mlp function testing using fake adult income dataset.
    - check shapes, predictions, proper history format.

- ### scripts/
  - *inspect_adult.py*:
    - Added script to manually test whether the real dataset import works
    - Added dummy classifiers and tested fairness metrics on them.
  - *run_baseline_logistic.py*: 
    - Add script to train and evaluate baseline logistic regression model.
    - Output confusion matrix, total accuracy, demographic parity difference, and equal opportunity difference.
  - *baseline_mlp.py*:
    - Add a script to train and evaluate the baseline small mlp model.
    - Output confusion matrix, total accuracy, demographic parity difference, and equal opportunity difference.

- *Makefile*
  - Added ```make inspect-adult```, ```make baseline-logistic```, ```make baseline-mlp```

# Adult data exploration

from ```make inspect-adult```

Num Features: 102

Num Training Samples: 27132

Training Set Stats:
 - Num Sensitive Group 0 (male): 18347  (~67%)
     - Positive Prediction Rate: 0.31362075543685614
 - Num Sensitive Group 1 (female): 8785  (~32%)
     - Positive Prediction Rate: 0.11041548093340922

Num Test Samples: 9045
Test Set Stats:
 - Num Sensitive Group 0 (male): 6026  (~66%)
     - Positive Prediction Rate: 0.31413873216063726
 - Num Sensitive Group 1 (female): 3019  (~33%)
     - Positive Prediction Rate: 0.1156011924478304

Num Val Samples: 9045
Val Set Stats:
 - Num Sensitive Group 0 (male): 6154  (~68%)
     - Positive Prediction Rate: 0.30744231394215144
 - Num Sensitive Group 1 (female): 2891  (~31%)
     - Positive Prediction Rate: 0.12106537530266344

Fairness Metrics for All Ones Dummy on Test Dataset: 
    - Demographic Parity Difference: 0.0
    - Equal Opportunity Difference: 0.0
    - Demographic Parity Surrogate: 0.0
    - Equal Opportunity Surrogate: 0.0

Fairness Metrics for Unfair Dummy on Test Dataset: 
    - Demographic Parity Difference: 1.0
    - Equal Opportunity Difference: 1.0
    - Demographic Parity Surrogate: 1.0
    - Equal Opportunity Surrogate: 1.0

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