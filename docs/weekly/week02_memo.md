# Week 1 Summary

## Dates

- Start: Monday, 6/8
- End: Friday, 6/12

## Hours Completed

- Planned: 30
- Completed: 

## Assigned Readings

- Hardt et al. on Fairness

## Questions And Blockers

- 

## Code Or Test Changes

- ### tests/
  - *test_imports.py*:
    - Added monkeypatch tests to check general shape, continuous mask functionality, and 
    - correct sensitive/label attribute extraction.
  - *test_baseline_logistic.py*:
    - Added logistic regression function testing using fake adult income dataset. 
    - Checks shapes, predictions, training history output. 

- ### scripts/
  - *inspect_adult.py*:
    - Added script to manually test whether the real dataset import works
    - Added dummy classifiers and tested fairness metrics on them.
  - *run_baseline_logistic.py*: 
    - Add script to train and evaluate baseline logistic regression model.
    - Output confusion matrix, total accuracy, demographic parity difference, and equal opportunity difference.

- *Makefile*
  - Added ```make inspect-adult``` and ```make baseline-logistic```

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


## Commit Hashes



## Next Actions
