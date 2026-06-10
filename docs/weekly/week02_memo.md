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

- tests/test_imports.py:
  - Added monkeypatch tests to check general shape, continuous mask functionality, and 
  - correct sensitive/label attribute extraction.

- scripts/inspect_adult.py:
  - Added script to manually test whether the real dataset import works
  - Added dummy classifiers and tested fairness metrics on them.

- tests/test_fairness_metrics.py: 
  - Added small test which ended up being redundant. 

# Adult data exploration

from ```bash python scripts/inspect_adult.py ```

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
    - Equal Opportunity Surrogate: nan

Fairness Metrics for Unfair Dummy on Test Dataset: 
    - Demographic Parity Difference: 1.0
    - Equal Opportunity Difference: 1.0
    - Demographic Parity Surrogate: 1.0
    - Equal Opportunity Surrogate: nan

## Commit Hashes



## Next Actions
