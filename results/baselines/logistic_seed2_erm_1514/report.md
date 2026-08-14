# Full Run Report

## Model Specifications
 - Architecture: logistic
 - Seed: 2
 - Method: erm
 - Epochs: 10
 - Patience: 5
 - Hidden Dimension: 64
 - Dropout: 0.0
 - Learning Rate: 0.0001

## Training History

![Training Curves](training_curves.png)

## Metrics

### Total Combined Metrics
| Metric | Value |
|----------|----------|
| Accuracy | 0.8364 |
| ROC AUC | 0.8968
| Demographic Parity Difference | 0.1837 |
| Equal Opportunity Difference | 0.2453 |

## Confusion Matrix & ROC Curve

![Confusion Matrix](combined_roc_confusion.png)
### Group 0 (Men) Metrics
| Metric | Value |
|----------|----------|
| Accuracy | 0.7982 |
| ROC AUC | 0.8715
| Advantaged TPR | 0.5323

## Confusion Matrix & ROC Curve

![Confusion Matrix](group0_roc_confusion.png)
### Group 1 (Women) Metrics
| Metric | Value |
|----------|----------|
| Accuracy | 0.9149 |
| ROC AUC | 0.9201
| Advantaged TPR | 0.2870

## Confusion Matrix & ROC Curve

![Confusion Matrix](group1_roc_confusion.png)
