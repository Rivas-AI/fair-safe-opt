# Full Run Report

## Model Specifications
 - Architecture: logistic
 - Seed: 1
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
| Accuracy | 0.8322 |
| ROC AUC | 0.8953
| Demographic Parity Difference | 0.1770 |
| Equal Opportunity Difference | 0.2323 |

## Confusion Matrix & ROC Curve

![Confusion Matrix](combined_roc_confusion.png)
### Group 0 (Men) Metrics
| Metric | Value |
|----------|----------|
| Accuracy | 0.7962 |
| ROC AUC | 0.8729
| Advantaged TPR | 0.5326

## Confusion Matrix & ROC Curve

![Confusion Matrix](group0_roc_confusion.png)
### Group 1 (Women) Metrics
| Metric | Value |
|----------|----------|
| Accuracy | 0.9056 |
| ROC AUC | 0.9180
| Advantaged TPR | 0.3003

## Confusion Matrix & ROC Curve

![Confusion Matrix](group1_roc_confusion.png)
