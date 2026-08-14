# Full Run Report

## Model Specifications
 - Architecture: logistic
 - Seed: 0
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
| Accuracy | 0.8331 |
| ROC AUC | 0.8896
| Demographic Parity Difference | 0.1787 |
| Equal Opportunity Difference | 0.2147 |

## Confusion Matrix & ROC Curve

![Confusion Matrix](combined_roc_confusion.png)
### Group 0 (Men) Metrics
| Metric | Value |
|----------|----------|
| Accuracy | 0.7947 |
| ROC AUC | 0.8616
| Advantaged TPR | 0.5357

## Confusion Matrix & ROC Curve

![Confusion Matrix](group0_roc_confusion.png)
### Group 1 (Women) Metrics
| Metric | Value |
|----------|----------|
| Accuracy | 0.9096 |
| ROC AUC | 0.9213
| Advantaged TPR | 0.3209

## Confusion Matrix & ROC Curve

![Confusion Matrix](group1_roc_confusion.png)
