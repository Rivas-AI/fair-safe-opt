# Full Run Report

## Model Specifications
 - Architecture: small_mlp
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
| Accuracy | 0.8510 |
| ROC AUC | 0.9049
| Demographic Parity Difference | 0.1772 |
| Equal Opportunity Difference | 0.0553 |

## Confusion Matrix & ROC Curve

![Confusion Matrix](combined_roc_confusion.png)
### Group 0 (Men) Metrics
| Metric | Value |
|----------|----------|
| Accuracy | 0.8135 |
| ROC AUC | 0.8798
| Advantaged TPR | 0.6313

## Confusion Matrix & ROC Curve

![Confusion Matrix](group0_roc_confusion.png)
### Group 1 (Women) Metrics
| Metric | Value |
|----------|----------|
| Accuracy | 0.9258 |
| ROC AUC | 0.9341
| Advantaged TPR | 0.5759

## Confusion Matrix & ROC Curve

![Confusion Matrix](group1_roc_confusion.png)
