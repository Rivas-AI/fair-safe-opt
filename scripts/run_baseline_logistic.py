from fair_safe_opt.models.logistic import LogisticRegression
from fair_safe_opt.data.adult import load_adult, AdultSplit
from fair_safe_opt.training.erm import train_erm, TrainingHistory
from fair_safe_opt.metrics.classification import accuracy_from_logits
from fair_safe_opt.metrics.fairness import demographic_parity_difference, equal_opportunity_difference
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_training_history(history: TrainingHistory):
    plt.plot(history.losses, label="Loss")
    plt.plot(history.accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def print_metrics(preds: torch.tensor, y: torch.tensor, a: torch.tensor):
    accuracy_total = (preds == y).float().mean().item()
    print(f"\nTotal Test Accuracy: {accuracy_total:.4f}\n")

    print(f"Confusion Matrix:")
    print(confusion_matrix(y, preds))

    print(f"\nDemographic Parity Difference:")
    print(demographic_parity_difference(preds, a))

    print(f"\nEqual Opportunity Difference:")
    print(equal_opportunity_difference(preds, y, a))

def main():
    adult_split: AdultSplit = load_adult()
    model = LogisticRegression(input_dim=adult_split.x_train.shape[1])
    history: TrainingHistory = train_erm(model, torch.tensor(adult_split.x_train), torch.tensor(adult_split.y_train), epochs=10)
    # plot_training_history(history)

    predictions = model.predict_classes(torch.tensor(adult_split.x_test))
    ConfusionMatrixDisplay.from_predictions(adult_split.y_test, predictions)
    # plt.show()
    
    print_metrics(predictions, torch.tensor(adult_split.y_test), torch.tensor(adult_split.a_test))

if __name__ == "__main__":
    main()