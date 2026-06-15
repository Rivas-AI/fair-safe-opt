from fair_safe_opt.models.logistic import LogisticRegression
from fair_safe_opt.data.adult import load_adult, AdultSplit
from fair_safe_opt.training.erm import train_erm, TrainingHistory
from fair_safe_opt.metrics.classification import accuracy_from_logits
from fair_safe_opt.metrics.fairness import demographic_parity_difference, equal_opportunity_difference
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(seed)
    
    
def plot_losses(history: TrainingHistory):
    plt.plot(history.train_losses, label="Training Loss")
    plt.plot(history.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    
def plot_accuracies(history: TrainingHistory):
    plt.plot(history.train_accuracies, label="Training Accuracy")
    plt.plot(history.val_accuracies, label="Validation Accuracy")
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
    seed = 1
    set_seed(seed)
    
    adult_split: AdultSplit = load_adult(random_state=seed)
    model = LogisticRegression(input_dim=adult_split.x_train.shape[1])
    history: TrainingHistory = train_erm(model, torch.tensor(adult_split.x_train), torch.tensor(adult_split.y_train), torch.tensor(adult_split.x_val), torch.tensor(adult_split.y_val), epochs=10)
    plot_losses(history)
    plot_accuracies(history)

    predictions = model.predict_classes(torch.tensor(adult_split.x_test))
    ConfusionMatrixDisplay.from_predictions(adult_split.y_test, predictions)
    plt.show()
    
    print_metrics(predictions, torch.tensor(adult_split.y_test), torch.tensor(adult_split.a_test))

if __name__ == "__main__":
    main()