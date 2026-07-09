from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingHistory:
    train_losses: list[float]
    train_accuracies: list[float]
    val_losses: list[float]
    val_accuracies: list[float]