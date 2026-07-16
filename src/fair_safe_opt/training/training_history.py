from dataclasses import dataclass
import json
from dataclasses import asdict

@dataclass(frozen=True)
class TrainingHistory:
    train_losses: list[float]
    train_accuracies: list[float]
    val_losses: list[float]
    val_accuracies: list[float]


def save_training_history(history: TrainingHistory, path: str) -> None:
    with open(path, "w") as f:
        json.dump(asdict(history), f)


def load_training_history(path: str) -> TrainingHistory:
    with open(path) as f:
        return TrainingHistory(**json.load(f))