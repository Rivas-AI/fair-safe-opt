from dataclasses import dataclass, field
import json
from dataclasses import asdict


@dataclass(frozen=True)
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    train_surrogates: list[float] = field(default_factory=list)
    val_surrogates: list[float] = field(default_factory=list)
    lambdas: list[float] = field(default_factory=list)
    penalties: list[float] = field(default_factory=list)


def save_training_history(history: TrainingHistory, path: str) -> None:
    with open(path, "w") as f:
        json.dump(asdict(history), f)


def load_training_history(path: str) -> TrainingHistory:
    with open(path) as f:
        return TrainingHistory(**json.load(f))
