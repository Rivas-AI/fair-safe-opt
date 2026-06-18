import torch

from fair_safe_opt.utils.directories import create_report_dir
from fair_safe_opt.utils.seeds import set_seed

from fair_safe_opt.data.adult import load_adult

from fair_safe_opt.models.logistic import LogisticRegression
from fair_safe_opt.models.mlp import SmallMLP, OptimizedMLP

from fair_safe_opt.training.erm import train_erm, TrainingHistory

from fair_safe_opt.evaluation.evaluate import evaluate_model, Evaluation
from fair_safe_opt.reporting.report import create_report

ARCHITECTURES = ["logistic", "small_mlp"]  # Valid: {'logistic', 'small_mlp', 'optimized_mlp'}
METHOD = "erm"  # Valid: {'erm'}
SEEDS = [0, 1, 2]

EPOCHS = 10
PATIENCE = 5
HIDDEN_DIM = 64
DROPOUT = 0.0
LEARNING_RATE = 1e-4


def run(ARCHITECTURE, SEED):
    set_seed(SEED)
    out_dir = create_report_dir(ARCHITECTURE, METHOD, SEED)

    split = load_adult(random_state=SEED)

    # Define model architecture
    if ARCHITECTURE == "logistic":
        model = LogisticRegression(split.x_train.shape[1])
    elif ARCHITECTURE == "small_mlp":
        model = SmallMLP(split.x_train.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    elif ARCHITECTURE == 'optimized_mlp':
        model = OptimizedMLP(split.x_train.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    else:
        raise ValueError("architecture not in {'logistic', 'small_mlp'}")

    # Train the model using the specified method
    if METHOD == "erm":
        history: TrainingHistory = train_erm(
            model,
            torch.tensor(split.x_train),
            torch.tensor(split.y_train),
            torch.tensor(split.x_val),
            torch.tensor(split.y_val),
            out_dir,
            epochs=EPOCHS,
            patience=PATIENCE,
            learning_rate=LEARNING_RATE
        )
    else:
        raise ValueError("method not in {'erm'}")

    eval: Evaluation = evaluate_model(model, split, history, out_dir)

    create_report(ARCHITECTURE, METHOD, SEED, EPOCHS, PATIENCE, HIDDEN_DIM, DROPOUT, LEARNING_RATE, eval, out_dir)


def main():
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            print(f"\n=== {architecture} | seed {seed} ===")
            run(architecture, seed)


if __name__ == "__main__":
    main()
