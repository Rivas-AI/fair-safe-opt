from fair_safe_opt.data.synthetic import make_synthetic_tabular
from fair_safe_opt.evaluation import evaluate_binary_classifier
from fair_safe_opt.models import LogisticRegression, SmallMLP
from fair_safe_opt.training import train_erm
from fair_safe_opt.utils import set_seed


def test_logistic_smoke_training_runs() -> None:
    set_seed(11)
    dataset = make_synthetic_tabular(n_samples=64, seed=11)
    model = LogisticRegression(dataset.x.shape[1])

    history = train_erm(
        model,
        dataset.x,
        dataset.y,
        dataset.x_val,
        dataset.y_val,
        epochs=2,
        batch_size=16,
        learning_rate=0.05,
        show_progress=False,
    )
    metrics = evaluate_binary_classifier(model, dataset.x, dataset.y, dataset.sensitive_train)

    assert len(history.train_losses) == 2
    assert len(history.train_accuracies) == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["demographic_parity_difference"] <= 1.0


def test_mlp_forward_shape() -> None:
    dataset = make_synthetic_tabular(n_samples=8, seed=12)
    model = SmallMLP(dataset.x.shape[1], hidden_dim=8)

    logits = model(dataset.x)

    assert logits.shape == dataset.y.shape
