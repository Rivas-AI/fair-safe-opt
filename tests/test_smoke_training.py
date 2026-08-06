import torch

from fair_safe_opt.data.synthetic import make_synthetic_tabular
from fair_safe_opt.evaluation import evaluate_binary_classifier
from fair_safe_opt.models import LogisticRegression, SmallMLP
from fair_safe_opt.training import train_erm
from fair_safe_opt.training.adversarial import train_pgd_madry
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
    )
    metrics = evaluate_binary_classifier(model, dataset.x, dataset.y, dataset.sensitive_train)

    assert len(history.train_losses) == 2
    assert len(history.train_accuracies) == 2
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["demographic_parity_difference"] <= 1.0


def test_mlp_forward_shape() -> None:
    dataset = make_synthetic_tabular(n_samples=8, seed=12)
    model = SmallMLP(dataset.x.shape[1], hidden_dim=8)

    logits = model(dataset.x)

    assert logits.shape == dataset.y.shape


def test_adversarial_smoke_training_runs() -> None:
    set_seed(21)
    dataset = make_synthetic_tabular(n_samples=32, seed=21)
    model = LogisticRegression(dataset.x.shape[1])

    history = train_pgd_madry(
        model,
        dataset.x,
        dataset.y,
        dataset.sensitive_train,
        dataset.x_val,
        dataset.y_val,
        dataset.sensitive_val,
        dataset.continuous_mask,
        epochs=2,
        batch_size=16,
        learning_rate=0.05,
        epsilon=0.05,
        alpha=0.02,
        steps=2,
    )

    assert len(history.train_losses) == 2
    assert len(history.train_accuracies) == 2
    assert len(history.val_losses) == 2
    assert len(history.val_accuracies) == 2
    assert all(torch.isfinite(torch.tensor(history.train_losses)))
    assert all(torch.isfinite(torch.tensor(history.val_losses)))
    assert all(0.0 <= acc <= 1.0 for acc in history.train_accuracies)
    assert all(0.0 <= acc <= 1.0 for acc in history.val_accuracies)

    # Madry training tracks the fairness surrogate; these are the fields whose
    # arrival broke the call above.
    assert len(history.train_surrogates) == 2
    assert len(history.val_surrogates) == 2
    assert all(torch.isfinite(torch.tensor(history.train_surrogates)))
    assert all(torch.isfinite(torch.tensor(history.val_surrogates)))
