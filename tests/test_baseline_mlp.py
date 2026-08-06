from fair_safe_opt.models.mlp import SmallMLP
from fair_safe_opt.training.erm import train_erm
import torch
import pytest
from test_imports import fake_fetch_openml
from fair_safe_opt.data.adult import load_adult, AdultSplit


def test_leq_zero_input_dimension() -> None:
    """Test that the model raises an error for non-positive input dimensions."""
    with pytest.raises(ValueError, match="Input dimension must be positive."):
        SmallMLP(input_dim=0)

    with pytest.raises(ValueError, match="Input dimension must be positive."):
        SmallMLP(input_dim=-5)


def test_malformed_input() -> None:
    """Test that the model raises an error for malformed input."""
    model = SmallMLP(input_dim=5)
    with pytest.raises(ValueError, match="Input feature dimension mismatch."):
        model.preds_probs(torch.tensor([[1, 2], [3, 4]]))  # Wrong input shape


def test_forward_logits() -> None:
    """Test that forward returns one logit per row."""
    model = SmallMLP(input_dim=2, hidden_dim=2)

    with torch.no_grad():
        model.net[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.net[0].bias.zero_()
        model.net[3].weight.copy_(torch.tensor([[1.0, -1.0]]))
        model.net[3].bias.zero_()

    x = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])

    logits = model(x)

    assert logits.shape == torch.Size([2])
    assert torch.allclose(logits, torch.tensor([1.0, -3.0]))


def test_predictions(monkeypatch) -> None:
    """Test that the model makes predictions with the correct shape and values."""
    monkeypatch.setattr("fair_safe_opt.data.adult.fetch_openml", fake_fetch_openml)
    adult_split: AdultSplit = load_adult()

    model = SmallMLP(input_dim=adult_split.x_train.shape[1])
    preds, probs = model.preds_probs(adult_split.x_test)
    assert preds.shape == adult_split.y_test.shape
    assert probs.shape == adult_split.y_test.shape
    assert set(preds.tolist()).issubset({0.0, 1.0})


def test_training(monkeypatch) -> None:
    """Test that the model trains correctly."""
    monkeypatch.setattr("fair_safe_opt.data.adult.fetch_openml", fake_fetch_openml)
    adult_split: AdultSplit = load_adult()

    model = SmallMLP(input_dim=adult_split.x_train.shape[1])
    history = train_erm(
        model,
        adult_split.x_train,
        adult_split.y_train,
        adult_split.x_val,
        adult_split.y_val,
        epochs=2,
    )
    assert len(history.train_losses) == 2
    assert len(history.train_accuracies) == 2
    assert len(history.val_losses) == 2
    assert len(history.val_accuracies) == 2
    assert all(torch.isfinite(torch.tensor(history.train_losses)))
    assert all(torch.isfinite(torch.tensor(history.val_losses)))
    assert all(0.0 <= acc <= 1.0 for acc in history.train_accuracies)
    assert all(0.0 <= acc <= 1.0 for acc in history.val_accuracies)


def test_training_updates_weights(monkeypatch) -> None:
    """Test that training updates model parameters."""
    monkeypatch.setattr("fair_safe_opt.data.adult.fetch_openml", fake_fetch_openml)
    adult_split: AdultSplit = load_adult()

    model = SmallMLP(input_dim=adult_split.x_train.shape[1])

    before = model.net[0].weight.detach().clone()

    train_erm(
        model,
        adult_split.x_train,
        adult_split.y_train,
        adult_split.x_val,
        adult_split.y_val,
        epochs=2,
    )

    after = model.net[0].weight.detach()

    assert not torch.equal(before, after)


def test_training_stops_after_patience_without_improvement() -> None:
    """Test that training stops when validation loss does not improve."""
    model = SmallMLP(input_dim=2, hidden_dim=2)

    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    x_val = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    y_val = torch.tensor([1.0, 1.0])

    history = train_erm(
        model,
        x,
        y,
        x_val,
        y_val,
        epochs=5,
        batch_size=4,
        learning_rate=0.0,
        patience=1,
    )

    assert len(history.train_losses) == 2
    assert len(history.val_losses) == 2


def test_prediction_threshold() -> None:
    """Test prediction threshold behavior."""
    model = SmallMLP(input_dim=1, hidden_dim=1)

    with torch.no_grad():
        model.net[0].weight.fill_(1.0)
        model.net[0].bias.fill_(0.0)
        model.net[3].weight.fill_(1.0)
        model.net[3].bias.fill_(0.0)

    x = torch.tensor([[-1.0], [0.0], [1.0]])

    preds, _ = model.preds_probs(x)

    assert torch.equal(preds, torch.tensor([1.0, 1.0, 1.0]))


def test_preds_probs_threshold_changes_predictions() -> None:
    """Test that the threshold argument actually changes which predictions are positive."""
    model = SmallMLP(input_dim=1, hidden_dim=1)

    with torch.no_grad():
        model.net[0].weight.fill_(1.0)
        model.net[0].bias.fill_(0.0)
        model.net[3].weight.fill_(1.0)
        model.net[3].bias.fill_(0.0)

    x = torch.tensor([[0.2]])

    preds_low, _ = model.preds_probs(x, threshold=0.2)
    preds_high, _ = model.preds_probs(x, threshold=0.8)

    assert not torch.equal(preds_low, preds_high)
