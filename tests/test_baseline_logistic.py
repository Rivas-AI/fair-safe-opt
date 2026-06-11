from fair_safe_opt.models.logistic import LogisticRegression
from fair_safe_opt.training.erm import train_erm
from fair_safe_opt.data.adult import load_adult, AdultSplit
from test_imports import make_fake_adult_frame, fake_fetch_openml 
import torch
import pytest


def test_leq_zero_input_dimension() -> None:
    '''Test that the model raises an error for non-positive input dimensions.'''
    with pytest.raises(ValueError, match="Input dimension must be positive."):
        LogisticRegression(input_dim=0)

    with pytest.raises(ValueError, match="Input dimension must be positive."):
        LogisticRegression(input_dim=-5)

def test_malformed_input() -> None:
    '''Test that the model raises an error for malformed input.'''
    model = LogisticRegression(input_dim=5)
    with pytest.raises(ValueError, match="Input feature dimension mismatch."):
        model.predict(torch.tensor([[1, 2], [3, 4]]))  # Wrong input shape

def test_predictions(monkeypatch) -> None:
    '''Test that the model makes predictions with the correct shape and values.'''
    monkeypatch.setattr("fair_safe_opt.data.adult.fetch_openml", fake_fetch_openml)
    adult_split: AdultSplit = load_adult()
    
    model = LogisticRegression(input_dim=adult_split.x_train.shape[1])
    preds = model.predict(torch.tensor(adult_split.x_test))
    assert preds.shape == adult_split.y_test.shape
    assert set(preds.tolist()).issubset({0.,1.})

def test_training(monkeypatch) -> None:
    '''Test that the model trains correctly.'''
    monkeypatch.setattr("fair_safe_opt.data.adult.fetch_openml", fake_fetch_openml)
    adult_split: AdultSplit = load_adult()
    
    model = LogisticRegression(input_dim=adult_split.x_train.shape[1])
    history = train_erm(model, torch.tensor(adult_split.x_train), torch.tensor(adult_split.y_train), epochs=2)
    assert len(history.losses) == 2
    assert len(history.accuracies) == 2
    assert all(torch.isfinite(torch.tensor(history.losses)))
    assert all(0.0 <= acc <= 1.0 for acc in history.accuracies)
