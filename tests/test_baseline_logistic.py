from fair_safe_opt.models.logistic import LogisticRegression
from fair_safe_opt.training.erm import train_erm
from fair_safe_opt.data.adult import load_adult, AdultSplit
from test_imports import make_fake_adult_frame, fake_fetch_openml 
import torch


def test_leq_zero_input_dimension() -> None:
    '''Test that the model raises an error for non-positive input dimensions.'''
    try:
        model = LogisticRegression(input_dim=0)
    except ValueError as e:
        assert str(e) == "Input dimension must be positive."
        
    try:
        model = LogisticRegression(input_dim=-5)
    except ValueError as e:
        assert str(e) == "Input dimension must be positive."

def test_malformed_input() -> None:
    '''Test that the model raises an error for malformed input.'''
    model = LogisticRegression(input_dim=5)
    try:
        model.predict(torch.tensor([[1, 2], [3, 4]]))  # Wrong input shape
    except ValueError as e:
        assert str(e) == "Input feature dimension mismatch."

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
    assert history is not None
