import fair_safe_opt
import pandas as pd, numpy as np
from fair_safe_opt.data.adult import load_adult
from sklearn.utils import Bunch

def test_import_package() -> None:
    assert fair_safe_opt.__version__ == "0.1.0"

'''
========================================================================
ADULT DATASET TESTING
========================================================================
'''

def make_fake_adult_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 38, 44, 28, 52, 31, 40, 60, 23, 49],
            "workclass": ["Private", "Private", "Gov", "Private", "Self", "Private", "Gov", "Self", "Private", "Gov"],
            "fnlwgt": [100000, 120000, 90000, 130000, 80000, 110000, 95000, 70000, 140000, 85000],
            "education": ["Bachelors", "HS-grad", "Masters", "HS-grad", "Bachelors", "HS-grad", "Masters", "HS-grad", "Bachelors", "Masters"],
            "education-num": [13, 9, 14, 9, 13, 9, 14, 9, 13, 14],
            "marital-status": ["Never", "Married", "Married", "Never", "Married", "Never", "Married", "Married", "Never", "Married"],
            "occupation": ["Tech", "Sales", "Exec", "Sales", "Exec", "Tech", "Tech", "Exec", "Sales", "Tech"],
            "relationship": ["Not-in-family", "Husband", "Wife", "Own-child", "Husband", "Not-in-family", "Wife", "Husband", "Own-child", "Wife"],
            "race": ["White", "Black", "White", "Asian", "White", "Black", "Asian", "White", "White", "Black"],
            "sex": ["Male", "Female", "Female", "Male", "Male", "Female", "Female", "Male", "Male", "Female"],
            "capital-gain": [0, 0, 5000, 0, 10000, 0, 0, 2000, 0, 3000],
            "capital-loss": [0, 0, 0, 0, 0, 1000, 0, 0, 0, 0],        "hours-per-week": [40, 35, 45, 20, 50, 30, 40, 55, 25, 45],
            "native-country": ["US", "US", "US", "Canada", "US", "Canada", "US", "US", "Canada", "US"],
            "class": ["<=50K", "<=50K", ">50K", "<=50K", ">50K", "<=50K", ">50K", ">50K", "<=50K", ">50K"],
        }
    )

def fake_fetch_openml(data_id, as_frame) -> Bunch:
    return Bunch(frame=make_fake_adult_frame())

def test_adult_columns(monkeypatch) -> None:
    monkeypatch.setattr("fair_safe_opt.data.adult.fetch_openml", fake_fetch_openml)
    adult = load_adult()

    assert adult.x_train.shape[1] == len(adult.feature_names)
    assert adult.x_train.shape[1] >= np.sum(adult.continuous_mask) + 18 #2*9 categorical columns
    assert "sex" not in adult.feature_names
    assert "class" not in adult.feature_names

def test_adult_data_shapes(monkeypatch) -> None:
    monkeypatch.setattr("fair_safe_opt.data.adult.fetch_openml", fake_fetch_openml)
    adult = load_adult()

    assert adult.x_train.shape[1] == 26
    assert adult.x_val.shape[1] == adult.x_train.shape[1]
    assert adult.x_test.shape[1] == adult.x_train.shape[1]
    assert adult.y_train.shape[0] == adult.x_train.shape[0]
    assert adult.y_val.shape[0] == adult.x_val.shape[0]
    assert adult.y_test.shape[0] == adult.x_test.shape[0]
    assert adult.a_train.shape[0] == adult.x_train.shape[0]
    assert adult.a_val.shape[0] == adult.x_val.shape[0]
    assert adult.a_test.shape[0] == adult.x_test.shape[0]
    
def test_continuous_mask(monkeypatch) -> None:
    monkeypatch.setattr("fair_safe_opt.data.adult.fetch_openml", fake_fetch_openml)
    adult = load_adult()

    continuous_cols = adult.x_train[0][adult.continuous_mask]
    categorical_cols = adult.x_train[0][~adult.continuous_mask]

    assert type(continuous_cols) == np.ndarray
    assert continuous_cols.shape[0] + categorical_cols.shape[0] == adult.x_train.shape[1]
    assert set(continuous_cols) != {0, 1}
    assert set(categorical_cols) == {0, 1}

def test_standard_scaled_features(monkeypatch) -> None:
    monkeypatch.setattr("fair_safe_opt.data.adult.fetch_openml", fake_fetch_openml)
    adult = load_adult()

    continuous_train = adult.x_train[:, adult.continuous_mask]

    assert np.all(abs(continuous_train.mean(axis=0)) < 1e-6)
    assert np.all(abs(1 - continuous_train.std(axis=0)) < 1e-6)