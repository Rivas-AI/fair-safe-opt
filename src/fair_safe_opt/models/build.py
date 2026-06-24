from fair_safe_opt.data.adult import AdultSplit
from fair_safe_opt.models.logistic import LogisticRegression
from fair_safe_opt.models.mlp import SmallMLP


def build_model(architecture: str, split: AdultSplit, hidden_dim: int, dropout: float):
    if architecture == "logistic":
        model = LogisticRegression(split.x_train.shape[1])
    elif architecture == "small_mlp":
        model = SmallMLP(split.x_train.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError("architecture not in {'logistic', 'small_mlp'}")
    
    return model