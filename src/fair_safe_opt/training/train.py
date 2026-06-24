import torch

from fair_safe_opt.training.erm import train_erm, TrainingHistory

def train_model(
    model: torch.nn.Module, 
    method: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    out_dir: torch.Tensor,
    epochs: int,
    patience: int, 
    learning_rate: float
) -> TrainingHistory:
    '''Given a model, method, and training parameters, train the model.'''
    if method == "erm":
        history: TrainingHistory = train_erm(
            model,
            x_train, 
            y_train, 
            x_val, 
            y_val, 
            out_dir,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate
        )
    else:
        raise ValueError("config['method'] not in {'erm'}")
    
    return history