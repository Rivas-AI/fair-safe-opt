import torch

def set_device():
    """
    Sets device used for training. Heirarchy goes CUDA, MPS, then CPU if none are available.
    returns:
    - (torch.device): device to be used.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")