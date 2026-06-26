import torch

def set_device(device):
    """
    Sets device used for training. Heirarchy goes CUDA, MPS, then CPU if none are available.
    returns:
    - (torch.device): device to be used.
    """
    if device is not None:
        try: 
            return torch.device(device)
        except ValueError as e:
            raise ValueError(f'Invalid device {device}. Use cpu, mps, or cuda.')
    else:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")