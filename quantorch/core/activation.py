import torch


def hard_tanh(x: torch.Tensor) -> torch.Tensor:
    r"""Hard tanh activation function.

    Args:
        x (torch.Tensor): Input tensor
    """
    return x.clamp(-1, 1)


def leaky_tanh(x: torch.Tensor, alpha: float=0.2) -> torch.Tensor:
    r"""Leaky tanh activation function.
    
    Args:
        x (torch.Tensor): Input tensor
        alpha (float): Slope of activation outside of [-1,1] ()(Default=0.2)
    """
    one = torch.tensor(1, dtype=x.dtype)
    return x.clamp(-1, 1) + (torch.max(x, one) - 1) * alpha + (torch.max(x, -one) + 1) * alpha
