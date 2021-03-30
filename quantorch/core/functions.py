import torch
from torch.functional import F


def sign(x: torch.Tensor) -> torch.Tensor:
    r"""A sign function that will never be zero
    \\[
    f(x) = \begin{cases}
      -1 & x < 0 \\\
      \hphantom{-}1 & x \geq 0
    \end{cases}
    \\]
    Args:
        x (torch.Tensor): Input Tensor
    """
    return torch.sign(torch.sign(x) + 0.1)


def heaviside(x: torch.Tensor) -> torch.Tensor:
    r"""Heaviside step function with output values 0 and 1.
    \\[
    q(x) = \begin{cases}
    +1 & x > 0 \\\
    \hphantom{+}0 & x \leq 0
    \end{cases}
    \\]
    Args:
        x (torch.Tensor): Input Tensor
    """
    return torch.sign(F.relu(x))
