from typing import Callable, Union


import torch

from quantorch.core import functions


class _STESign(torch.autograd.Function):

    def forward(ctx, x: torch.Tensor, clip_value: float = 1.0):
        ctx.save_for_backward(x)
        ctx.clip_value = clip_value
        return functions.sign(x)

    def backward(ctx, grad_outputs):
        x, _ = ctx.saved_tensors
        clip_value = ctx.clip_value

        grad_inputs = grad_outputs.clone()
        zeros = torch.zeros_like(grad_inputs)
        return torch.where(torch.abs(x) <= clip_value, grad_inputs, zeros)


def ste_sign(x: torch.Tensor, clip_value: float = 1.0) -> torch.Tensor:
    _ste_sign = _STESign.apply
    return _ste_sign(x, clip_value)


class STESign(torch.nn.Module):

    def __init__(self, clip_value: float = 1.0):
        self.clip_value = clip_value
        super(STESign, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ste_sign(x, self.clip_value)


class _ApproxSign(torch.autograd.Function):

    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        return functions.sign(x)

    def backward(ctx, grad_outputs):
        x, _ = ctx.saved_tensors
        abs_x = torch.abs(x)

        grad_inputs = grad_outputs.clone()
        zeros = torch.zeros_like(grad_inputs)
        return torch.where(abs_x <= 1., (1 - abs_x)*2*grad_inputs, zeros)


def approx_sign(x: torch.Tensor) -> torch.Tensor:
    _approx_sign = _ApproxSign.apply
    return _approx_sign(x)


class ApproxSign(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return approx_sign(x)


class _SwishSign(torch.autograd.Function):

    def forward(ctx, x: torch.Tensor, beta: float = 5.0):
        ctx.save_for_backward(x)
        ctx.beta = 5.0
        return functions.sign(x)

    def backward(ctx, grad_outputs):
        x, _ = ctx.saved_tensors
        beta = ctx.beta

        beta_x = beta * x
        grad_inputs = grad_outputs.clone()
        return grad_inputs * beta * (2 - beta_x * torch.tanh(beta_x * 0.5)) / (1 + torch.cosh(beta_x))

def swish_sign(x: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    _swish_sign = _SwishSign.apply
    return _swish_sign(x, beta)


class SwishSign(torch.nn.Module):

    def __init__(self, beta: float = 5.0):
        self.beta = beta
        super(SwishSign, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish_sign(x, self.beta)


class _STETern(torch.autograd.Function):

    def forward(ctx, x: torch.Tensor, threshold_value: float = 0.05, ternary_weight_networks: bool = False, clip_value: float = 1.0):
        ctx.save_for_backward(x)
        ctx.threshold_value = threshold_value
        ctx.ternary_weight_networks = ternary_weight_networks
        ctx.clip_value = clip_value

        if ternary_weight_networks:
            threshold = 0.7 * torch.sum(torch.abs(x)) / x.numel().to(x.dtype)
        else:
            threshold = threshold_value

        return torch.sign(torch.sign(x + threshold) + torch.sign(x - threshold))

    def backward(ctx, grad_outputs):
        x, _ = ctx.saved_tensors

        clip_value = ctx.clip_value

        grad_inputs = grad_outputs.clone()
        zeros = torch.zeros_like(grad_inputs)
        return torch.where(torch.abs(x) <= clip_value, grad_inputs, zeros)


def ste_tern(x: torch.Tensor, threshold_value: float = 0.05, ternary_weight_networks: bool = False, clip_value: float = 1.0) -> torch.Tensor:
    _ste_tern = _STETern.apply
    return _ste_tern(x, threshold_value, ternary_weight_networks, clip_value)


class STETern(torch.nn.Module):

    def __init__(self, threshold_value: float = 0.05, ternary_weight_networks: bool = False, clip_value: float = 1.0):
        self.threshold_value = threshold_value
        self.ternary_weight_networks = ternary_weight_networks
        self.clip_value = clip_value
        super(STETern, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ste_tern(x, self.threshold_value, self.ternary_weight_networks, self.clip_value)


class _STEHeaveside(torch.autograd.Function):

    def forward(ctx, x: torch.Tensor, clip_value: float = 0.1):
        ctx.save_for_backward(x)
        return functions.heaveside(x)

    def backward(ctx, grad_outputs):
        x, _ = ctx.saved_tensors

        clip_value = ctx.clip_value

        grad_inputs = grad_outputs.clone()
        zeros = torch.zeros_like(grad_inputs)
        return torch.where(torch.abs(x) <= clip_value, grad_inputs, zeros)


def ste_heaveside(x: torch.Tensor, clip_value: float = 0.1) -> torch.Tensor:
    _ste_heaveside = _STEHeaveside.apply
    return _ste_heaveside(x, clip_value)


class STEHeaveside(torch.nn.Module):

    def __init__(self, clip_value: float = 1.0):
        self.clip_value = clip_value
        super(STEHeaveside, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ste_heaveside(x, self.clip_value)
