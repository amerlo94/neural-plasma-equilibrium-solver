"""Utility functions."""

import torch
from torch import Tensor


def grad(outputs: Tensor, inputs: Tensor, **kwargs) -> Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones(outputs.shape),
        **kwargs
        )[0]

def log_gradients(model, lr: float, t: int):
    string = []
    for p in model.parameters():
        grad = p.grad
        if grad is not None:
            ratio = lr * torch.linalg.norm(grad) / torch.linalg.norm(p)
            string.append(f"ratio = {ratio.item():.2e}")
    print(f"iter={t:4d}, " + ", ".join(string))