"""Utility functions."""

import math
from typing import Optional

import netCDF4
import numpy as np
import torch
from torch import Tensor


def grad(outputs: Tensor, inputs: Tensor, **kwargs) -> Tensor:
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones(outputs.shape), **kwargs
    )[0]


def log_gradients(model, lr: float, t: int):
    string = []
    for p in model.parameters():
        grad = p.grad
        if grad is not None:
            ratio = lr * torch.linalg.norm(grad) / torch.linalg.norm(p)
            string.append(f"ratio={ratio.item():.2e}")
    print(f"iter={t:5d}, " + ", ".join(string))


def mae(preds: Tensor, target: Tensor, eps=1.17e-6):
    mae = (preds - target).abs() / torch.clamp(torch.abs(target), min=eps)
    return mae.mean()


def get_wout(wout_path: str):
    return netCDF4.Dataset(wout_path)


def get_profile_from_wout(wout_path: str, profile: str):
    """
    Get f(psi) = R * Bv or p(psi) from a vmec equilibrium.

    Psi is the poloidal flux, which is called chi in VMEC.
    """
    assert profile in ("p", "f")
    wout = get_wout(wout_path)
    #  Compute chi (i.e., domain) on half-mesh, normalized to boundary value
    phi = wout["phi"][:].data
    phi_edge = phi[-1]
    chi = wout["chi"][:].data
    chi_edge = chi[-1]
    if profile == "p":
        #  Get pressure
        p = np.polynomial.Polynomial(wout["am"][:].data)
        p_fit = np.polynomial.Polynomial.fit(chi / chi_edge, p(phi / phi_edge), deg=5)
        return p_fit.coef.tolist()
    #  Get Fourier coefficients for f
    rmnc = torch.as_tensor(wout["rmnc"][:]).clone()
    bsubvmnc = torch.as_tensor(wout["bsubvmnc"][:]).clone()
    assert rmnc.shape[0] == bsubvmnc.shape[0]
    #  Compute quantities
    R = ift(rmnc, basis="cos")
    bsubv = ift(bsubvmnc, basis="cos")
    #  Move quantities to half-mesh
    R = 0.5 * (R[1:] + R[:-1])
    bsubv = bsubv[1:]
    chi = 0.5 * (chi[1:] + chi[:-1])
    f = (R * bsubv).mean(dim=1)
    #  Perform fit for f, use fifth-order polynomial as in the paper
    f_fit = np.polynomial.Polynomial.fit(chi, f, deg=5)
    return f_fit.coef.tolist()


def ift(xm: Tensor, basis: str, ntheta: Optional[int] = 40):
    """The inverse Fourier transform."""
    assert basis in ("cos", "sin")
    theta = torch.linspace(0, 2 * math.pi, ntheta + 1, dtype=xm.dtype)[:-1]
    mpol = xm.shape[1]
    tm = theta[..., None] * torch.arange(mpol, dtype=xm.dtype)[None, ...]
    if basis == "cos":
        tm = torch.cos(tm)
    else:
        tm = torch.sin(tm)
    tm = tm[None, ...]
    return torch.einsum("stm,sm->st", tm, xm)
