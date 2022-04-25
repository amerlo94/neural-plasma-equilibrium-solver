"""Utility functions."""

import math
from typing import Optional, Union

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
    Get f(psi) = R**2 * Bsupv or p(psi) from a vmec equilibrium.

    Psi is the poloidal flux, which is called chi in VMEC.

    TODO: check if f fitting is correct!
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
        p_fit = np.polynomial.Polynomial.fit(
            chi / chi_edge, p(phi / phi_edge), deg=5, domain=[0, 1], window=[0, 1]
        )
        return p_fit.coef.tolist()
    #  In VMEC terms, bvco == f(psi)
    #  See bcovar.f
    bvco = wout["bvco"][:].data
    #  Move it to full mesh
    f = np.empty_like(bvco)
    f[1:-1] = 0.5 * (bvco[1:-1] + bvco[2:])
    #  Extend it to axis and LCFS
    f[0] = 1.5 * bvco[1] - 0.5 * bvco[2]
    f[-1] = 1.5 * bvco[-1] - 0.5 * bvco[-2]
    #  Perform f**2 fit
    f_fit = np.polynomial.Polynomial.fit(
        chi / chi_edge, f**2, deg=5, domain=[0, 1], window=[0, 1]
    )
    return f_fit.coef.tolist()


def get_flux_surfaces_from_wout(wout_path: str):
    wout = get_wout(wout_path)
    rmnc = torch.as_tensor(wout["rmnc"][:]).clone()
    zmns = torch.as_tensor(wout["zmns"][:]).clone()
    R = ift(rmnc, basis="cos")
    Z = ift(zmns, basis="sin")
    #  Return poloidal on the flux surfaces also
    chi = torch.as_tensor(wout["chi"][:]).clone()
    #  Return flux surfaces as grid
    return torch.stack([R.view(-1), Z.view(-1)], dim=-1), chi


def ift(
    xm: Tensor,
    basis: str,
    ntheta: Optional[Union[int, Tensor]] = 40,
    endpoint: Optional[bool] = True,
):
    """The inverse Fourier transform."""
    assert basis in ("cos", "sin")
    assert len(xm.shape) <= 2
    if isinstance(ntheta, Tensor):
        theta = ntheta
    else:
        if endpoint:
            theta = torch.linspace(0, 2 * math.pi, ntheta, dtype=xm.dtype)
        else:
            theta = torch.linspace(0, 2 * math.pi, ntheta + 1, dtype=xm.dtype)[:-1]
    mpol = xm.shape[-1]
    tm = torch.outer(theta, torch.arange(mpol, dtype=xm.dtype))
    if basis == "cos":
        tm = torch.cos(tm)
    else:
        tm = torch.sin(tm)
    #  One flux surface only
    if len(xm.shape) == 1:
        return (tm * xm).sum(dim=1)
    #  Multiple flux surfaces
    tm = tm[None, ...]
    return torch.einsum("stm,sm->st", tm, xm).contiguous()


def get_solovev_boundary(
    Ra: float = 4.0,
    p0: float = 0.125,
    psi_0: float = 1.0,
    mpol: int = 5,
    tolerance: float = 1e-4,
    tolerance_change: float = 1e-9,
):
    """
    Get Fourier coefficients which describe a given Solov'ev boundary.

    Example:

    >>> from utils import get_solovev_boundary
    >>> Rb = get_solovev_boundary(mpol=5)
    >>> len(Rb)
    5
    """

    #  Build theta grid
    ntheta = 40
    theta = torch.linspace(0, 2 * math.pi, ntheta, dtype=torch.float64)

    #  Build boundary from analytical Solov'ev solution
    #  R**2 = Ra**2 - psi_0 sqrt(8 / p0) rho cos(theta)
    Rsq = Ra**2 - psi_0 * math.sqrt(8 / p0) * torch.cos(theta)

    #  Build boundary and set initial guess
    Rb = torch.zeros(mpol, dtype=torch.float64)
    Rb[0] = Ra
    Rb.requires_grad_(True)

    optim = torch.optim.LBFGS([Rb], lr=1e-2)

    def loss_fn():
        R = ift(Rb, basis="cos", ntheta=ntheta)
        return ((R**2 - Rsq) ** 2).sum()

    def closure():
        optim.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    #  Get initial loss
    loss = loss_fn()

    while True:
        loss_old = loss
        optim.step(closure)
        loss = loss_fn()
        if loss < tolerance:
            break
        if abs(loss_old.item() - loss.item()) < tolerance_change:
            break

    return Rb.detach()
