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
    Ra: float,
    p0: float,
    psi_0: float,
    mpol: int = 5,
    tolerance: float = 1e-4,
    tolerance_change: float = 1e-9,
):
    """
    Get Fourier coefficients which describe a given Solov'ev boundary.

    Examples:

        >>> from utils import get_solovev_boundary
        >>> Rb = get_solovev_boundary(Ra=4.0, p0=0.125, psi_0=1.0)
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


def get_RlZ_from_wout(x: Tensor, wout_path: str, mpol: Optional[int] = 5):
    """
    Compute flux surfaces geometry on a given grid from a VMEC wout file.

    Args:
        x (torch.Tensor): the rho-theta grid on which to compute the RlZ tensor.
        wout_path (str): the wout file path.
        mpol (int, optional): the highest poloidal modes to use to build the RlZ tensor. Default: 2.
    """

    wout = get_wout(wout_path)

    rho = x[:, 0]
    theta = x[:, 1]

    #  Build differentiable solution
    ns = wout["ns"][:].data.item()
    hs = 1 / (ns - 1)
    phi = torch.linspace(0, 1, ns)

    def get_fit(x: str):
        xmn = torch.from_numpy(wout[x][:].data)
        xmns = []
        #  lmns is defined on half-mesh
        if x == "lmns":
            phi[1:] = torch.linspace(0, 1, ns)[1:] - 0.5 / (ns - 1)
        for m in range(min(xmn.shape[1], mpol + 1)):
            #  Interpolate xmn / rho ** m
            xmn_ = xmn
            if m != 0:
                xmn_ = xmn / torch.sqrt(phi)[:, None] ** m
            #  Quadratic interpolation
            idx_l = (
                torch.argmin(
                    torch.relu(rho.detach()[:, None] ** 2 - phi[None, :]), dim=1
                )
                - 1
            )
            #  Boundary indices
            if m == 0:
                idx_l[idx_l == -1] = 0
            else:
                idx_l[idx_l == -1] = 1
                idx_l[idx_l == 0] = 1
            idx_l[idx_l == ns - 2] = ns - 3
            b0 = xmn_[idx_l, m]
            b1 = (xmn_[idx_l + 1, m] - xmn_[idx_l, m]) / hs
            b2 = (xmn_[idx_l + 2, m] - 2 * xmn_[idx_l + 1, m] + xmn_[idx_l, m]) / (
                2 * hs**2
            )
            interp = (
                b0
                + b1 * (rho**2 - phi[idx_l])
                + b2 * (rho**2 - phi[idx_l]) * (rho**2 - phi[idx_l + 1])
            )
            #  Linearly interpolate on axis
            if m != 0:
                #  TODO: this has no effect due to line 216 and 217
                interp[idx_l == 0] = xmn[1, m] / phi[1] ** (m / 2)
            interp *= rho**m
            xmns.append(interp)
        return torch.stack(xmns, dim=-1)

    rmnc = get_fit("rmnc")
    lmns = get_fit("lmns")
    zmns = get_fit("zmns")

    def get_x(xm, basis):
        mpol = xm.shape[-1]
        tm = torch.outer(theta, torch.arange(mpol, dtype=xm.dtype))
        if basis == "cos":
            tm = torch.cos(tm)
        else:
            tm = torch.sin(tm)
        return (tm * xm).sum(dim=1).view(-1, 1)

    R = get_x(rmnc, basis="cos")
    l = get_x(lmns, basis="sin")
    Z = get_x(zmns, basis="sin")

    return torch.cat([R, l, Z], dim=-1)
