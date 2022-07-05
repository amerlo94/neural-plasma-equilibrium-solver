"""Utility functions."""

import math
# from functools import cache
from typing import Optional, Union, List, Tuple

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


def ift_2D(
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


# @cache
def get_fourier_basis(
    mpol: int,
    ntor: int,
    ntheta: int,
    nzeta: int,
    endpoint: bool,
    num_field_period: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Compute and cache the Fourier basis."""

    poloidal_modes = torch.arange(0, mpol + 1, dtype=dtype)[:, None]
    toroidal_modes = torch.arange(-ntor, ntor + 1, dtype=dtype)[None, :]

    if endpoint:
        thetas = torch.linspace(0, 2 * math.pi, ntheta + 1, dtype=dtype)[:-1]
        phis = torch.linspace(
            0, 2 * math.pi / num_field_period, nzeta + 1, dtype=dtype)[:-1]
    else:
        thetas = torch.linspace(0, 2 * math.pi, ntheta, dtype=dtype)
        phis = torch.linspace(
            0, 2 * math.pi / num_field_period, nzeta, dtype=dtype
        )

    costzmn = torch.empty(ntheta, nzeta, mpol + 1, 2 * ntor + 1, dtype=dtype)
    sintzmn = torch.empty(ntheta, nzeta, mpol + 1, 2 * ntor + 1, dtype=dtype)

    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            costzmn[i, j] = torch.cos(
                poloidal_modes * theta - num_field_period * toroidal_modes * phi
            )
            sintzmn[i, j] = torch.sin(
                poloidal_modes * theta - num_field_period * toroidal_modes * phi
            )

    costzmn = costzmn.to(device)
    sintzmn = sintzmn.to(device)

    return costzmn[None, ...], sintzmn[None, ...]


def ift(
    tensors: Union[List[Tensor], Tuple[Tensor]],
    ntheta: Optional[int] = None,
    nzeta: Optional[int] = None,
    endpoint: Optional[bool] = False,
    num_field_period: Optional[int] = 5,
):
    """
    Inverse Fourier transform.

    Examples:
        # s, m, n
        >>> from utils import ift
        >>> tensor = torch.rand(1, 2, 3)
        >>> ift((tensor, None)).size()
        torch.Size([1, 3, 3])

        >>> ift((tensor, None), ntheta=40, nzeta=36).size()
        torch.Size([1, 40, 36])

        >>> out = ift((tensor, None))
        >>> out[0, 0, 0] == tensor.sum()
        tensor(True)

    TODO-1(@amerlo): improv function signature
    TODO-1(@amerlo): time me and optimize me (this should be feasible with cartesian product)
    TODO-1(@amerlo): does it make sense to make it a class?
    TODO-1(@amerlo): add documentation.
    """

    assert len(tensors) == 2
    assert not (tensors[0] is None and tensors[1] is None)

    for tensor in tensors:
        if tensor is not None:
            assert (
                len(tensor.shape) == 3
            ), "Input tensor should be a tensor of (num_samples, poloidal_modes, toroidal_modes)."

    cosmn = tensors[0]
    sinmn = tensors[1]

    if cosmn is not None and sinmn is not None:
        assert cosmn.shape == sinmn.shape

    #  Select tensor as reference
    tensor = cosmn if cosmn is not None else sinmn

    mpol = tensor.size(1) - 1
    ntor = int((tensor.size(2) - 1) / 2)

    dtype = tensor.dtype
    device = tensor.device

    if ntheta is None:
        ntheta = 2 * mpol + 1
    if nzeta is None:
        nzeta = 2 * ntor + 1

    costzmn, sintzmn = get_fourier_basis(
        mpol,
        ntor,
        ntheta,
        nzeta,
        endpoint,
        num_field_period,
        dtype=dtype,
        device=device,
    )

    if cosmn is not None and sinmn is None:
        return torch.einsum("stzmn,smn->stz", costzmn, cosmn).contiguous()

    if sinmn is not None and cosmn is None:
        return torch.einsum("stzmn,smn->stz", sintzmn, sinmn).contiguous()

    return (
        torch.einsum("stzmn,smn->stz", costzmn, cosmn)
        + torch.einsum("stzmn,smn->stz", sintzmn, sinmn)
    ).contiguous()



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


def get_RlZ_from_wout(
    x: Tensor, wout_path: str, deg: Optional[int] = 1, mpol: Optional[int] = 2
):
    """
    Compute flux surfaces geometry on a given grid from a VMEC wout file.

    Args:
        x (torch.Tensor): the rho-theta grid on which to compute the RlZ tensor.
        wout_path (str): the wout file path.
        deg (int, optional): the degree of the polynomials to use to fit the VMEC solution. Default: 1.
        mpol (int, optional): the highest poloidal modes to use to build the RlZ tensor. Default: 2.
    """

    wout = get_wout(wout_path)

    rho = x[:, 0]
    theta = x[:, 1]

    #  Build differentiable solution
    ns = wout["ns"][:].data.item()
    phi = np.linspace(0, 1, ns)

    def get_fit(x: str):
        xmn = wout[x][:].data
        xmn_ = []
        #  lmns is defined on half-mesh
        if x == "lmns":
            ns = xmn.shape[0]
            phi[1:] = np.linspace(0, 1, ns)[:-1] + 1 / (2 * ns)
        for m in range(min(xmn.shape[1], mpol + 1)):
            #  Fit xm / rho ** m
            factor = np.ones_like(phi)
            if m != 0:
                factor = np.sqrt(phi) ** m
            coef = np.polynomial.Polynomial.fit(
                phi[1:], xmn[1:, m] / factor[1:], deg=deg, domain=[0, 1], window=[0, 1]
            ).coef.tolist()
            fit = 0
            for i, c in enumerate(coef):
                fit += c * rho ** (2 * i + m)
            xmn_.append(fit)
        return torch.stack(xmn_, dim=-1)

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
