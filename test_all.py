"""Test file."""

import pytest
import torch
import numpy as np

from models import GradShafranovMLP
from physics import (
    HighBetaEquilibrium,
    GradShafranovEquilibrium,
    InverseGradShafranovEquilibrium,
)
from utils import grad, get_profile_from_wout, ift, get_RlZ_from_wout, get_wout

#########
# Utils #
#########


def test_grad():
    x = torch.randn(10, 2)
    x.requires_grad_()
    y = (x**2).sum(dim=1)
    assert (2 * x == grad(y, x, retain_graph=True)).all()
    a = torch.randn(2, 16)
    y = (x[..., None] * a[None, ...]).sum(dim=(1, 2))
    for i in range(y.shape[0]):
        assert torch.allclose(
            a.sum(dim=-1), grad(y, x, retain_graph=True)[i], atol=0, rtol=0
        )


###########
# Physics #
###########


@pytest.mark.parametrize("ns", (5, 10, 50))
def test_high_beta_normalized_psi(ns: int):
    equi = HighBetaEquilibrium()
    #  Not-normalized equilibrium
    x = equi.grid(ns=ns, normalized=False)
    psi = equi.psi(x)
    #  Normalized equilibrium
    x_ = equi.grid(ns=ns, normalized=True)
    psi_ = equi.psi_(x_)
    assert ((psi_ * equi.psi_0 - psi).abs() < 1e-9).all()


@pytest.mark.parametrize("normalized", (True, False))
@pytest.mark.parametrize("ns", (5, 10, 50))
def test_high_beta_pde_closure(normalized: bool, ns: int):
    equi = HighBetaEquilibrium(normalized=normalized)
    x = equi.grid(ns=ns)
    x.requires_grad_()
    if normalized:
        psi = equi.psi_(x)
    else:
        psi = equi.psi(x)
    residual = equi.pde_closure(x, psi).item()
    assert abs(residual) < 1e-10


@pytest.mark.parametrize("normalized", (True, False))
@pytest.mark.parametrize("ns", (5, 10, 50))
def test_high_beta_mae_pde_loss(normalized: bool, ns: int):
    equi = HighBetaEquilibrium(normalized=normalized)
    x = equi.grid(ns=ns)
    x.requires_grad_()
    if normalized:
        psi = equi.psi_(x)
    else:
        psi = equi.psi(x)
    mae = equi.mae_pde_loss(x, psi).item()
    assert mae < 1e-5


@pytest.mark.parametrize("normalized", (True, False))
@pytest.mark.parametrize("nbatches", (1, 5))
@pytest.mark.parametrize("ns", (10,))
def test_high_beta_points_different_than_grid(normalized: bool, nbatches: int, ns: int):
    equi = HighBetaEquilibrium(normalized=normalized, ndomain=ns**2)
    grid = equi.grid(ns=ns)
    for _, (x, _, _) in zip(range(nbatches), equi):
        #  The first point is on the axis in both cases
        assert (x[1:] != grid[1:]).all()


@pytest.mark.parametrize("normalized", (True, False))
@pytest.mark.parametrize("nbatches", (1, 5))
def test_high_beta_consistent_points(normalized: bool, nbatches: int):
    equi = HighBetaEquilibrium(normalized=normalized)
    domain_points = []
    boundary_points = []
    for _, (x_domain, x_boundary, _) in zip(range(nbatches), equi):
        domain_points.append(x_domain)
        boundary_points.append(x_boundary)
    for i, (x_domain, x_boundary, _) in zip(range(nbatches), equi):
        assert (domain_points[i] == x_domain).all()
        assert (boundary_points[i] == x_boundary).all()
    #  Points should be different if we change the equilibrium seed
    equi.seed = equi.seed - 1
    for i, (x_domain, x_boundary, _) in zip(range(nbatches), equi):
        assert (domain_points[i][1:] != x_domain[1:]).all()
        #  Check only theta value here
        assert (boundary_points[i][:, 1] != x_boundary[:, 1]).all()


@pytest.mark.parametrize("basis", ("cos", "sin"))
@pytest.mark.parametrize("mpol", (2, 7, 14))
@pytest.mark.parametrize("seed", (42, 24))
def test_ift_analytical_values(basis, mpol, seed):
    atol = 1e-14
    generator = torch.Generator()
    generator.manual_seed(seed)
    xm = torch.randn((1, mpol), generator=generator, dtype=torch.float64)
    res = ift(xm, basis=basis, endpoint=False)
    if basis == "cos":
        #  x(s, 0) = \sum xm
        assert abs(res[0, 0] - xm.sum()) < atol
        assert abs(res.mean() - xm[0, 0]) < atol
    else:
        #  x(s, 0) = 0
        assert res[0, 0] == 0
        #  x(s, pi) = 0
        assert abs(res[0, int(res.shape[1] / 2)]) < atol
        #  x(s, pi/2) = sum of odd modes with +- sign
        res_ = 0
        sign = 1
        for m in range(mpol):
            if m % 2 == 0:
                continue
            res_ += sign * xm[0, m]
            sign *= -1
        assert abs(res[0, int(res.shape[1] / 4)] - res_) < atol


@pytest.mark.parametrize("basis", ("cos", "sin"))
@pytest.mark.parametrize("ns", (0, 1, 10))
@pytest.mark.parametrize("mpol", (1, 7, 14))
@pytest.mark.parametrize("ntheta", (18, 36))
def test_ift_shape(basis, ns, mpol, ntheta):
    if ns == 0:
        xm = torch.rand(mpol)
    else:
        xm = torch.randn((ns, mpol))
    res = ift(xm, basis=basis, ntheta=ntheta)
    if len(xm.shape) == 2:
        assert res.shape == (ns, ntheta)
    else:
        assert res.shape[-1] == ntheta


@pytest.mark.parametrize("wout_path", ("data/wout_DSHAPE.nc", "data/wout_SOLOVEV.nc"))
@pytest.mark.parametrize("profile", ("p", "f"))
def test_len_vmec_profile_coefs(wout_path, profile):
    """The fit should return a fifth-order polynomail."""
    coef = get_profile_from_wout(wout_path, profile)
    assert len(coef) == 6


@pytest.mark.parametrize("psi", (0,))
@pytest.mark.parametrize("tolerance", (1e-3, 1e-5))
def test_model_find_x_of_psi(psi, tolerance):
    initial_guess = torch.rand((1, 2)) * 1e-3
    model = GradShafranovMLP()
    x = model.find_x_of_psi(psi=psi, initial_guess=initial_guess, tolerance=tolerance)
    psi_hat = model.forward(x)
    assert abs(psi - psi_hat) < tolerance


@pytest.mark.parametrize("noise", (0, 1e-3, 1e-2, 1e-1))
@pytest.mark.parametrize("reduction", ("mean", None))
@pytest.mark.parametrize("fsq0", (1, 4, 7, 11))
@pytest.mark.parametrize("normalized", (False, True))
def test_grad_shafranov_eps(noise, reduction, fsq0, normalized):
    #  Construct a Solovev like F**2 profile
    fsq = (fsq0, -4 * fsq0 / 10)
    equi = GradShafranovEquilibrium(fsq=fsq, normalized=normalized)
    x = equi.grid()
    x.requires_grad_()
    not_normalized_grid = x
    if normalized:
        not_normalized_grid = x * equi.Rb[0]
    psi = equi.psi(not_normalized_grid)
    if noise == 0:
        eps = equi.eps(x, psi=psi, reduction=reduction).max().item()
        assert eps < 1e-6
    else:
        #  Apply Gaussian noise to solution
        psi *= 1 + torch.randn(psi.shape) * noise
        eps = equi.eps(x, psi=psi, reduction=reduction).max().item()
        assert eps > noise


#  TODO: improve fitting of RlZ, especially for lambda
@pytest.mark.parametrize("wout_path", ("data/wout_DSHAPE.nc", "data/wout_SOLOVEV.nc"))
@pytest.mark.parametrize("ntheta", (32,))
@pytest.mark.parametrize("s", range(128))
@pytest.mark.parametrize("xmn", ("rmnc", "lmns", "zmns"))
def test_get_RlZ_from_wout(wout_path, ntheta, xmn, s):

    equi = InverseGradShafranovEquilibrium.from_vmec(wout_path)
    equi.ntheta = ntheta
    grid = equi.grid().to(torch.float64)

    RlZ = get_RlZ_from_wout(grid, wout_path)
    if xmn == "rmnc":
        x = RlZ[:, 0]
    elif xmn == "lmns":
        x = RlZ[:, 1]
    elif xmn == "zmns":
        x = RlZ[:, 2]
    x = x.view(-1, equi.ntheta)

    wout = get_wout(wout_path)
    basis = "cos" if xmn == "rmnc" else "sin"
    vmec_x = ift(
        torch.as_tensor(wout[xmn][:]).clone(),
        basis=basis,
        ntheta=grid[: equi.ntheta, 1],
    )

    #  Get quantity at the same radial location
    rho_idx = int(s / 128 * x.shape[0])
    rho = grid[:: equi.ntheta, 0][rho_idx]

    ns = vmec_x.shape[0]
    if xmn == "lmns":
        phi = torch.zeros(ns)
        phi[1:] = torch.linspace(0, 1, ns)[:-1] + 1 / (2 * ns)
    else:
        phi = torch.linspace(0, 1, vmec_x.shape[0])

    phi_idx = (phi - rho**2).abs().argmin()

    #  Skip if radial location does not match
    if (phi[phi_idx] - rho**2).abs() > 1e-3:
        return

    #  Skip if on axis
    if phi[phi_idx] == 0:
        return

    assert torch.allclose(
        x[rho_idx], vmec_x[phi_idx], rtol=1e-2, atol=0
    ), f"mae={((x[rho_idx] - vmec_x[phi_idx])/vmec_x[phi_idx]).abs().mean():.2e} at rho={rho:.2f}"


@pytest.mark.parametrize("wout_path", ("data/wout_DSHAPE.nc", "data/wout_SOLOVEV.nc"))
@pytest.mark.parametrize("ntheta", (32,))
def test_inverse_grad_shafranov_pde_closure(wout_path, ntheta):

    equi = InverseGradShafranovEquilibrium.from_vmec(wout_path)
    equi.ntheta = ntheta
    x = equi.grid()

    #  Do not use axis and boundary
    x = x[equi.ntheta : -equi.ntheta, :]
    x.requires_grad_()

    RlZ = get_RlZ_from_wout(x, wout_path)

    fsq = equi.pde_closure(x, RlZ).item()
    mean_f = np.sqrt(fsq) / (equi.ntheta * equi.ndomain)

    assert mean_f < 1e-3


#  TODO: review test once improved fitting for RlZ has been fixed
@pytest.mark.parametrize("wout_path", ("data/wout_DSHAPE.nc", "data/wout_SOLOVEV.nc"))
@pytest.mark.parametrize("ntheta", (32,))
@pytest.mark.parametrize("s", range(128))
def test_inverse_grad_shafranov_jacobian(wout_path, ntheta, s):

    equi = InverseGradShafranovEquilibrium.from_vmec(wout_path)
    equi.ntheta = ntheta
    equi.ndomain = 32
    x = equi.grid().to(torch.float64)

    x.requires_grad_()
    rho = x[:, 0]
    theta = x[:, 1]

    #  Use simply polynomials to avoid derivative issues
    RlZ = get_RlZ_from_wout(x, wout_path)

    #  Get differentiable jacobian
    R = RlZ[:, 0]
    Z = RlZ[:, 2]
    dR_dx = grad(R, x, retain_graph=True)
    Rs = dR_dx[:, 0]
    Ru = dR_dx[:, 1]
    dZ_dx = grad(Z, x, retain_graph=True)
    Zs = dZ_dx[:, 0]
    Zu = dZ_dx[:, 1]
    jacobian = R * (Ru * Zs - Zu * Rs)
    #  Get jacobian in VMEC flux coordinates
    jacobian /= 2 * rho
    jacobian = jacobian.view(-1, equi.ntheta)

    #  Get VMEC jacobian
    wout = get_wout(wout_path)
    gmnc = torch.as_tensor(wout["gmnc"][:]).clone()
    with torch.no_grad():
        vmec_jacobian = ift(gmnc, basis="cos", ntheta=theta[: equi.ntheta])

    #  Get quantity at the same radial location
    rho_idx = int(s / 128 * jacobian.shape[0])
    rho = rho[:: equi.ntheta][rho_idx]

    ns = vmec_jacobian.shape[0]
    phi = torch.zeros(ns)
    phi[1:] = torch.linspace(0, 1, ns)[:-1] + 1 / (2 * ns)

    phi_idx = (phi - rho**2).abs().argmin()

    #  Skip if radial location does not match
    if (phi[phi_idx] - rho**2).abs() > 1e-3:
        return

    #  Skip if on axis
    if phi[phi_idx] == 0:
        return

    assert torch.allclose(
        jacobian[rho_idx], vmec_jacobian[phi_idx], rtol=1e-2, atol=0
    ), f"mae={((jacobian[rho_idx] - vmec_jacobian[phi_idx])/vmec_jacobian[phi_idx]).abs().mean():.2e} at rho={rho:.2f}"
