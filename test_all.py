"""Test file."""

import pytest
import torch

from models import GradShafranovMLP
from physics import HighBetaEquilibrium
from utils import grad, get_profile_from_wout, ift

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
    assert mae > 0 and mae < 1e-5


@pytest.mark.parametrize("normalized", (True, False))
@pytest.mark.parametrize("nbatches", (1, 5))
@pytest.mark.parametrize("ns", (10,))
def test_high_beta_points_different_than_grid(normalized: bool, nbatches: int, ns: int):
    equi = HighBetaEquilibrium(normalized=normalized, ndomain=ns**2)
    grid = equi.grid(ns=ns)
    for _, (x, _) in zip(range(nbatches), equi):
        #  The first point is on the axis in both cases
        assert (x[1:] != grid[1:]).all()


@pytest.mark.parametrize("normalized", (True, False))
@pytest.mark.parametrize("nbatches", (1, 5))
def test_high_beta_consistent_points(normalized: bool, nbatches: int):
    equi = HighBetaEquilibrium(normalized=normalized)
    domain_points = []
    boundary_points = []
    for _, (x_domain, x_boundary) in zip(range(nbatches), equi):
        domain_points.append(x_domain)
        boundary_points.append(x_boundary)
    for i, (x_domain, x_boundary) in zip(range(nbatches), equi):
        assert (domain_points[i] == x_domain).all()
        assert (boundary_points[i] == x_boundary).all()
    #  Points should be different if we change the equilibrium seed
    equi.seed = equi.seed - 1
    for i, (x_domain, x_boundary) in zip(range(nbatches), equi):
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
