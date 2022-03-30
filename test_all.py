"""Test file."""

import pytest
import torch

from physics import HighBetaEquilibrium
from utils import grad


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
