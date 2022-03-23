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
    x = equi.get_collocation_points(ns=ns)
    psi = equi.psi(x)
    #  Normalized equilibrium
    equi.normalized = True
    x_ = equi.get_collocation_points(ns=ns)
    psi_ = equi.psi_(x_)
    assert ((psi_ * equi.psi_0 - psi).abs() < 1e-9).all()


@pytest.mark.parametrize("normalized", (True, False))
@pytest.mark.parametrize("ns", (5, 10, 50))
def test_high_beta_pde_closure(normalized: bool, ns: int):
    equi = HighBetaEquilibrium(normalized=normalized)
    x = equi.get_collocation_points(ns=ns)
    x.requires_grad_()
    if normalized:
        psi = equi.psi_(x)
        residual = equi.pde_closure_(x, psi).item()
    else:
        psi = equi.psi(x)
        residual = equi.pde_closure(x, psi).item()
    assert abs(residual) < 1e-10
