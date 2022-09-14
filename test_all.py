"""Test file."""

import pytest
import torch
import numpy as np

from models import GradShafranovMLP
from physics import (
    HighBetaEquilibrium,
    GradShafranovEquilibrium,
    InverseGradShafranovEquilibrium,
    Inverse3DMHD,
)
from utils import grad, get_profile_from_wout, ift, ift_2D, get_RlZ_from_wout, get_wout

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


#  TODO: improve interpolation for lmns and zmns
@pytest.mark.parametrize("wout_path", ("data/wout_DSHAPE.nc", "data/wout_SOLOVEV.nc"))
@pytest.mark.parametrize("ntheta", (32,))
@pytest.mark.parametrize("js", range(256))
@pytest.mark.parametrize("xmn", ("rmnc", "lmns", "zmns"))
def test_get_RlZ_from_wout(wout_path, ntheta, xmn, js):

    wout = get_wout(wout_path)

    equi = InverseGradShafranovEquilibrium.from_vmec(wout_path)
    equi.ntheta = ntheta

    ns = wout["ns"][:].data.item()

    #  Skip tests if we have reached the boundary
    if js >= ns:
        return

    #  Create VMEC radial grid in phi
    if xmn == "lmns":
        phi = torch.zeros(ns)
        phi[1:] = torch.linspace(0, 1, ns)[1:] - 0.5 / (ns - 1)
    else:
        phi = torch.linspace(0, 1, ns)
    rho = torch.sqrt(phi)

    #  Set grid as from VMEC
    grid = equi.grid()
    grid[:, 0] = rho.repeat_interleave(equi.ntheta)
    grid = grid.to(torch.float64)

    #  Get differentiable quantity
    RlZ = get_RlZ_from_wout(grid, wout_path)
    if xmn == "rmnc":
        x = RlZ[:, 0]
    elif xmn == "lmns":
        x = RlZ[:, 1]
    elif xmn == "zmns":
        x = RlZ[:, 2]
    x = x.view(-1, equi.ntheta)

    #  Get quantity from VMEC
    basis = "cos" if xmn == "rmnc" else "sin"
    vmec_x = ift(
        torch.as_tensor(wout[xmn][:]).clone(),
        basis=basis,
        ntheta=grid[: equi.ntheta, 1],
    )

    #  Check if quantiies are below 1e-3 m for R and Z,
    #  and below 1e-3 rad for lambda
    assert torch.allclose(
        x[js], vmec_x[js], rtol=0, atol=1e-3
    ), f"mae={(x[js] - vmec_x[js]).abs().mean():.2e} at rho={rho[js]:.4f}"


@pytest.mark.parametrize("wout_path", ("data/wout_DSHAPE.nc", "data/wout_SOLOVEV.nc"))
@pytest.mark.parametrize("ntheta", (32,))
def test_inverse_grad_shafranov_pde_closure(wout_path, ntheta):

    equi = InverseGradShafranovEquilibrium.from_vmec(wout_path)
    equi.ntheta = ntheta
    x = equi.grid().to(torch.float64)

    #  Do not use axis and boundary
    x = x[equi.ntheta : -equi.ntheta, :]
    x.requires_grad_()

    RlZ = get_RlZ_from_wout(x, wout_path)

    fsq = equi.pde_closure(x, RlZ).item()
    mean_f = np.sqrt(fsq) / (equi.ntheta * equi.ndomain)

    assert mean_f < 1e-3


#  TODO: improve jacobian, especially close to the axis
@pytest.mark.parametrize("wout_path", ("data/wout_DSHAPE.nc", "data/wout_SOLOVEV.nc"))
@pytest.mark.parametrize("ntheta", (32,))
@pytest.mark.parametrize("js", range(1, 256))
def test_inverse_grad_shafranov_jacobian(wout_path, ntheta, js):

    wout = get_wout(wout_path)

    equi = InverseGradShafranovEquilibrium.from_vmec(wout_path)
    equi.ntheta = ntheta

    ns = wout["ns"][:].data.item()

    #  Skip tests if we have reached the boundary
    if js >= ns:
        return

    #  Create VMEC radial grid in phi
    #  Half-mesh since jacobian is computed on half-mesh
    phi = torch.zeros(ns)
    phi[1:] = torch.linspace(0, 1, ns)[1:] - 0.5 / (ns - 1)
    rho = torch.sqrt(phi)

    #  Set grid as from VMEC
    x = equi.grid()
    x[:, 0] = rho.repeat_interleave(equi.ntheta)
    x = x.to(torch.float64)
    x.requires_grad_()

    rho = x[:, 0]
    theta = x[:, 1]

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
        vmec_jacobian = ift_2D(gmnc, basis="cos", ntheta=theta[: equi.ntheta])

    assert torch.allclose(
        jacobian[js], vmec_jacobian[js], atol=1e-2, rtol=0
    ), f"mae={(jacobian[js] - vmec_jacobian[js]).abs().mean():.2e} at rho={rho[::equi.ntheta][js]:.4f}"


#  TODO: fix lmns[js=1]
@pytest.mark.parametrize("wout_path", ("data/wout_HELIOTRON.nc",))
@pytest.mark.parametrize("ntheta", (16,))
@pytest.mark.parametrize("nzeta", (18,))
@pytest.mark.parametrize("xmn", ("rmnc", "lmns", "zmns"))
def test_get_3DRlZ_from_wout(wout_path, ntheta, nzeta, xmn):

    wout = get_wout(wout_path)

    ns = wout["ns"][:].data.item()
    nfp = wout["nfp"][:].data.item()

    #  Create VMEC radial grid in phi
    if xmn == "lmns":
        phi = torch.zeros(ns)
        phi[1:] = torch.linspace(0, 1, ns)[1:] - 0.5 / (ns - 1)
    else:
        phi = torch.linspace(0, 1, ns)

    #  Define grid from VMEC
    rho = torch.sqrt(phi)
    theta = (2 * torch.linspace(0, 1, ntheta) - 1) * torch.pi
    zeta = (torch.linspace(0, 1, nzeta)) * torch.pi / nfp

    grid = torch.cartesian_prod(rho, theta, zeta).to(torch.float64)

    #  Get differentiable quantity
    RlZ = get_RlZ_from_wout(grid, wout_path)
    if xmn == "rmnc":
        x = RlZ[:, 0]
    elif xmn == "lmns":
        x = RlZ[:, 1]
    elif xmn == "zmns":
        x = RlZ[:, 2]
    x = x.view(-1, ntheta, nzeta)

    #  Get quantity from VMEC
    #  Put this into an optimized ift using VMEC scheme
    xm = torch.from_numpy(wout["xm"][:].data)
    xn = torch.from_numpy(wout["xn"][:].data)
    angle = (
        theta[:, None, None] * xm[None, None, :]
        - zeta[None, :, None] * xn[None, None, :]
    )
    if xmn == "rmnc":
        tzmn = torch.cos(angle)
    else:
        tzmn = torch.sin(angle)
    wout_xmn = torch.as_tensor(wout[xmn][:]).clone()
    vmec_x = torch.einsum("stzf,sf->stz", tzmn[None, ...], wout_xmn)

    for js in range(ns):
        #  lambda is not defined on axis
        if xmn == "lmns" and js in (0,):
            continue
        assert torch.allclose(
            x[js], vmec_x[js], rtol=0, atol=1e-4
        ), f"mae={(x[js] - vmec_x[js]).abs().mean():.2e} at rho={rho[js]:.4f}"


# @pytest.mark.parametrize("ns", (21,))
# @pytest.mark.parametrize("ntheta", (16,))
# @pytest.mark.parametrize("nzeta", (18,))
#  TODO: increase grid size
@pytest.mark.parametrize(
    "wout_path, ns, ntheta, nzeta",
    [("data/wout_HELIOTRON.nc", 64, 48, 38), ("data/wout_W7X.nc", 64, 16, 32)],
)
def test_inverse_3d_pde_closure(wout_path, ns, ntheta, nzeta):

    equi = Inverse3DMHD.from_vmec(wout_path)
    equi.ns = ns
    equi.ntheta = ntheta
    equi.nzeta = nzeta
    x = equi.grid().to(torch.float64)

    #  Do not use axis and boundary
    x = x[equi.ntheta * equi.nzeta : -equi.ntheta * equi.nzeta, :]
    x.requires_grad_()

    RlZ = get_RlZ_from_wout(x, wout_path)

    fsq = equi.pde_closure(x, RlZ).item()
    mean_f = np.sqrt(fsq) / (equi.ns * equi.ntheta * equi.nzeta)

    assert mean_f < 1e-3


# TODO: as expected, the VMEC jacobian close to the axis is ~1e-2 away
#       however, for the great majority of the plasma volume,
#       the difference is ~1e-6
@pytest.mark.parametrize("wout_path", ("data/wout_HELIOTRON.nc", "data/wout_W7X.nc"))
@pytest.mark.parametrize("ntheta", (16,))
@pytest.mark.parametrize("nzeta", (18,))
def test_inverse_3D_jacobian(wout_path, ntheta, nzeta, js_min=1, atol=1e-3):

    wout = get_wout(wout_path)

    ns = wout["ns"][:].data.item()
    nfp = wout["nfp"][:].data.item()

    # VMEC radial coordinate (VMEC calculates jacobian on half-mesh)
    phi = torch.zeros(ns)
    phi[1:] = torch.linspace(0, 1, ns)[1:] - 0.5 / (ns - 1)

    #  Define grid from VMEC
    rho = torch.sqrt(phi)
    theta = (2 * torch.linspace(0, 1, ntheta) - 1) * torch.pi
    zeta = (torch.linspace(0, 1, nzeta)) * torch.pi / nfp
    x = torch.cartesian_prod(rho, theta, zeta).to(torch.float64)
    x.requires_grad_(True)

    #  Get differentiable jacobian
    RlZ = get_RlZ_from_wout(x, wout_path)
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
    jacobian /= 2 * x[:, 0]
    jacobian = jacobian.nan_to_num()
    jacobian = jacobian.view(-1, ntheta, nzeta)

    #  Get VMEC jacobian
    gmnc = torch.as_tensor(wout["gmnc"][:]).clone()
    xm_nyq = torch.as_tensor(wout["xm_nyq"][:]).clone()
    xn_nyq = torch.as_tensor(wout["xn_nyq"][:]).clone()

    angle = (
        theta[:, None, None] * xm_nyq[None, None, :]
        - zeta[None, :, None] * xn_nyq[None, None, :]
    )
    costzmn = torch.cos(angle)
    vmec_jacobian = torch.einsum("stzf,sf->stz", costzmn[None, ...], gmnc)

    for js in range(js_min, ns):
        assert torch.allclose(
            jacobian[js], vmec_jacobian[js], atol=atol, rtol=0
        ), f"mae={(jacobian[js] - vmec_jacobian[js]).abs().mean():.2e} at rho={rho[js]:.4f}"


@pytest.mark.parametrize("wout_path", ("data/wout_HELIOTRON.nc", "data/wout_W7X.nc"))
@pytest.mark.parametrize("ntheta", (16,))
@pytest.mark.parametrize("nzeta", (18,))
@pytest.mark.parametrize("ns", (11,))
def test_inverse_3D_equif(wout_path, ntheta, nzeta, ns, atol=5e-3):
    """
    Compute and compare equif.

    Achtung: this is not equif as defined in VMEC,
    but <mu0 fs> when fbeta=0.
    """

    wout = get_wout(wout_path)

    vmec_ns = wout["ns"][:].data.item()
    nfp = wout["nfp"][:].data.item()

    equi = Inverse3DMHD.from_vmec(wout_path)
    equi.ns, equi.ntheta, equi.nzeta = ns, ntheta, nzeta

    #  VMEC radial coordinate (VMEC calculates jacobian on half-mesh)
    phi = torch.zeros(vmec_ns)
    phi[1:] = torch.linspace(0, 1, vmec_ns)[1:] - 0.5 / (vmec_ns - 1)

    #  Select subset of flux surfaces to spare memory
    #  Avoid magnetic axis
    jss = torch.arange(vmec_ns)
    jss = jss[torch.linspace(1, vmec_ns - 1, ns, dtype=int)]
    phi = phi[jss]

    #  Define grid from VMEC
    rho = torch.sqrt(phi)
    theta = (2 * torch.linspace(0, 1, ntheta + 1)[:-1] - 1) * torch.pi
    zeta = (torch.linspace(0, 1, nzeta + 1))[:-1] * 2 * torch.pi / nfp
    x = torch.cartesian_prod(rho, theta, zeta).to(torch.float64)
    x.requires_grad_(True)

    RlZ = get_RlZ_from_wout(x, wout_path)
    fs = equi.f_rho(x, RlZ)

    #  Define utility functions
    def gradient(tensor: torch.Tensor, factor: float = 1.5):
        output = torch.empty_like(tensor)
        hs = 1 / (tensor.size(0) - 1)
        output[1:-1] = (tensor[2:] - tensor[1:-1]) / hs
        #  Pad output
        output[0] = factor * output[1] - (factor - 1) * output[2]
        output[-1] = factor * output[-2] - (factor - 1) * output[-3]
        return output

    def to_full_mesh(tensor: torch.Tensor, factor: float = 1.5):
        tensorf = torch.empty_like(tensor)
        tensorf[1:-1] = 0.5 * (tensor[1:-1] + tensor[2:])
        tensorf[0] = factor * tensor[1] - (factor - 1) * tensor[2]
        tensorf[-1] = factor * tensor[-1] - (factor - 1) * tensor[-2]
        return tensorf

    vmec_equif = torch.from_numpy(wout["equif"][:].data)

    mu0 = 4 * torch.pi * 1e-7
    twopi = 2 * torch.pi

    signgs = torch.from_numpy(wout["signgs"][:].data)
    phiedge = torch.from_numpy(wout["phi"][-1].data)
    iotaf = torch.from_numpy(wout["iotaf"][:].data)
    pres = torch.from_numpy(wout["pres"][:].data) * mu0

    phipf = phiedge / twopi / signgs

    presgrad = gradient(pres, factor=2.0)

    vp = torch.from_numpy(wout["vp"][:].data)
    vpf = to_full_mesh(vp)

    bsubu = torch.from_numpy(wout["bsubumnc"][:, 0].data)
    bsubv = torch.from_numpy(wout["bsubvmnc"][:, 0].data)

    jcuru = -gradient(bsubv, factor=2.0) * signgs
    jcurv = gradient(bsubu, factor=2.0) * signgs

    #  First make sure that equif in torch is equal to equif from vmec
    equif = phipf * iotaf * jcurv - phipf * jcuru + presgrad * vpf
    equif /= abs(phipf * iotaf * jcurv) + abs(phipf * jcuru) + abs(presgrad) * vpf
    equif[0] = 2.0 * equif[1] - equif[2]
    equif[-1] = 2.0 * equif[-2] - equif[-3]

    assert torch.allclose(equif, vmec_equif, atol=1e-13, rtol=0)
    del equif

    #  Now compute f
    vmec_fs = phipf * iotaf * jcurv - phipf * jcuru + presgrad * vpf
    vmec_fs[0] = 2.0 * vmec_fs[1] - vmec_fs[2]
    vmec_fs[-1] = 2.0 * vmec_fs[-2] - vmec_fs[-3]

    #  Select computed flux surfaces
    vmec_fs = vmec_fs[jss]

    for js in range(len(jss)):
        assert torch.allclose(
            fs[js], vmec_fs[js], atol=atol, rtol=0
        ), f"mae={(fs[js] - vmec_fs[js]).abs():.2e} at rho={rho[js]:.4f}"


@pytest.mark.parametrize("wout_path", ("data/wout_HELIOTRON.nc", "data/wout_W7X.nc"))
@pytest.mark.parametrize("ntheta", (16,))
@pytest.mark.parametrize("nzeta", (18,))
@pytest.mark.parametrize("ns", (11,))
def test_J_components(wout_path, ntheta, nzeta, ns, atol=1e-1):

    """
    jcuru - Jsuptheta
    jcurv - Jsupzeta
    - both are given on full mesh. See:
    https://github.com/jonathanschilling/vmec-internals/blob/master/vmec_var_names.pdf

    why is there not jcurs?? Can we make VMEC output jcurs?
    """

    wout = get_wout(wout_path)

    # VMEC NS NFP
    vmec_ns = wout["ns"][:].data.item()
    nfp = wout["nfp"][:].data.item()

    equi = Inverse3DMHD.from_vmec(wout_path)
    equi.ns, equi.ntheta, equi.nzeta = ns, ntheta, nzeta

    #  VMEC radial coordinate (VMEC calculates jacobian on half-mesh)
    phi = torch.zeros(vmec_ns)
    phi[1:] = torch.linspace(0, 1, vmec_ns)[1:] - 0.5 / (vmec_ns - 1)

    #  Select subset of flux surfaces to spare memory
    #  Avoid magnetic axis
    jss = torch.arange(vmec_ns)
    jss = jss[torch.linspace(1, vmec_ns - 1, ns, dtype=int)]
    phi = phi[jss]

    #  Define grid from VMEC
    rho = torch.sqrt(phi)
    theta = (2 * torch.linspace(0, 1, ntheta + 1)[:-1] - 1) * torch.pi
    zeta = (torch.linspace(0, 1, nzeta + 1))[:-1] * 2 * torch.pi / nfp
    x = torch.cartesian_prod(rho, theta, zeta).to(torch.float64)
    x.requires_grad_(True)

    RlZ = get_RlZ_from_wout(x, wout_path)
    Jsuptheta, Jsupzeta = equi.Jsup(x, RlZ)

    #  Define utility functions
    def gradient(tensor: torch.Tensor, factor: float = 1.5):
        output = torch.empty_like(tensor)
        hs = 1 / (tensor.size(0) - 1)
        output[1:-1] = (tensor[2:] - tensor[1:-1]) / hs
        #  Pad output
        output[0] = factor * output[1] - (factor - 1) * output[2]
        output[-1] = factor * output[-2] - (factor - 1) * output[-3]
        return output

    # signgs = torch.from_numpy(wout["signgs"][:].data)
    # bsubu = torch.from_numpy(wout["bsubumnc"][:, 0].data)
    # bsubv = torch.from_numpy(wout["bsubvmnc"][:, 0].data)
    # jcuru = -gradient(bsubv, factor=2.0) * signgs
    # jcurv = gradient(bsubu, factor=2.0) * signgs

    jcuru = torch.from_numpy(wout["jcuru"][:].data)
    jcurv = torch.from_numpy(wout["jcurv"][:].data)

    # mu0 = 4 * torch.pi * 1e-7
    # Jsuptheta /= mu0
    # Jsupzeta /= mu0

    for js in range(len(jss)):
        assert torch.allclose(
            Jsupzeta[js], jcurv[js], atol=atol, rtol=0
        ), f"jsupv mae={(Jsupzeta[js] - jcurv[js]).abs():.2e} at rho={rho[js]:.4f}"
        assert torch.allclose(
            Jsuptheta[js], jcuru[js], atol=atol, rtol=0
        ), f"jsupu mae={(Jsuptheta[js] - jcuru[js]).abs():.2e} at rho={rho[js]:.4f}"
        print(f"rho={rho[js]:.4f} passed")


@pytest.mark.parametrize("wout_path", ("data/wout_HELIOTRON.nc", "data/wout_W7X.nc"))
@pytest.mark.parametrize("ntheta", (16,))
@pytest.mark.parametrize("nzeta", (18,))
@pytest.mark.parametrize("ns", (11,))
@pytest.mark.parametrize("B", ("B_zeta", "B_theta"))
def test_B_components(wout_path, ntheta, nzeta, ns, B):

    """
    In VMEC: buco bvco
    """
    pass
