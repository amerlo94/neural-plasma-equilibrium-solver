import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from utils import ift, grad, mae


mu0 = 4 * math.pi * 1e-7


class Equilibrium(IterableDataset):
    def __init__(
        self,
        ndomain: int = 2500,
        nboundary: int = 50,
        normalized: bool = False,
        seed: int = 42,
    ) -> None:

        super().__init__()

        #  Number of collocation points to use in the domain
        self.ndomain = ndomain

        #  Number of collocation points to use on the boundary
        self.nboundary = nboundary

        #  Whether to use the normalized PDE system
        self.normalized = normalized

        #  Seed to initialize the random generators
        self.seed = seed

        #  Closure functions
        if normalized:
            self.pde_closure = self._pde_closure_
            self.boundary_closure = self._boundary_closure_
            self.axis_closure = self._axis_closure_
            self.data_closure = self._data_closure_
            self.mae_pde_loss = self._mae_pde_loss_
        else:
            self.pde_closure = self._pde_closure
            self.boundary_closure = self._boundary_closure
            self.axis_closure = self._axis_closure
            self.data_closure = self._data_closure
            self.mae_pde_loss = self._mae_pde_loss

    def closure(
        self,
        x_domain: Tensor,
        psi_domain: Tensor,
        x_boundary: Tensor,
        psi_boundary: Tensor,
        x_axis: Optional[Tensor] = None,
        psi_axis: Optional[Tensor] = None,
        return_dict: Optional[bool] = False,
    ) -> Tensor:
        loss = {"pde": self.pde_closure(x_domain, psi_domain),
                "boundary": self.boundary_closure(x_boundary, psi_boundary)}
        loss["tot"] = loss["pde"] + loss["boundary"]
        if x_axis is not None:
            loss["axis"] = self.axis_closure(x_axis, psi_axis)
            loss["tot"] += loss["axis"]
        if return_dict:
            return loss
        return loss["tot"]

    def grid(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def fluxplot(self, *args, **kwargs):
        raise NotImplementedError

    def _data_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _pde_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _axis_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _data_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _axis_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _mae_pde_loss(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _mae_pde_loss_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError


class HighBetaEquilibrium(Equilibrium):
    def __init__(
        self, a: float = 0.1, A: float = 1, C: float = 10, R0: float = 0.6, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.a = a
        self.A = A
        self.C = C
        self.R0 = R0
        self.psi_0 = -2 * A * a**2 / 8

    def __iter__(self):

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.normalized:
            rho_b = 1.0
        else:
            rho_b = self.a

        while True:
            #  Domain collocation points
            rho = torch.empty(self.ndomain)
            rho[0] = 0
            rho[1:] = torch.rand(self.ndomain - 1, generator=generator) * rho_b
            theta = (2 * torch.rand(self.ndomain, generator=generator) - 1) * math.pi
            domain = torch.stack([rho, theta], dim=-1)
            #  Boundary collocation points
            rho = rho_b * torch.ones(self.nboundary)
            theta = (2 * torch.rand(self.nboundary, generator=generator) - 1) * math.pi
            boundary = torch.stack([rho, theta], dim=-1)
            yield domain, boundary, None

    def psi(self, x: Tensor) -> Tensor:
        rho = x[:, 0]
        theta = x[:, 1]
        return (
            0.125
            * (rho**2 - self.a**2)
            * (2 * self.A + self.C * rho * torch.cos(theta))
        )

    def psi_(self, x: Tensor) -> Tensor:
        rho = x[:, 0]
        theta = x[:, 1]
        return (
            0.125
            * self.a**2
            * (rho**2 - 1)
            * (2 * self.A + self.C * rho * self.a * torch.cos(theta))
            / self.psi_0
        )

    def _data_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.psi(x)) ** 2).sum()

    def _data_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.psi_(x)) ** 2).sum()

    def _pde_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dtheta = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dtheta2 = grad(dpsi_dtheta, x, create_graph=True)[:, 1]
        rho = x[:, 0]
        theta = x[:, 1]
        #  The one below is the original formulation with axis singularity
        # residual = 1 / rho * dpsi_drho + dpsi2_drho2
        # residual += 1 / rho ** 2 * dpsi2_dtheta2
        # residual -= A + C * rho * torch.cos(theta)
        residual = rho * dpsi_drho + rho**2 * dpsi2_drho2 + dpsi2_dtheta2
        residual -= rho**2 * (self.A + self.C * rho * torch.cos(theta))
        return (residual**2).sum()

    def _mae_pde_loss(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dtheta = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dtheta2 = grad(dpsi_dtheta, x, create_graph=True)[:, 1]
        rho = x[:, 0]
        theta = x[:, 1]
        residual = rho * dpsi_drho + rho**2 * dpsi2_drho2 + dpsi2_dtheta2
        denom = rho**2 * (self.A + self.C * rho * torch.cos(theta))
        #  TODO: avoid to compute error at the boundary to avoid division by 0
        return mae(residual[rho != self.a], denom[rho != self.a])

    def _pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dtheta = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dtheta2 = grad(dpsi_dtheta, x, create_graph=True)[:, 1]
        rho = x[:, 0]
        theta = x[:, 1]
        residual = rho * dpsi_drho + rho**2 * dpsi2_drho2 + dpsi2_dtheta2
        residual -= (
            self.a**2
            / self.psi_0
            * rho**2
            * (self.A + self.a * self.C * rho * torch.cos(theta))
        )
        return (residual**2).sum()

    def _mae_pde_loss_(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dtheta = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dtheta2 = grad(dpsi_dtheta, x, create_graph=True)[:, 1]
        rho = x[:, 0]
        theta = x[:, 1]
        residual = rho * dpsi_drho + rho**2 * dpsi2_drho2 + dpsi2_dtheta2
        denom = (
            self.a**2
            / self.psi_0
            * rho**2
            * (self.A + self.a * self.C * rho * torch.cos(theta))
        )
        #  TODO: avoid to compute error at the boundary to avoid division by 0
        return mae(residual[rho != 1], denom[rho != 1])

    def _boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        rho = x[:, 0]
        boundary = rho == self.a
        return (psi[boundary] ** 2).sum()

    def _boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        rho = x[:, 0]
        boundary = rho == 1
        return (psi[boundary] ** 2).sum()

    def grid(self, ns: int = None, normalized: bool = None) -> Tensor:

        if normalized is None:
            normalized = self.normalized

        if ns is None:
            ns = int(math.sqrt(self.ndomain))

        if normalized:
            rho_b = 1.0
        else:
            rho_b = self.a

        rho = torch.linspace(0, rho_b, ns)
        theta = torch.linspace(-math.pi, math.pi, ns)

        return torch.cartesian_prod(rho, theta)

    def fluxplot(self, x, psi, ax, *args, **kwargs):

        xrho = x[:, 0]
        ytheta = x[:, 1]

        ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xrho = xrho.view(ns, ns)
        ytheta = ytheta.view(ns, ns)
        xx = self.R0 + xrho * torch.cos(ytheta)
        yy = xrho * torch.sin(ytheta)

        #  Detach and reshape tensors
        psi = psi.view(xx.shape)

        ax.contour(xx, yy, psi, levels=10, **kwargs)
        ax.axis("equal")

        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")

        return ax


class GradShafranovEquilibrium(Equilibrium):
    """
    The default case is a Solov'ev equilibrium as in the original VMEC paper.

    This repository keeps VMEC 2D equilibria under the `data` folder,
    they are taken from the DESC repository:

    https://github.com/PlasmaControl/DESC/tree/master/tests/inputs
    """

    def __init__(
        self,
        p: Tuple[float] = (0.125 / mu0, -0.125 / mu0),
        fsq: Tuple[float] = (4, -4 * 4 / 10),
        Rb: Tuple[float] = (
            3.9334e00,
            -1.0258e00,
            -6.8083e-02,
            -9.0720e-03,
            -1.4531e-03,
        ),
        Zb: Tuple[float] = (0, math.sqrt(10) / 2, 0, 0, 0),
        Ra: float = 3.9332,
        Za: float = 0.0,
        psi_0: float = 1,
        #  TODO: put these definitions in *.yaml files
        #  TODO: Solov'ev as from VMEC wout file
        # p: Tuple[float] = (0.125 / mu0, -0.125 / mu0),
        # fsq: Tuple[float] = (6.58, -0.142),
        # Rb: Tuple[float] = (3.999, 1.026, -0.068),
        # Zb: Tuple[float] = (0, 1.58, 0.01),
        # Ra: float = 3.999,
        # Za: float = 0.0,
        # psi_0: float = -1,
        #  TODO: DSHAPE equilibrium, to be fixed!
        # p: Tuple[float] = (
        #     1598.34455725,
        #     -2064.73509916,
        #     -309.20392583,
        #     1871.28768454,
        #     -2400.0528889,
        #     1301.74040632,
        # ),
        # fsq: Tuple[float] = (
        #     0.5840779,
        #     -0.01771422,
        #     0.02795285,
        #     -0.04720095,
        #     0.06766905,
        #     -0.03307103,
        # ),
        # Rb: Tuple[float] = (3.51, 1.0, 0.106),
        # Zb: Tuple[float] = (0, 1.47, -0.16),
        # Ra: float = 3.71270844,
        # Za: float = 0.0,
        # psi_0: float = -0.665,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        #  Pressure and current profile
        self.p = torch.as_tensor(p)
        self.fsq = torch.as_tensor(fsq)

        #  Boundary definition
        assert len(Rb) == len(Zb)
        self.Rb = torch.as_tensor(Rb)
        self.Zb = torch.as_tensor(Zb)

        if self.normalized:
            self.R0 = Ra
            self.psi = self.psi_
            self.eps = self.eps_
        else:
            self.R0 = 1
            self.psi = self.psi
            self.eps = self.eps

        #  Initial guess for the axis
        self.Ra = Ra / self.R0
        self.Za = Za / self.R0
        #  Running axis location
        self._Ra = self.Ra
        self._Za = self.Za

        #  Boundary condition on psi (i.e., psi_edge), the poloidal flux (chi in VMEC)
        self.psi_0 = psi_0

    @property
    def _mpol(self) -> int:
        return len(self.Rb)

    def p_fn(self, psi):
        psi_ = psi / self.psi_0
        p = 0
        for i, coef in enumerate(self.p):
            p += coef * psi_**i
        return p

    def fsq_fn(self, psi):
        psi_ = psi / self.psi_0
        fsq = 0
        for i, coef in enumerate(self.fsq):
            fsq += coef * psi_**i
        return fsq

    def get_ab(self):
        # minor radius a, in Z direction called b, for running axis estimate self._Ra
        # only in case of Solov'ev equilibrium

        l2 = self.fsq[0]  # R0**2 in VMEC
        f0 = -self.fsq[1] / 4 / l2  # beta1 in VMEC
        p0 = self.p[0] * mu0  # beta0 in VMEC

        R_min = math.sqrt(self._Ra ** 2 - self.psi_0 * math.sqrt(8 / p0))
        R_max = math.sqrt(self._Ra ** 2 + self.psi_0 * math.sqrt(8 / p0))

        l2 = math.sqrt(l2)  # R0
        Z_max = (self.psi_0 / l2) * (1 / math.sqrt(f0))
        Z_min = - (self.psi_0 / l2) * (1 / math.sqrt(f0))

        a = (R_max - R_min) / 2
        b = (Z_max - Z_min) / 2

        return a, b

    def __iter__(self):

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        while True:
            #  Domain collocation points
            #  Create grid by scaling the boundary from the LCFS to the axis
            #  Achtung: these are not flux surfaces!
            domain = []
            ns = int(math.sqrt(self.ndomain))
            #  TODO: use random point and speed up theta grid computation with ift
            #  TODO: Normalized version does not work with random hs, why?
            # hs = torch.rand(ns, generator=generator) ** 2
            hs = torch.linspace(0, 1, ns + 2)[1:-1] ** 2
            for s in hs:
                theta = (2 * torch.rand(ns, generator=generator) - 1) * math.pi
                # theta = (2 * torch.linspace(0, 1, ns)) * math.pi
                Rb = torch.as_tensor([self.Rb_fn(t) for t in theta])
                Zb = torch.as_tensor([self.Zb_fn(t) for t in theta])
                if self.normalized:
                    Rb = Rb / self.R0
                    Zb = Zb / self.R0
                    R = ((Rb - self._Ra) * s) / self.R0 + self._Ra
                    Z = ((Zb - self._Za) * s) / self.R0 + self._Za
                else:
                    R = (Rb - self._Ra) * s + self._Ra
                    Z = (Zb - self._Za) * s + self._Za
                domain.append(torch.stack([R, Z], dim=-1))
            domain = torch.cat(domain)
            #  Boundary collocation points
            theta = (2 * torch.rand(self.nboundary, generator=generator) - 1) * math.pi
            R = torch.as_tensor([self.Rb_fn(t) for t in theta])
            Z = torch.as_tensor([self.Zb_fn(t) for t in theta])
            if self.normalized:
                R = R / self.R0
                Z = Z / self.R0
            boundary = torch.stack([R, Z], dim=-1)
            #  Axis point
            axis = torch.Tensor([self._Ra, self._Za]).view(1, 2)
            #  TODO: is it ok to have the same number of points as boundary, but effective always the same?
            #        the same can be achieved with a factor in front of the loss
            # axis = axis.expand(self.nboundary, 2)
            yield domain, boundary, axis

    def Rb_fn(self, theta):
        basis = torch.cos(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Rb * basis).sum()

    def Zb_fn(self, theta):
        basis = torch.sin(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Zb * basis).sum()

    def update_axis(self, axis_guess):
        #  Axis should have Za=0 by symmetry
        self._Ra = axis_guess[0]

    def eps(self, x: Tensor, psi: Tensor) -> Tensor:
        # TODO this needs to be normalized as well
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, retain_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, retain_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, retain_graph=True)
        fsq = self.fsq_fn(psi)
        dfsq_dpsi = grad(fsq, psi, retain_graph=True)
        R = x[:, 0]
        #  Force components
        gs = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        gs += mu0 * R**2 * dp_dpsi + 0.5 * dfsq_dpsi
        fR = -1 / (mu0 * R**2) * dpsi_dR * gs
        fZ = -1 / (mu0 * R**2) * dpsi_dZ * gs
        fsq = fR**2 + fZ**2
        #  grad-p
        gradpsq = dp_dpsi**2 * (dpsi_dR**2 + dpsi_dZ**2)
        #  Compute the normalized averaged force balance
        #  The `x` domain is not enforced to be on a equally spaced grid,
        #  so the sum() here is not strictly an equivalent to an integral
        return torch.sqrt(fsq.sum() / gradpsq.sum())

    def eps_(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, retain_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, retain_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, retain_graph=True)
        fsq = self.fsq_fn(psi)
        dfsq_dpsi = grad(fsq, psi, retain_graph=True)
        R = x[:, 0]
        gs_ = - dpsi_dR + R * dpsi2_dR2 + R * dpsi2_dZ2
        gs_ += R * (self.R0 ** 2 / self.psi_0 ** 2) * \
            (mu0 * self.R0**2 * R**2 * dp_dpsi + 0.5 * dfsq_dpsi)
        fR = - 1 / (mu0 * self.R0 ** 2 * R**2) * dpsi_dR * gs_
        fZ = - 1 / (mu0 * self.R0 ** 2 * R**2) * dpsi_dZ * gs_
        fsq = fR ** 2 + fZ ** 2
        gradpsq = dp_dpsi**2 * (dpsi_dR**2 + dpsi_dZ**2)
        return torch.sqrt(fsq.sum() / gradpsq.sum())

    def _pde_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, create_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, create_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, create_graph=True)
        fsq = self.fsq_fn(psi)
        dfsq_dpsi = grad(fsq, psi, create_graph=True)
        R = x[:, 0]
        residual = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        residual += mu0 * R**2 * dp_dpsi + 0.5 * dfsq_dpsi
        return (residual**2).sum()

    def _mae_pde_loss(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, retain_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, retain_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, retain_graph=True)
        fsq = self.fsq_fn(psi)
        dfsq_dpsi = grad(fsq, psi, retain_graph=True)
        R = x[:, 0]
        nabla_star = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        denom = mu0 * R**2 * dp_dpsi + 0.5 * dfsq_dpsi
        return mae(nabla_star, -denom)

    def _boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.psi_0) ** 2).sum()

    def _axis_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        return (psi**2).sum()

    # def _pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
    #     # 2010 Cerfon, Freidberg a = (Rmax-Rmin)/2 ; b = (Zmax-Zmin)/2
    #     dpsi_dx = grad(psi, x, create_graph=True)
    #     dpsi_drho = dpsi_dx[:, 0]
    #     dpsi_dZ = dpsi_dx[:, 1]
    #     dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
    #     dpsi2_dZ2 = grad(dpsi_dZ, x, create_graph=True)[:, 1]
    #     p = self.p_fn(psi)
    #     dp_dpsi = grad(p, psi, create_graph=True)
    #     f = self.fsq_fn(psi)
    #     dfsq_dpsi = grad(f, psi, create_graph=True)
    #     rho = x[:, 0]
    #     residual = self.b**2 * (rho * dpsi2_drho2 - dpsi_drho) + self.a**2 * rho * dpsi2_dZ2
    #     residual += rho * (self.a**2 * self.b**2 / self.psi_0**2) * (rho**2 * mu0 * self.a**2 * dp_dpsi + 0.5 * dfsq_dpsi)
    #     return (residual ** 2).sum()

    def _pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        # 2010 Cerfon, Freidberg a = (Rmax-Rmin)/2 = b
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, create_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, create_graph=True)
        f = self.fsq_fn(psi)
        dfsq_dpsi = grad(f, psi, create_graph=True)
        rho = x[:, 0]
        residual = - dpsi_drho + rho * dpsi2_drho2 + rho * dpsi2_dZ2
        residual += rho * (self.R0**2 / self.psi_0**2) * (mu0 * self.R0**2 * rho ** 2 * dp_dpsi +
                                                     0.5 * dfsq_dpsi)
        return (residual ** 2).sum()

    def _mae_pde_loss_(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, create_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, create_graph=True)
        f = self.fsq_fn(psi)
        dfsq_dpsi = grad(f, psi, create_graph=True)
        rho = x[:, 0]
        nabla_star = dpsi_drho + rho * (dpsi2_drho2 + dpsi2_dZ2)
        denom = rho * (self.R0 ** 2 / self.psi_0 ** 2) * \
                (mu0 * self.R0 ** 2 * rho ** 2 * dp_dpsi +
                                  0.5 * dfsq_dpsi)
        return mae(nabla_star, -denom)

    def _boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - 1.0) ** 2).sum()

    def _axis_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        return (psi**2).sum()

    def psi(self, x: Tensor) -> Tensor:
        """
        See Bauer1978 for the nomenclature.

        Achtung: this is the analytical solution only in case of a Solov'ev equilibrium.
        Could be improved: https://arxiv.org/pdf/1210.2113.pdf section 6.1
        """
        R = x[:, 0]
        Z = x[:, 1]
        l2 = self.fsq[0]  # R0**2 in VMEC
        f0 = -self.fsq[1] / 4 / l2  # beta1 in VMEC
        p0 = self.p[0] * mu0  # beta0 in VMEC
        #  Get axis from R at boundary
        Ra = math.sqrt(self.Rb.sum().item() ** 2 + math.sqrt(8 / p0))
        return f0 * l2 * Z**2 + p0 / 8 * (R**2 - Ra**2) ** 2

    def psi_(self, x: Tensor) -> Tensor:
        """
        Normalized analytical solution in Solov'ev equilibrium
        """
        y = x[:, 1]
        x = x[:, 0]

        l2 = self.fsq[0]  # R0**2 in VMEC
        f0 = -self.fsq[1] / 4 / l2  # beta1 in VMEC
        p0 = self.p[0] * mu0 # beta0 in VMEC

        Ra = math.sqrt(self.Rb.sum().item()**2 + math.sqrt(8/p0))
        return (f0 * l2 * y ** 2 * self.R0**2 + p0 / 8 * (x**2 * self.R0**2 - Ra ** 2) ** 2) / self.psi_0

    def grid(self, ns: int = None, normalized: bool = None) -> Tensor:

        if ns is None:
            ns = int(math.sqrt(self.ndomain))

        Rb = ift(self.Rb, basis="cos", ntheta=ns)
        Zb = ift(self.Zb, basis="sin", ntheta=ns)

        grid = []

        #  Create grid by linearly scaling the boundary from the LCFS to the axis
        #  Achtung: these are not flux surfaces!
        hs = 1 / (ns - 1)
        for i in range(ns):
            if normalized:
                Rb = Rb / self.R0
                Zb = Zb / self.R0
                R = ((Rb - self._Ra) * i * hs) / self.R0 + self._Ra
                Z = ((Zb - self._Za) * i * hs) / self.R0 + self._Za
            else:
                R = ((Rb - self._Ra * self.R0) * i * hs) + self._Ra * self.R0
                Z = ((Zb - self._Za * self.R0) * i * hs) + self._Za * self.R0

            grid.append(torch.stack([R, Z], dim=-1))

        return torch.cat(grid)

    def fluxplot(self, x, psi, ax, *args, **kwargs):

        R = x[:, 0]
        Z = x[:, 1]

        ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xx = R.view(ns, ns)
        yy = Z.view(ns, ns)

        #  Detach and reshape tensors
        psi = psi.view(xx.shape)

        cs = ax.contour(xx, yy, psi, levels=10, **kwargs)
        ax.clabel(cs, inline=True, fontsize=10, fmt="%1.3f")
        ax.axis("equal")

        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")

        return ax

    def fluxsurfacesplot(
        self,
        x,
        ax,
        psi: Optional[Tensor] = None,
        ns: Optional[int] = None,
        nplot: Optional[int] = 10,
    ):
        """
        Plot flux surfaces on (R, Z) plane.

        TODO: improve ns and nplot handling.
        """

        assert len(x.shape) == 2

        if ns is None:
            #  Infer number of flux surfaces
            ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xx = x[:, 0].view(ns, -1)
        yy = x[:, 1].view(ns, -1)

        if nplot > ns:
            nplot = ns

        #  Plot nplot + 1 since the first one is the axis
        ii = torch.linspace(0, ns - 1, nplot + 1, dtype=torch.int).tolist()
        #  If psi is given, pick equally spaced flux surfaces in terms of psi
        if psi is not None:
            psi_i = torch.linspace(0, psi[-1], nplot + 1)
            ii = []
            for p in psi_i:
                idx = torch.argmin((psi - p).abs())
                ii.append(idx)

        for i in ii:
            ax.plot(xx[i], yy[i])
            if psi is not None:
                pi_half = int(xx.shape[1] / 4)
                ax.text(xx[i][pi_half], yy[i][pi_half], f"{psi[i].item():.3f}")
        ax.axis("equal")

        return ax


class InverseGSEquilibrium(Equilibrium):

    def __init__(
        self,
        # DSHAPE from DESC
        # https://github.com/PlasmaControl/DESC/blob/master/examples/DESC/DSHAPE
        p: Tuple[float] =  (1.6e+3, -3.2e+3, 1.6e+3),
        iota: Tuple[float] = (1, -0.67),
        Rb: Tuple[float] = (
            3.51,
            1,
            0.106, 0, 0
        ),
        Zb: Tuple[float] = (0, 1.47, -0.16, 0, 0),
        psi_0: float = 1,
        Ra: float = 3.71270844,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        #  Pressure and current profile
        self.p = torch.as_tensor(p)
        self.iota = torch.as_tensor(iota)
        self.Ra = Ra

        # psi_boundary
        self.psi_0 = psi_0

        #  Boundary definition
        assert len(Rb) == len(Zb)
        self.Rb = torch.as_tensor(Rb)
        self.Zb = torch.as_tensor(Zb)

    @property
    def _mpol(self) -> int:
        return len(self.Rb)

    def __iter__(self):

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        while True:
            # collocation points

            # domain
            ns = int(math.sqrt(self.ndomain))
            theta = (2 * torch.rand(ns, generator=generator) - 1) * math.pi
            rho = torch.linspace(0, 1, ns+2)[1:-1] # ** 2
            # r = torch.rand(ns, generator=generator) ** 2
            domain = torch.cartesian_prod(rho, theta)

            # boundary
            theta = (2 * torch.rand(ns, generator=generator) - 1) * math.pi
            rho = torch.ones_like(theta)
            boundary = torch.stack([rho, theta], dim=-1)

            # axis
            theta = (2 * torch.rand(ns, generator=generator) - 1) * math.pi
            rho = torch.zeros_like(theta)
            axis = torch.stack([rho, theta], dim=-1)

            yield domain, boundary, axis

    def Rb_fn(self, theta):
        basis = torch.cos(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Rb * basis).sum()

    def Zb_fn(self, theta):
        basis = torch.sin(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Zb * basis).sum()

    def p_fn(self, rho):
        # p = p * rho^l l=0,2,4,...
        # rho = sqrt(s) = sqrt(psi/psi_0)
        p = 0
        for i, coef in enumerate(self.p):
            p += coef * rho**(i*2)
        return p

    def iota_fn(self, rho):
        # iota = iota * rho^l l=0,2,4,...
        # rho = sqrt(s) = sqrt(psi/psi_0)
        iota = 0
        for i, coef in enumerate(self.iota):
            iota += coef * rho**(i*2)
        return iota * 2 * torch.pi

    def _lam(self, theta):
        # \lambda = f(\rho, \theta, \zeta)
        # return lambda as the free coordinate
        # lambda = 0 for now, theta-theta s.t. torch AD registers operation
        lam = theta - theta
        return lam

    def _jacobian_debug(self, x, rtheta):
        dR_drtheta = grad(x[:, 0], rtheta, create_graph=True)
        dZ_drtheta = grad(x[:, 1], rtheta, create_graph=True)
        dR_drho = dR_drtheta[:, 0]
        dR_dtheta = dR_drtheta[:, 1]
        dZ_drho = dZ_drtheta[:, 0]
        dZ_dtheta = dZ_drtheta[:, 1]
        g_theta2 = dR_dtheta * dR_dtheta + dZ_dtheta * dZ_dtheta
        g_rhotheta = dR_drho * dR_dtheta + dZ_drho * dZ_dtheta
        jacobian = self._jacobian(x[:, 0], dR_drho=dR_drho, dR_dtheta=dR_dtheta,
                                  dZ_drho=dZ_drho, dZ_dtheta=dZ_dtheta)
        print(f"sqrt(g) = {jacobian}; g_rt = {g_rhotheta}; g_tt = {g_theta2}")

    def _jacobian(self, R, dR_drho, dR_dtheta, dZ_drho, dZ_dtheta):
        # area element of metric tensor in axisymmetric toroidal domain
        # eq. 17a&17b of Hirshman&Whitson 1983
        # dR_drtheta = grad(x[:, 0], rtheta, create_graph=True)
        # dZ_drtheta = grad(x[:, 1], rtheta, create_graph=True)
        # dR_drho = dR_drtheta[:, 0]
        # dR_dtheta = dR_drtheta[:, 1]
        # dZ_drho = dZ_drtheta[:, 0]
        # dZ_dtheta = dZ_drtheta[:, 1]
        # jaco = R * (dR_dtheta * dZ_drho - dR_drho * dZ_dtheta)
        # if torch.min(jaco) * torch.max(jaco) < 0:
        #     print(torch.min(jaco), torch.max(jaco))
        return R * (dR_dtheta * dZ_drho - dR_drho * dZ_dtheta)

    def _F_of_rho(self, R, jacobian):
        # dpsi_drho = psi' = self.psi_0,
        # but this excludes dpsi_drho from autograd graph
        # todo this assumes lambda=0 (or theta* = theta)
        # the sqrt(rho**2) operation exists because otherwise
        # pytorch AD engine does not pick up on the operation (rho*psi_0), why??
        # psi = (torch.sqrt(rho**2) * self.psi_0)
        # dpsi_drho = grad(psi, rho, create_graph=True)
        # dpsi_drho = torch.tensor(self.psi_0, requires_grad=True, dtype=torch.float)
        flux_term = (self.psi_0 * R**2)/jacobian
        # lambda_term = 1 + grad(self._lam(theta), theta, create_graph=True)
        return flux_term #* lambda_term

    def _mae_pde_loss(self, x: Tensor, cartesians: Tensor) -> Tensor:
        return torch.zeros(1)

    def F_covariant_rho(self, x, rtheta):
        # rho = rtheta[:, 0]
        # rtheta = torch.hstack([rtheta[:,0].unsqueeze(1),
        #                        torch.atan2(rtheta[:, 1], rtheta[:, 2]).unsqueeze(1)])
        f_iota = self.iota_fn(rtheta[:, 0])
        f_p = self.p_fn(rtheta[:, 0])
        dR_drtheta = grad(x[:, 0], rtheta, create_graph=True)
        dZ_drtheta = grad(x[:, 1], rtheta, create_graph=True)
        dR_drho = dR_drtheta[:, 0]
        dR_dtheta = dR_drtheta[:, 1]
        dZ_drho = dZ_drtheta[:, 0]
        dZ_dtheta = dZ_drtheta[:, 1]
        g_theta2 = dR_dtheta * dR_dtheta + dZ_dtheta * dZ_dtheta
        g_rhotheta = dR_drho * dR_dtheta + dZ_drho * dZ_dtheta
        jacobian = self._jacobian(x[:, 0], dR_drho=dR_drho, dR_dtheta=dR_dtheta,
                                  dZ_drho=dZ_drho, dZ_dtheta=dZ_dtheta)

        dchi_drho = self.psi_0 * f_iota # * 2 * rtheta[:, 0]
        F_of_rho = self._F_of_rho(x[:, 0], jacobian)
        dFrho_drho = grad(F_of_rho, rtheta, create_graph=True)[:, 0]
        dp_drho = grad(f_p, rtheta, create_graph=True)[:, 0]

        term1 = (dchi_drho * g_theta2) / jacobian
        term1 = grad(term1, rtheta, create_graph=True)[:, 0]
        term2 = (dchi_drho * g_rhotheta) / jacobian
        term2 = grad(term2, rtheta, create_graph=True)[:, 1]

        F_cov_rho = dchi_drho / (jacobian) * (term1 - term2) \
                    + (F_of_rho * dFrho_drho) / (torch.pow(x[:, 0], 2)) \
                    + mu0 * dp_drho
        F_cov_rho = mu0 * F_cov_rho

        # F_cov_rho = dchi_drho / (jacobian * mu0) * (term1 - term2) \
        #             + (F_of_rho * dFrho_drho) / (torch.pow(x[:, 0], 2) * mu0) \
        #             + dp_drho

        return torch.pow(F_cov_rho, 2).sum()

    def _pde_closure(self, x: Tensor, cartesians: Tensor) -> Tensor:
        return self.F_covariant_rho(rtheta=x, x=cartesians)

    def _boundary_closure(self, x: Tensor, cartesians: Tensor) -> Tensor:
        # theta = torch.atan2(x[:, 1], x[:, 2])
        theta = x[:, 1]
        Rb = torch.as_tensor([self.Rb_fn(t) for t in theta])
        Zb = torch.as_tensor([self.Zb_fn(t) for t in theta])
        return (torch.pow(Rb - cartesians[:, 0], 2) + torch.pow((Zb - cartesians[:, 1]), 2)).sum()

    def _axis_closure(self, x: Tensor, cartesians: Tensor) -> Tensor:
        # at axis (s,theta)=(0,0)
        axis_loc = torch.Tensor([self.Ra, 0]).view(1,2)
        axis_loc = axis_loc.expand_as(cartesians)
        return torch.pow(cartesians - axis_loc, 2).sum()

    def grid(self, ns: int = None, normalized: bool = None) -> Tensor:

        if ns is None:
            ns = int(math.sqrt(self.ndomain))

        theta = (2 * torch.linspace(0, 1, ns) - 1) * torch.pi
        rho = torch.pow(torch.linspace(0, 1, ns), 2)
        domain = torch.cartesian_prod(rho, theta)

        return domain

    def fluxplot(self, x, ax, *args, **kwargs):

        R, Z = x[:, 0], x[:, 1]
        ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        # xx = R.view(ns, ns)
        # yy = Z.view(ns, ns)

        # cs = ax.contour(xx, yy, )
        ax.plot(R, Z,)# color="k")

        # ax.clabel(cs, inline=True, fontsize=10, fmt="%1.3f")
        ax.axis("equal")

        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")

        return ax

    def fluxsurfacesplot(
        self,
        x,
        ax,
        psi: Optional[Tensor] = None,
        ns: Optional[int] = None,
        nplot: Optional[int] = 10,
    ):
        """
        Plot flux surfaces on (R, Z) plane.

        TODO: improve ns and nplot handling.
        """

        assert len(x.shape) == 2

        if ns is None:
            #  Infer number of flux surfaces
            ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xx = x[:, 0].view(ns, -1)
        yy = x[:, 1].view(ns, -1)

        if nplot > ns:
            nplot = ns

        #  Plot nplot + 1 since the first one is the axis
        ii = torch.linspace(0, ns - 1, nplot + 1, dtype=torch.int).tolist()
        #  If psi is given, pick equally spaced flux surfaces in terms of psi
        if psi is not None:
            psi_i = torch.linspace(0, psi[-1], nplot + 1)
            ii = []
            for p in psi_i:
                idx = torch.argmin((psi - p).abs())
                ii.append(idx)

        for i in ii:
            ax.plot(xx[i], yy[i])

        ax.axis("equal")

        return ax

    def closure(
        self,
        x_domain: Tensor,
        psi_domain: Tensor,
        x_boundary: Tensor,
        psi_boundary: Tensor,
        x_axis: Optional[Tensor] = None,
        psi_axis: Optional[Tensor] = None,
        return_dict: Optional[bool] = False,
    ) -> Tensor:
        loss = {"pde": self.pde_closure(x_domain, psi_domain),
                "boundary": self.boundary_closure(x_boundary, psi_boundary)}
        loss["tot"] = loss["pde"] + loss["boundary"]
        if x_axis is not None:
            loss["axis"] = self.axis_closure(x_axis, psi_axis)
            loss["tot"] += loss["axis"]
        if return_dict:
            return loss
        return loss["tot"]