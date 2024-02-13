"""Main train script."""

import argparse
import math
from copy import copy
from typing import Dict, Any

import yaml
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec

from models import (
    HighBetaMLP,
    GradShafranovMLP,
    InverseGradShafranovMLP,
    Inverse3DMHDMLP,
)
from physics import (
    HighBetaEquilibrium,
    GradShafranovEquilibrium,
    InverseGradShafranovEquilibrium,
    Inverse3DMHD,
)
from utils import (
    log_gradients,
    mae,
    get_flux_surfaces_from_wout,
    get_3d_flux_surfaces_from_wout,
)

torch.set_default_tensor_type(torch.DoubleTensor)

_TRUE_COLOR = "#af8dc3"
_PREDICTED_COLOR = "#7fbf7b"

def get_equilibrium_and_model(**equi_kws):

    _equi_kws = copy(equi_kws)
    target = _equi_kws.pop("_target_")

    model_kws = {}

    if target == "high-beta":
        equi = HighBetaEquilibrium(**_equi_kws)
        if not equi.normalized:
            model_kws = {"a": equi.a, "psi_0": equi.psi_0}
        model = HighBetaMLP(**model_kws)
        return equi, model

    if target == "grad-shafranov":
        equi = GradShafranovEquilibrium(**_equi_kws)
        if not equi.normalized:
            model_kws = {
                "R0": equi.Rb[0],
                "a": equi.Rb[1],
                "b": equi.Zb[1],
                "psi_0": equi.psi_0,
            }
        model = GradShafranovMLP(**model_kws)
        return equi, model

    if target == "inverse-grad-shafranov":
        equi = InverseGradShafranovEquilibrium(**_equi_kws)
        if not equi.normalized:
            model_kws = {"Rb": equi.Rb, "Zb": equi.Zb}
        model = InverseGradShafranovMLP(**model_kws)
        return equi, model

    if target == "inverse-3d-mhd":
        equi = Inverse3DMHD(**_equi_kws)
        if not equi.normalized:
            model_kws = {
                "Rb": equi.Rb,
                "Zb": equi.Zb,
                "Ra": equi.Ra,
                "Za": equi.Za,
                "nfp": equi.nfp,
                "mpol": equi.max_mpol,
                "ntor": equi.max_ntor,
                "sym": equi.sym,
            }
        model = Inverse3DMHDMLP(**model_kws)
        return equi, model

    raise RuntimeError("Equilibrium " + target + " is not supported")


def train(
    seed: int,
    max_epochs: int,
    nbatches: int,
    log_every_n_steps: int,
    update_axis_every_n_epochs: int,
    learning_rate: float,
    equilibrium: Dict[str, Any],
):
    """
    Train model to solve the given equilibrium and plot solution.

    Args:
        seed (int): random seed.
        max_epochs (int): number of epochs to train.
        nbatches (int): number of batches.
        log_every_n_steps (int): frequency for training log.
        update_axis_every_n_epochs: frequency for updating axis guess.
            Valid only for GradShafranov equilibria.
        learning_rate (float): model learning rate.
        equilibrium (dict): dict of keyword arguments for equilibrium and model
            definition. Dict must have a `_target_` key to define the equilibrium.
    """

    #  Set seed
    torch.manual_seed(seed)

    ###############
    # Instantiate #
    ###############

    #  Get equilibrium and model
    equi, model = get_equilibrium_and_model(seed=seed, **equilibrium)
    model.train()

    #  shorthand
    target = equilibrium["_target_"]

    # TODO: define optimizer in config
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=learning_rate,
        tolerance_grad=0,
        tolerance_change=0,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=learning_rate,
    # )

    #########
    # Train #
    #########

    for epoch in range(max_epochs):
        for batch_idx, (x_domain, x_boundary, x_axis) in zip(range(nbatches), equi):

            x_domain.requires_grad_()
            x_boundary.requires_grad_()

            def closure():
                optimizer.zero_grad()
                loss = equi.closure(
                    x_domain,
                    model(x_domain),
                    x_boundary,
                    model(x_boundary),
                    x_axis,
                    model(x_axis) if x_axis is not None else None,
                )
                loss.backward()
                return loss

            # loss = closure()
            # for param in model.parameters():
            #     torch.nn.utils.clip_grad_norm_(param, 1e-4)
            # optimizer.step()
            # TODO: step for L-BFGS-B
            optimizer.step(closure)
            scheduler.step()

            #  Print the current loss:
            #  this is the loss at the given step and not the
            #  loss aggregated over all batches.
            global_step = epoch * nbatches + batch_idx
            if global_step % log_every_n_steps == log_every_n_steps - 1:
                # TODO: remove me
                rmnl_grad = f"rmnl_g={(model.rmnl.grad[model.rmnl != 0] / model.rmnl[model.rmnl != 0]).abs().mean():.2e}"
                lmnl_grad = f"lmnl_g={(model.lmnl.grad[model.lmnl != 0] / model.lmnl[model.lmnl != 0]).abs().mean():.2e}"
                zmnl_grad = f"zmnl_g={(model.zmnl.grad[model.zmnl != 0] / model.zmnl[model.zmnl != 0]).abs().mean():.2e}"
                optimizer.zero_grad()
                loss = equi.closure(
                    x_domain,
                    model(x_domain),
                    x_boundary,
                    model(x_boundary),
                    x_axis,
                    model(x_axis) if x_axis is not None else None,
                    return_dict=True,
                )
                string = f"[{epoch:5d}/{max_epochs:5d}][{batch_idx:3d}/{nbatches:3d}]"
                for k, v in loss.items():
                    string += f", {k}={v.item():.2e}"
                for k, v in zip(
                    ("rmnl_g", "lmnl_g", "zmnl_g"), (rmnl_grad, lmnl_grad, zmnl_grad)
                ):
                    string += ", " + v
                # string += f", lr={scheduler.get_last_lr()[0]:.2e}"
                print(string)

            #  Update running axis guess
            if target == "grad-shafranov":
                if epoch % update_axis_every_n_epochs == update_axis_every_n_epochs - 1:
                    if equi.psi_0 > 0:
                        psi = "min"
                    else:
                        psi = "max"
                    axis_guess = model.find_x_of_psi(psi, x_axis)
                    if equi.normalized:
                        axis_guess *= equi.Rb[0]
                    equi.update_axis(axis_guess[0])
                    string = (
                        f"[{epoch:5d}/{max_epochs:5d}][{batch_idx:3d}/{nbatches:3d}]"
                    )
                    string += f", update axis guess to [{axis_guess[0][0]:.2f}, {axis_guess[0][1]:.2f}]"
                    print(string)

    #############
    # Visualize #
    #############

    #  Get solution on test collocation points on a regular grid
    x = equi.grid()
    if target == "inverse-grad-shafranov":
        #  Do not include axis to avoid `nan` in eps() computation
        x = x[equi.ntheta :]
    x.requires_grad_()
    psi_hat = model(x)

    #  Compute normalized residual error
    pde_mae = equi.mae_pde_loss(x, psi_hat)
    print(f"pde mae={pde_mae:.2e}")

    #  Compute the normalized averaged force
    if target == "grad-shafranov" or target == "inverse-grad-shafranov":
        eps = equi.eps(x, psi_hat)
        print(f"eps={eps:.2e}")

    #  Scale model solution
    if equi.normalized:
        psi_hat *= equi.psi_0

    #  Get grid points
    equi.ns = 99
    equi.ntheta = 32
    equi.nzeta = 36
    grid = equi.grid(normalized=False)

    has_analytical_solution = target == "high-beta" or (
        target == "grad-shafranov" and equi.is_solovev
    )

    #  Compute mae between model solution and analytical solution
    if has_analytical_solution:
        psi = equi.psi(grid)
        psi_mae = mae(psi_hat.detach(), psi)
        print(f"psi mae={psi_mae:.2e}")

    #  Plot magnetic flux
    if target == "inverse-grad-shafranov":
        fig, ax = plt.subplots(1, 1, tight_layout=True)

        RlZ_hat = model(grid)
        equi.fluxsurfacesplot(
            RlZ_hat[:, [0, 2]], ax, linestyle="dashed", interpolation="linear"
        )
    elif target == "inverse-3d-mhd":
        # model
        RlZ_hat = model(grid)
        # VMEC
        range_r = (RlZ_hat[:, 0].min(), RlZ_hat[:, 0].max())
        range_z = (RlZ_hat[:, 2].min(), RlZ_hat[:, 2].max())
        if equi.wout_path is not None:
            rz, phi = get_3d_flux_surfaces_from_wout(
                equi.wout_path, nzeta=equi.nzeta, ntheta=equi.ntheta
            )
            range_r = (
                min(range_r[0], rz[:, 0].min()) * 0.9,
                max(range_r[1], rz[:, 0].max()) * 1.1,
            )
            range_z = (
                min(range_z[0], rz[:, 1].min()) * 1.1,
                max(range_z[1], rz[:, 1].max()) * 1.1,
            )

        nrows = int(math.sqrt(equi.nzeta))
        ncols = math.ceil(equi.nzeta / nrows)
        fig = plt.figure(
            tight_layout=True, figsize=(ncols * 5, nrows * 4), dpi=nrows * 100
        )
        axs = []
        gs = GridSpec(nrows=nrows, ncols=ncols)
        for i, zeta in enumerate(torch.linspace(0, equi.nzeta - 1, 4, dtype=int)):
            axs.append(fig.add_subplot(gs[int(zeta / ncols), int(zeta % ncols)]))
            equi.fluxsurfacesplot(
                RlZ_hat[:, [0, 2]],
                axs[zeta],
                zeta=zeta,
                linestyle="dashed",
                interpolation="linear",
            )
            if equi.wout_path is not None:
                equi.fluxsurfacesplot(
                    rz.clone(),
                    ax,
                    phi=torch.linspace(0, 1, phi.shape[0]),
                    nplot=4,
                    zeta=zeta,
                    interpolation="linear",
                    linestyle="solid",
                    color=_TRUE_COLOR,
                    label="VMEC" if i == 0 else None,
                    linewidth=3,
                )
            equi.fluxsurfacesplot(
                RlZ_hat[:, [0, 2]],
                ax,
                zeta=zeta,
                phi=torch.linspace(0, 1, equi.ns)**2,
                nplot=4,
                linestyle="dashed",
                interpolation="linear",
                color=_PREDICTED_COLOR,
                label="NN" if i == 0 else None,
                linewidth=3,
            )

            if equi.wout_path is not None:
                equi.fluxsurfacesplot(
                    rz.clone(),
                    axs[zeta],
                    phi=torch.linspace(0, 1, phi.shape[0]),
                    zeta=zeta,
                    interpolation="linear",
                    linestyle="solid",
                )
        for zeta in range(equi.nzeta):
            axs[zeta].set_xlim(range_r[0].detach().numpy(), range_r[1].detach().numpy())
            axs[zeta].set_ylim(range_z[0].detach().numpy(), range_z[1].detach().numpy())

    else:
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        equi.fluxplot(grid, psi_hat, ax, linestyles="dashed")
    if has_analytical_solution:
        #  Plot analytical solution
        equi.fluxplot(grid, psi, ax, linestyles="solid")
    if target == "grad-shafranov" and equi.wout_path is not None:
        #  Plot VMEC flux surfaces
        rz, psi = get_flux_surfaces_from_wout(equi.wout_path)
        equi.fluxsurfacesplot(rz, ax, psi=psi, ns=psi.shape[0])
    if target == "inverse-grad-shafranov" and equi.wout_path is not None:
        #  Plot VMEC flux surfaces
        rz, psi = get_flux_surfaces_from_wout(equi.wout_path)
        equi.fluxsurfacesplot(
            rz, ax, phi=torch.linspace(0, 1, psi.shape[0]), interpolation="linear"
        )

    #  Plot scatter plot
    if target == "high-beta":
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        _, _, _, im = ax.hist2d(psi_hat.tolist(), psi.tolist(), bins=50, cmin=1)
        ax.plot([psi.min(), psi.max()], [psi.min(), psi.max()], "r--", linewidth=2)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\hat{\Psi}$")
        ax.set_ylabel(r"$\Psi$")

    #  Plot eps over entire domain
    if target == "grad-shafranov":
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        equi.fluxplot(
            x,
            equi.eps(x, psi_hat, reduction=None),
            ax,
            filled=True,
            locator=ticker.LogLocator(),
        )
    if target == "inverse-grad-shafranov":
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        equi.fluxsurfacesplot(
            psi_hat[:, [0, 2]],
            ax,
            # scalar=equi.eps(x, psi_hat, reduction=None),
            phi=x[:: equi.ntheta, 0] ** 2,
            contourf_kwargs={"locator": ticker.LogLocator()},
        )
    if target == "inverse-3d-mhd":
        for idx in torch.linspace(0, equi.nzeta - 1, 4):
            fig, ax = plt.subplots(1, 1, tight_layout=True)
            # Reduce resolution to avoid memory issues.
            equi.ns = 11
            equi.plot_pde_loss_on_cross_section(ax=ax, model=model, zeta_index=int(idx))
            fig.savefig(f"w7x-eps-{int(idx)}.pdf", dpi=300)

    #  Show figures
    plt.show()

    # if target == "inverse-3d-mhd":
    #     fig, ax = plt.subplots(1, 1, tight_layout=True)
    #     equi.plot_force_history(ax)
    #     plt.show()
    #     fig, ax = plt.subplots(1, 1, tight_layout=True)
    #     equi.plot_pde_loss_on_rho_surface(fig, ax, rho=0.5, model=model)
    #     plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/inverse_w7x.yaml",
        help="Configuration file to use",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(args)
    train(**config)
