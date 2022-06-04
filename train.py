"""Main train script."""

import argparse
import math
from copy import copy
from typing import Dict, Any

import yaml
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker

from models import HighBetaMLP, GradShafranovMLP, InverseGradShafranovMLP
from physics import (
    HighBetaEquilibrium,
    GradShafranovEquilibrium,
    InverseGradShafranovEquilibrium,
)
from utils import log_gradients, mae, get_flux_surfaces_from_wout

torch.set_default_tensor_type(torch.DoubleTensor)


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

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=learning_rate,
        tolerance_grad=0,
        tolerance_change=0,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )

    #########
    # Train #
    #########

    for epoch in range(max_epochs):
        for batch_idx, (x_domain, x_boundary, x_axis) in zip(range(nbatches), equi):

            x_domain.requires_grad_()

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

            optimizer.step(closure)

            #  Print the current loss:
            #  this is the loss at the given step and not the
            #  loss aggregated over all batches.
            global_step = epoch * nbatches + batch_idx
            if global_step % log_every_n_steps == log_every_n_steps - 1:
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
    x.requires_grad_()
    psi_hat = model(x)

    #  Compute normalized residual error
    pde_mae = equi.mae_pde_loss(x, psi_hat)
    print(f"pde mae={pde_mae:.2e}")

    #  Compute the normalized averaged force
    if target == "grad-shafranov":
        eps = equi.eps(x, psi_hat)
        print(f"eps={eps:.2e}")

    #  Scale model solution
    if equi.normalized:
        psi_hat *= equi.psi_0

    #  Get grid points
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
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    if target == "inverse-grad-shafranov":
        #  psi_hat == RlZ
        equi.fluxsurfacesplot(psi_hat[:, [0, 2]], ax, linestyle="dashed")
    else:
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
        equi.fluxsurfacesplot(rz, ax, phi=torch.linspace(0, 1, psi.shape[0]))

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

    #  Show figures
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        # default="configs/solovev.yaml",
        default="configs/inverse_dshape.yaml",
        help="Configuration file to use",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(**config)
