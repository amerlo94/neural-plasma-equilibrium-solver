"""Train script."""

import math
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker

from models import HighBetaMLP, GradShafranovMLP
from physics import HighBetaEquilibrium, GradShafranovEquilibrium
from utils import log_gradients, mae, get_flux_surfaces_from_wout

torch.set_default_tensor_type(torch.DoubleTensor)


def train(equilibrium: str, nepochs: int, normalized: bool, seed: int = 42):

    assert equilibrium in ("high-beta", "grad-shafranov")

    torch.manual_seed(seed)

    #  TODO: implement me argparse or make me cleaner
    params = {"normalized": normalized, "seed": seed}
    if equilibrium == "high-beta":
        equi = HighBetaEquilibrium(**params)
        params = {}
        if not equi.normalized:
            params = {"a": equi.a, "psi_0": equi.psi_0}
        model = HighBetaMLP(**params)
    else:
        equi = GradShafranovEquilibrium(**params)
        params = {}
        if not equi.normalized:
            params = {
                "R0": equi.Ra,
                "a": equi.Rb[1],
                "b": equi.Zb[1],
                "psi_0": equi.psi_0,
            }
        model = GradShafranovMLP(**params)

    model.train()

    learning_rate = 1e-1
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=learning_rate,
        tolerance_grad=0,
        tolerance_change=0,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )

    #  Number of optimizer steps per epoch (i.e., number of batches)
    nsteps = 1
    log_every_n_steps = 10

    #  Frequency for axis guess update
    update_axis_every_n_epochs = 20

    for e in range(nepochs):
        for s, (x_domain, x_boundary, x_axis) in zip(range(nsteps), equi):

            x_domain.requires_grad_()

            def closure():
                optimizer.zero_grad()
                loss = equi.closure(
                    x_domain,
                    model(x_domain),
                    x_boundary,
                    model(x_boundary),
                    x_axis,
                    model(x_axis),
                )
                loss.backward()
                return loss

            optimizer.step(closure)

            #  Print the current loss (not aggregated across batches)
            global_step = e * nsteps + s
            if global_step % log_every_n_steps == log_every_n_steps - 1:
                optimizer.zero_grad()
                loss = equi.closure(
                    x_domain,
                    model(x_domain),
                    x_boundary,
                    model(x_boundary),
                    x_axis,
                    model(x_axis),
                    return_dict=True,
                )
                string = f"[{e:5d}/{nepochs:5d}][{s:3d}/{nsteps:3d}]"
                for k, v in loss.items():
                    string += f", {k}={v.item():.2e}"
                print(string)

            #  Update running axis guess
            if equilibrium == "grad-shafranov":
                if e % update_axis_every_n_epochs == update_axis_every_n_epochs - 1:
                    if equi.psi_0 > 0:
                        psi = "min"
                    else:
                        psi = "max"
                    axis_guess = model.find_x_of_psi(psi, x_axis)
                    equi.update_axis(axis_guess[0])
                    string = f"[{e:5d}/{nepochs:5d}][{s:3d}/{nsteps:3d}]"
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
    if equilibrium == "grad-shafranov":
        eps = equi.eps(x, psi_hat)
        print(f"eps={eps:.2e}")

    #  Scale model solution
    if equi.normalized:
        psi_hat *= equi.psi_0

    #  Get grid points
    grid = equi.grid(normalized=False)

    has_analytical_solution = equilibrium == "high-beta" or (
        equilibrium == "grad-shafranov" and equi.is_solovev
    )

    #  Compute mae between model solution and analytical solution
    if has_analytical_solution:
        psi = equi.psi(grid)
        psi_mae = mae(psi_hat.detach(), psi)
        print(f"psi mae={psi_mae:.2e}")

    #  Plot magnetic flux
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    equi.fluxplot(grid, psi_hat, ax, linestyles="dashed")
    if has_analytical_solution:
        #  Plot analytical solution
        equi.fluxplot(grid, psi, ax, linestyles="solid")
    if equilibrium == "grad-shafranov" and equi.wout_path is not None:
        #  Plot VMEC flux surfaces
        rz, psi = get_flux_surfaces_from_wout(equi.wout_path)
        equi.fluxsurfacesplot(rz, ax, psi=psi, ns=psi.shape[0])

    #  Plot scatter plot
    if equilibrium == "high-beta":
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        _, _, _, im = ax.hist2d(psi_hat.tolist(), psi.tolist(), bins=50, cmin=1)
        ax.plot([psi.min(), psi.max()], [psi.min(), psi.max()], "r--", linewidth=2)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\hat{\Psi}$")
        ax.set_ylabel(r"$\Psi$")

    #  Plot eps over entire domain
    if equilibrium == "grad-shafranov":
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
    #  TODO: add argparse with default configuration
    train(equilibrium="grad-shafranov", normalized=False, nepochs=200)
