"""Train script."""

import math
import torch
import matplotlib.pyplot as plt

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
                "R0": equi.Rb[0],
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

    # #######
    # rz, psi = get_flux_surfaces_from_wout("data/wout_SOLOVEV.nc")
    # psi = torch.repeat_interleave(psi, int(len(rz) / len(psi)))
    # rz.requires_grad_()
    # #####

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
                string = f"[{e:5d}/{nepochs:5d}][{s:3d}/{nsteps:3d}], "
                for k, v in loss.items():
                    string += f"{k}={v.item():.2e}, "
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

    #  Scale model solution
    psi_hat = psi_hat.detach()
    if equi.normalized:
        psi_hat *= equi.psi_0

    #  Get grid points
    x = equi.grid(normalized=False)

    #  Compute mae between model solution and analytical solution
    if equilibrium == "high-beta":
        psi = equi.psi(x)
        psi_mae = mae(psi_hat, psi)
        print(f"psi mae={psi_mae:.2e}")

    #  Plot magnetic flux
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    equi.fluxplot(x, psi_hat, ax, linestyles="dashed")
    if equilibrium == "high-beta":
        #  Plot analytical solution
        equi.fluxplot(x, psi, ax, linestyles="solid")
    elif equilibrium == "grad-shafranov":
        #  Plot VMEC flux surfaces
        #  TODO: bound VMEC solution to equilibrium, get ns from object?
        # rz, psi = get_flux_surfaces_from_wout("data/wout_DSHAPE.nc")
        rz, psi = get_flux_surfaces_from_wout("data/wout_SOLOVEV.nc")
        equi.fluxsurfacesplot(rz, ax, psi=psi, ns=psi.shape[0])

    #  Plot scatter plot
    if equilibrium == "high-beta":
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        _, _, _, im = ax.hist2d(psi_hat.tolist(), psi.tolist(), bins=50, cmin=1)
        ax.plot([psi.min(), psi.max()], [psi.min(), psi.max()], "r--", linewidth=2)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\hat{\Psi}$")
        ax.set_ylabel(r"$\Psi$")

    #  Show figures
    plt.show()


if __name__ == "__main__":
    #  TODO: add argparse with default configuration
    # train(equilibrium="high-beta", normalized=True, nepochs=200)
    train(equilibrium="grad-shafranov", normalized=False, nepochs=200)
