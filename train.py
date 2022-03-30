"""Train script."""

import math
import torch
import matplotlib.pyplot as plt

from models import HighBetaMLP, GradShafranovMLP
from physics import HighBetaEquilibrium, GradShafranovEquilibrium
from utils import log_gradients, mae

torch.set_default_tensor_type(torch.DoubleTensor)


def train(equilibrium: str, nepochs: int, normalized: bool, seed: int = 42):

    assert equilibrium in ("high-beta", "grad-shafranov")

    torch.manual_seed(seed)

    params = {"normalized": normalized, "seed": seed}
    if equilibrium == "high-beta":
        equi = HighBetaEquilibrium(**params)
        params = {}
        if not equi.normalized:
            params = {"a": equi.a, "psi_0": equi.psi_0}
        model = HighBetaMLP(**params)
    else:
        equi = GradShafranovEquilibrium(**params)
        if not equi.normalized:
            #  TODO: a = Rb[1]? psi_0 == phi_edge?
            params = {
                "R0": equi.Rb[0],
                "a": equi.Rb[1],
                "b": equi.Zb[1],
                "psi_0": equi.phi_edge,
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

    nsteps = 1
    log_every_n_steps = 10

    for e in range(nepochs):
        for s, (x_domain, x_boundary) in zip(range(nsteps), equi):

            x_domain.requires_grad_()
            x_boundary.requires_grad_()

            def closure():
                optimizer.zero_grad()
                loss = equi.closure(
                    x_domain, model(x_domain), x_boundary, model(x_boundary)
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
                    return_dict=True,
                )
                print(
                    f"[{e:5d}/{nepochs:5d}][{s:3d}/{nsteps:3d}], "
                    + f"loss={loss['tot'].item():.2e}, "
                    + f"pde_loss={loss['pde'].item():.2e}, "
                    + f"boundary_loss={loss['boundary'].item():.2e}, "
                )

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
        equi.fluxplot(x, psi, ax, linestyles="solid")

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
    #  TODO: add argparse or hydra
    # train(equilibrium="high-beta", normalized=True, nepochs=200)
    train(equilibrium="grad-shafranov", normalized=False, nepochs=200)
