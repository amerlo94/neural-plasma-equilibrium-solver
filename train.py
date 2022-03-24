"""Train script."""

import math
import torch
import matplotlib.pyplot as plt

from physics import HighBetaEquilibrium
from utils import log_gradients, mae

torch.set_default_tensor_type(torch.DoubleTensor)


##########
# Models #
##########


class MLP(torch.nn.Module):
    def __init__(self, width: int = 16, a: float = 1.0, psi_0: float = 1.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, width)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(width, 1)

        #  Initialize last bias to unit, since psi(r=0)=psi_0
        torch.nn.init.ones_(self.fc2.bias)
        torch.nn.init.normal_(self.fc2.weight, std=3e-2)

        #  Scaling parameters
        self.a = a
        self.psi_0 = psi_0

    def forward(self, x):
        rho = x[:, 0] / self.a
        theta = x[:, 1]
        #  Compute features
        x1 = rho * torch.cos(theta)
        x2 = rho * torch.sin(theta)
        #  Compute psi
        psi_hat = self.fc1(torch.stack((x1, x2), dim=1))
        psi_hat = self.tanh(psi_hat / 2)
        return self.psi_0 * self.fc2(psi_hat).view(-1)


#########
# Train #
#########


def train(niter: int, seed: int = 42, normalized: bool = True):

    #  Set seed
    torch.manual_seed(seed)

    equi = HighBetaEquilibrium(normalized=normalized)
    x = equi.get_collocation_points(kind="random")
    x.requires_grad_()

    params = {}
    if not equi.normalized:
        params = {"a": equi.a, "psi_0": equi.psi_0}
    model = MLP(**params)
    model.train()

    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log_every_n_iter = 500

    for t in range(niter):

        psi_hat = model(x)
        loss = equi.closure(x, psi_hat)

        if t % log_every_n_iter == log_every_n_iter - 1:
            print(
                f"iter={t:5d}, "
                + f"loss={loss['tot'].item():.2e}, "
                + f"pde_loss={loss['pde'].item():.2e}, "
                + f"boundary_loss={loss['boundary'].item():.2e}, "
                + f"data_loss={loss['data'].item():.2e}"
            )
            log_gradients(model, learning_rate, t)

        optimizer.zero_grad()
        loss["tot"].backward()
        optimizer.step()

    #############
    # Visualize #
    #############

    #  Get solution on test collocation points on a regular grid
    x = equi.get_collocation_points(kind="grid")
    x.requires_grad_()
    psi_hat = model(x)

    #  Compute normalized residual error
    pde_mae = equi.mae_pde_loss(x, psi_hat)
    print(f"pde mae={pde_mae:.2e}")

    #  Scale model solution
    psi_hat = psi_hat.detach()
    if equi.normalized:
        psi_hat *= equi.psi_0

    #  Analytical solution
    x = equi.get_collocation_points(kind="grid", normalized=False)
    psi = equi.psi(x)

    #  Compute mae between model solution and analytical solution
    psi_mae = mae(psi_hat, psi)
    print(f"psi mae={psi_mae:.2e}")

    #  Plot magnetic flux
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    equi.fluxplot(x, psi, ax, linestyles="solid")
    equi.fluxplot(x, psi_hat, ax, linestyles="dashed")

    #  Plot scatter plot
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    _, _, _, im = ax.hist2d(psi_hat.tolist(), psi.tolist(), bins=50, cmin=1)
    ax.plot([psi.min(), psi.max()], [psi.min(), psi.max()], "r--", linewidth=2)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\hat{\Psi}$")
    ax.set_ylabel(r"$\Psi$")
    plt.show()


if __name__ == "__main__":
    train(niter=1000)
