"""Train script."""

import math
import torch
import matplotlib.pyplot as plt

from physics import HighBetaEquilibrium
from utils import log_gradients

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


def train(niter: int, seed: int = 42):

    #  Set seed
    torch.manual_seed(seed)

    equi = HighBetaEquilibrium()
    x = equi.get_collocation_points(kind="grid")
    x.requires_grad_()

    model = MLP(a=equi.a, psi_0=equi.psi_0)
    model.train()

    learning_rate = 3e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log_every_n_iter = 500

    for t in range(niter):

        psi_hat = model(x)
        loss = equi.closure(x, psi_hat)

        if t % log_every_n_iter == log_every_n_iter - 1:
            print(
                f"iter={t:4d}, "
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

    #  Check solution on a grid
    x = equi.get_collocation_points(kind="grid")

    #  Get model solution
    model.eval()
    psi_hat = model(x).detach()
    if equi.normalized:
        psi_hat *= equi.psi_0

    #  Analytical solution
    psi = equi.psi(x)

    #  Plot magnetic flux
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    equi.fluxplot(x, psi, ax, linestyles="solid")
    equi.fluxplot(x, psi_hat, ax, linestyles="dashed")

    #  Plot scatter plot
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.scatter(psi_hat.view(-1), psi.view(-1))
    ax.plot([psi.min(), psi.max()], [psi.min(), psi.max()], "r--", linewidth=2)

    #  Plot 1D slices
    #  TODO: fix me!
    # fig, ax = plt.subplots(1, 1, tight_layout=True)
    # thetas_to_plot = torch.linspace(0, int(equi.ns / 2) - 1, 3, dtype=int)
    # for i in thetas_to_plot:
    #     ax.plot(x[theta][:, 0], psi[:, i], "-", label=f"true theta={thetas[i]:.2f}")
    #     ax.plot(x[theta][:, 0], psi_hat[:, i], "--", label=f"pred theta={thetas[i]:.2f}")
    # ax.legend()

    plt.show()


if __name__ == "__main__":
    train(niter=10000)
