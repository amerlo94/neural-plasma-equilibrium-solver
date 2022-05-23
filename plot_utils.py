import numpy as np
import matplotlib.pyplot as plt


def plot_psi_collocationpoints(psi, grid, axis_guess=None):
    print(f"Grid: x = [{grid[:,0].min()}, {grid[:,0].max()}] "
    f"y = [{grid[:,1].min()}, {grid[:,1].max()}]")
    print(f"Psi = [{psi.min()}, {psi.max()}]")
    grid = grid.detach().numpy()
    psi = psi.detach().numpy()
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.axis("equal")
    ax.set_xlabel(r"$R [m]$")
    ax.set_ylabel(r"$Z [m]$")
    plt.scatter([i[0] for i in grid], [i[1] for i in grid], c=psi)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Psi", rotation=270)
    if axis_guess is not None:
        ax.scatter(axis_guess[0], axis_guess[1], marker="x")
        plt.show()


def plot_grid(grid, polar=True):
    print(f"Grid: x = [{grid[:, 0].min()}, {grid[:, 0].max()}] "
    f"y = [{grid[:, 1].min()}, {grid[:, 1].max()}]")
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    if polar:
        ax.scatter(grid[:, 0].detach().numpy() * np.cos(grid[:,1].detach().numpy()),
        grid[:, 0].detach().numpy() * np.sin(grid[:,1].detach().numpy()))
    else:
        ax.scatter(grid[:, 0].detach().numpy(), grid[:, 1].detach().numpy())
        plt.show()

def plot_solution(grid, preds, ax=None, colorvar: int = 0):
    # 0 - rho , 1 - theta
    print(f"Grid: x = [{grid[:, 0].min()}, {grid[:, 0].max()}] "
    f"y = [{grid[:, 1].min()}, {grid[:, 1].max()}]")
    print(f"preds: x = [{preds[:, 0].min()}, {preds[:, 0].max()}] "
    f"y = [{preds[:, 1].min()}, {preds[:, 1].max()}]")
    grid = grid.detach().numpy()
    preds = preds.detach().numpy()
    if ax is None:
        fig, ax = plt.subplots(1, 1, tight_layout=True, dpi=400)
        ax.axis("equal")
        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")
        plt.scatter(preds[:,0], preds[:, 1], c=grid[:, colorvar], alpha=0.7)
        plt.colorbar()
        plt.show()
    else:
        ax.axis("equal")
        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")
        plt.scatter(preds[:,0], preds[:, 1], c=grid[:, colorvar], alpha=0.7)
        plt.colorbar()
        plt.show()