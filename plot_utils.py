import numpy as np
import matplotlib.pyplot as plt


def plot_psi_collocationpoints(psi, grid, axis_guess=None):
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

