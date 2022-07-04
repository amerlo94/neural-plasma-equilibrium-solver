from math import cos, sin
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize as mpl_norm


def plot_boundary_2D():
    mpol=3

    def Rb_fn(theta, Rb):
        basis = np.cos([i * theta for i in range(mpol)])
        return (Rb * basis).sum()

    def Zb_fn(theta, Zb):
        basis = np.sin([i * theta for i in range(mpol)])
        return (Zb * basis).sum()

    Rb, Zb = [(3.51, 1., 0.106)], [(0, 1.47, -0.16)]

    for i in range(0):
        Rb.append([i + np.random.normal(scale=0.05) for i in Rb[0]])
        Zb.append([i + np.random.normal(scale=0.05) for i in Zb[0]])

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    theta = np.linspace(-np.pi, np.pi, 10)

    for i, (r, z) in enumerate(zip(Rb, Zb)):
        nr, nz = [], []
        for t in theta:
            nr.append(Rb_fn(t, Rb=r))
            nz.append(Zb_fn(t, Zb=z))
        ax.plot(nr, nz, alpha=1, color="blue")

    # [11.0, 10.623489801858733, 9.777479066043686, 9.099031132097581, 9.099031132097581,
    #  9.777479066043686, 10.623489801858733, 11.0]
    # z
    # [-1.0, -0.6234898018587334, 0.22252093395631445, 0.9009688679024193, 0.9009688679024193,
    #  0.22252093395631445, -0.6234898018587334, -1.0]

    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.ylim(-1.75, 1.75)
    plt.xlim(2., 6.)
    plt.show()


def plot_boundary_3D():

    NFP = 19

    # ((m,n), Xmn)
    Rb = (
        ((0, 0), 10.),
        ((1, 0), -1.),
        ((1, 1), -0.3),
        ((-1, -1), 0.3)
    )

    Zb = (
        ((-1, 0), 1.),
        ((-1, 1), -0.3),
        ((1, -1), -0.3)
    )

    theta = np.linspace(-np.pi, np.pi, 50)
    zeta = np.linspace(0, 2 * np.pi / NFP, 5)

    print(theta[0] / np.pi * 180, "\t<= theta <=\t", theta[-1] / np.pi * 180)
    print(zeta[0] / np.pi * 180, "\t<= zeta  <=\t", zeta[-1] / np.pi * 180)

    def get_fourier_basis(theta, zeta, n: int, m: int, nfp: int):
        if m >= 0:
            if n >= 0:
                return cos(abs(m) * theta) * cos(abs(n) * nfp * zeta)
            else:
                return cos(abs(m) * theta) * sin(abs(n) * nfp * zeta)
        else:
            if n >= 0:
                return sin(abs(m) * theta) * cos(abs(n) * nfp * zeta)
            else:
                return sin(abs(m) * theta) * sin(abs(n) * nfp * zeta)

    def b_fn(theta, zeta, xb):
        x_boundary = np.array([
            xmn * get_fourier_basis(theta=theta, zeta=zeta,
                                    n=n, m=m, nfp=NFP)
            for (m, n), xmn in xb
        ])
        x_sum = x_boundary.sum()
        return x_sum

    def plot_single_zeta(rb, zb, title=""):
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.plot(rb, zb, alpha=1, color="blue")
        plt.xlabel("R [m]")
        plt.ylabel("Z [m]")
        plt.title(title)
        ax.axis('equal')
        # plt.ylim(-1.75, 1.75)
        # plt.xlim(2., 6.)
        plt.show()

    xbs, ybs, zbs, rbs = [], [], [], []
    for ze in zeta:
        cosze = cos(ze)
        sinze = sin(ze)
        z, x, y = [], [], []
        re = []
        for t in theta:
            r = b_fn(theta=t, zeta=ze, xb=Rb)
            re.append(r)
            x.append(r * cosze)
            y.append(r * sinze)
            z.append(b_fn(theta=t, zeta=ze, xb=Zb))
        zbs.append(z)
        xbs.append(x)
        ybs.append(y)
        rbs.append(re)
    xbs = np.array(xbs)
    ybs = np.array(ybs)
    zbs = np.array(zbs)
    rbs = np.array(rbs)

    # mpl.use('macosx')

    for i, zi in enumerate(zeta):
        plot_single_zeta(rbs[i], zbs[i], title=zi)

    # Z = np.outer(zbs.T, zbs)
    # X, Y = np.meshgrid(xbs, ybs)
    #
    # color_dimension = Y  # change to desired fourth dimension
    # minn, maxx = color_dimension.min(), color_dimension.max()
    # norm = mpl_norm(minn, maxx)
    # m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    # m.set_array([])
    # fcolors = m.to_rgba(color_dimension)
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx,
    #                 shade=False)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()

    # nr
    # [2.6159999999999997, 2.7623622637137166, 3.2367444045297633, 3.957, 4.530893331756521,
    #  4.530893331756521, 3.957, 3.2367444045297633, 2.7623622637137166, 2.6159999999999997]
    # nz
    # [-2.192117770473762e-16, -1.1024670267211665, -1.5023906198600527, -1.1344932789576148,
    #  -0.39992359313888676, 0.39992359313888676, 1.1344932789576143, 1.5023906198600527,
    #  1.1024670267211665, 2.192117770473762e-16]


if __name__ ==  '__main__':
    plot_boundary_3D()
