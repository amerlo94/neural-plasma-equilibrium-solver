# Neural Netowrk Differential Equation Plasma Equilibrium Solver

PyTorch implementation of [Neural Netowrk Differential Equation Plasma Equilibrium Solver](https://doi.org/10.1103/PhysRevLett.75.3594).

## Equilibria

The implemented equilibria are described in `physics.py`:

- `HighBetaEquilibrium`: simplified high-beta tokamak;
- `GradShafranovEquilibrium`: fixed-boundary Grad-Shafranov tokamak;
- `InverseGradShafranovEquilibrium`: fixed-boundary inverse Grad-Shafranov 2D equilibrium;

## Train

Define the equilibrium and training procedure arguments via a yaml configuration file:

```shell
python train.py --config=configs/solovev.yaml
```

Available configurations:

- `configs/solovev.yaml`: Solov'ev case as in Hirshman. The Physics of fluids 26.12 (1983): 3553-3568.
- `configs/dshape.yaml`: a D-shape tokamak equilibrium as in Dudt. Physics of Plasmas 27.10 (2020): 102513.
- `configs/high_beta.yaml`: high-beta case as in van Milligen. Physical review letters 75.20 (1995): 3594.
- `configs/inverse_solovev.yaml`: inverse Solov'ev tokamak equilibrium.
- `configs/inverse_dshape.yaml`: inverse D-shape tokamak equilibrium.

## Test

To run all tests, simply run:

```shell
pytest
```

## TODO

- [ ] fix equilibrium definition from VMEC wout (i.e., F function parsing)