# Getting started

## Installation
`dolfinx_padapt` is a pure Python package, which requires DOLFINx (the main branch, which will
become v0.12) with PETSc and SLEPc, and GMSH. With these installed, run

```bash
python3 -m pip install --no-build-isolation .
```

The optional dependencies are `plot` (PyVista and Matplotlib, used by the demos), `docs` (to build
this book) and `test`:

```bash
python3 -m pip install --no-build-isolation .[docs,test]
```

The [DOLFINx docker images](https://github.com/FEniCS/dolfinx/pkgs/container/dolfinx%2Flab), e.g.
`ghcr.io/fenics/dolfinx/lab:nightly`, have all other requirements installed.

## Running the demos
The demos are Python files in the `demos` folder, which can be run in serial or in parallel:

```bash
cd demos
python3 demo_submesh.py
mpirun -n 4 python3 demo_poisson.py
```

They are [jupytext](https://jupytext.readthedocs.io) scripts in the light format, which the book
reads directly. They can be opened as notebooks with jupytext, e.g. in JupyterLab, or converted
with

```bash
jupytext --to ipynb demos/demo_submesh.py
```

## Testing
The tests check that the three approaches agree with each other and with standard DOLFINx forms on
a single space, in serial and in parallel:

```bash
python3 -m pytest tests
mpirun -n 3 python3 -m pytest tests
```

## Building the documentation
This book is built with [Jupyter Book](https://jupyterbook.org) (version 1), which executes the
demos:

```bash
jupyter-book build .
```
