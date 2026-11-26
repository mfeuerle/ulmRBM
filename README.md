# ulmRBM

The documentation can be found [here](https://mfeuerle.github.io/ulmRBM/).
Was developed using `python=3.12.13`, `numpy=2.4.3`, `scipy=1.17.1` and `fenics-dolfinx=0.10` (optional for building full-order models)

## Installing

To install this project:

```
conda env create -f environment.yml
conda activate ulmRBM
```

## Building the Docs

```
cd docs
make html
```

Then open `docs/_build/html/index.html`

## Running tests

Use `pytest`

## Package structure

Each subpackage (`affine`, `fom`, `rom`, `fenicsx`, ...) exposes its public API in `__init__.py`, which also holds the documentation of the subpackage. The implementation lives in private modules, following these conventions:

- private modules use a single leading underscore and are named after their content, e.g. `fom/_operators.py`, `fom/_models.py`, `fom/_timestepping.py`, never after the package itself,
- counterparts in different subpackages use the same name, e.g. `fom/_models.py` and `rom/_models.py`, `fom/_timestepping.py` and `rom/_timestepping.py`,
- public submodules (e.g. `fenicsx/utils.py`, `fenicsx/norms.py`) have no leading underscore,
- the tests in `tests/` mirror the package structure, i.e. `ulmRBM/fom/_models.py` is tested in `tests/fom/test_models.py`.
