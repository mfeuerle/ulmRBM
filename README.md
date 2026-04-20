# ulmMOR

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
