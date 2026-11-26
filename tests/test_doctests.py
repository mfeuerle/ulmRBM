"""Run the examples in the docstrings of the core modules."""

import doctest
import importlib

import pytest

MODULES = [
    'ulmRBM.core',
    'ulmRBM.solver',
    'ulmRBM.products',
    'ulmRBM.affine._decomposition',
]


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name):
    result = doctest.testmod(importlib.import_module(name), optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0
