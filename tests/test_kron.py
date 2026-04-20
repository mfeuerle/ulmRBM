import pytest
import numpy as np
import scipy.sparse as sp

kron = pytest.importorskip("kron")

from ulmRBM.affine import AffineObject, AffineLinear, AffineLinear
from ulmRBM.core import TrivialParametric

# affine zerlegung mit kron, insbesondere matrix vecor multiplikation
# solver mit kron
# eigenwerte, also stability constants with kron