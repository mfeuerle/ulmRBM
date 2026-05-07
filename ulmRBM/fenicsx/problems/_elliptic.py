import numpy as np
import pyvista as pv
from itertools import product

from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from ulmRBM.core import Mu
from ulmRBM.affine import AffineObject
from ulmRBM.fenicsx import utils, FEniCSxSpaceWithDirichletBCs
from ulmRBM.fenicsx.problems import weak_problem

__all__ = [
    'thermal_block',
    'simple_elliptic',
]

# def simple_elliptic_operator(dim: int) -> tuple[AffineObject, AffineObject, AffineObject]:


def simple_elliptic(n: list[int]) -> tuple[AffineObject[Mu, ufl.Form], AffineObject[Mu, ufl.Form], FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r"""Parametric elliptic problem on the unit square.

    The model uses `weak_problem` with

    .. math::
        A_\mu(x) = -\mu\,I, \qquad b(x) = 0, \qquad c(x) = 0, \qquad f(x) = 1,

    where :math:`\mu` is the parameter. All exterior facets are treated as
    Dirichlet boundaries for trial and test spaces, i.e. :math:`\Gamma_D =
    \partial\Omega` and :math:`\Gamma_N = \emptyset` with dirichlet data
    :math:`g=0`.

    Args:
        n : 
            Number of mesh cells in each spatial direction, where ``len(n)`` is the spatial dimension.
            
    Returns
    -------
    See `weak_problem` for details on the return values.
    """
    
    if len(n) == 1:
        msh = mesh.create_unit_interval(MPI.COMM_WORLD, *n)
    elif len(n) == 2:
        msh = mesh.create_unit_square(MPI.COMM_WORLD, *n)
    elif len(n) == 3:
        msh = mesh.create_unit_cube(MPI.COMM_WORLD, *n)
    else:
        raise ValueError("Only 1D, 2D and 3D problems are supported.")
    
    gdim = msh.geometry.dim
    
    dbdry = [lambda x: np.ones(x.shape[1], dtype=bool)]
    
    f = AffineObject([1.0], [1.0])
    g = [AffineObject([0.0], [0.0])]
    h = []
    
    A = AffineObject([lambda mu: mu], [lambda x: -np.eye(gdim).reshape(-1,1)])
    b = AffineObject([0.0], [np.zeros(gdim)])
    c = AffineObject([0.0], [0.0])
    
    return weak_problem(msh, (A,b,c), (f,g,h), dbdry)


def thermal_block(nh: list[int,int], nblocks: list[int,int], plot: bool = False) -> tuple[AffineObject[Mu, ufl.Form], AffineObject[Mu, ufl.Form], FEniCSxSpaceWithDirichletBCs, FEniCSxSpaceWithDirichletBCs]:
    r"""Create the parametric thermal block problem.

    The domain is the unit square :math:`\Omega = (0,1)^2`, partitioned into
    :math:`n_{\mathrm{blocks},1} \times n_{\mathrm{blocks},2}` rectangular blocks.
    The model uses

    .. math::
        b(x) = 0, \qquad c(x) = 0, \qquad f(x) = 1,

    and a blockwise affine-parametric diffusion tensor

    .. math::
        A_\mu(x) = -\sum_{q=1}^{Q} \theta_q(\mu)\,\chi_q(x)\,I,

    where :math:`\chi_q` are indicator functions of the blocks and
    :math:`\theta_q(\mu) = \mu_q`.

    All exterior facets are treated as Dirichlet boundaries for 
    trial and test spaces, i.e. :math:`\Gamma_D = \partial\Omega`
    and :math:`\Gamma_N = \emptyset` with dirichlet data :math:`g=0`.

    Parameters
    ----------
    nh : 
        Number of mesh cells in each spatial direction.
    nblocks : 
        Number of thermal blocks in each spatial direction.
    plot :
        If ``True``, visualize all block indicator functions :math:`\chi_q`.

    Returns
    -------
    See `weak_problem` for details on the return values.
    """
    msh = mesh.create_unit_square(MPI.COMM_WORLD, *nh)
    gdim = msh.geometry.dim
    
    dbdry = [lambda x: np.ones(x.shape[1], dtype=bool)]
    
    f = AffineObject([1.0], [1.0])
    g = [AffineObject([0.0], [0.0])]
    h = []
    
    blocks = [np.linspace(0, 1, nblocks[i]+1) for i in range(2)]
    
    def chi(x, block_id):
        value = np.ones(x.shape[1], dtype=bool)
        for i,id in enumerate(block_id):
            if id != 0:
                value &= blocks[i][id] <= x[i]
            if id != nblocks[i] - 1:
                value &= x[i] < blocks[i][id+1]
        return value

    A = AffineObject()
    for idx in product(*[range(n) for n in nblocks]):
        A += [(lambda mu, idx=idx: mu[idx], lambda x, idx=idx: -np.eye(gdim).reshape(-1,1) * chi(x, idx) )]
    b = AffineObject([0.0], [np.zeros(gdim)])
    c = AffineObject([0.0], [0.0])
        
    if plot:
        plotter = pv.Plotter(shape=(nblocks))
        L2 = fem.functionspace(msh, ("DG", 0))
        for idx in product(range(nblocks[0]), range(nblocks[1])):
            plotter.subplot(*idx)
            tmp = fem.Function(utils.change_element(L2, shape=()))
            tmp.interpolate(lambda x: chi(x, idx))
            utils.plot_pyvista(tmp.x.array, utils.change_element(L2, shape=()), f"chi {idx}", plotter)
        plotter.show(interactive_update=True)
    
    return weak_problem(msh, (A,b,c), (f,g,h), dbdry)