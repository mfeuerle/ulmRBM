r""" Utility functions for FEniCSx, not directly related to model order reduction.

Functions
----------------
.. autosummary::
   :toctree: generated/
   
    assemble_matrix
    assemble_vector
    create_measure
    interpolate_function
    dirichletbc
    change_element
    plot_pyvista
    isclose
    get_interpolation_points
    point_cells
    point_evaluation
    point_functional
    projection_matrices
"""


import numpy as np
import scipy as sp
from numbers import Number
from scipy.sparse import csr_array

from dolfinx import mesh, fem, plot, geometry
import ufl
import basix.ufl

import pyvista as pv

from ulmRBM.core import Mu
from ulmRBM.affine import AffineObject, AffineLinear
from ulmRBM.affine._affine import _ConstructNew


__all__ = [
    'create_measure',
    'interpolate_function',
    'dirichletbc',
    'change_element',
    'plot_pyvista',
    'isclose',
    'get_interpolation_points',
    'assemble_matrix',
    'assemble_vector',
    'point_cells',
    'point_evaluation',
    'point_functional',
    'projection_matrices'
]

def create_measure(integral_type: str, domain: mesh.Mesh, tdim: int, entities: list, tags = None) -> ufl.Measure:
    r""" Create a measure for the given entities and tags.
    
    Args:
        integral_type:
            Type of the integral, e.g. ``"dx"``, ``"ds"``, ``"dS"``,
        domain:
            The mesh on which the measure is defined.
        tdim:
            Topological dimension of the entities, e.g. 2 for facets in a 3D mesh.
        entities:
            List of arrays of entity indices on which the measure is defined, e.g. ``entities[i]`` can be a list of entities as created by `dolfinx.mesh.locate_entities`.
        tags:
            List of tags corresponding to the entities, e.g. ``tags[i]`` is the tag for ``entities[i]``. If tags is ``None``, the tags are set to ``0, 1, ..., len(entities)-1``. 
        
    Returns:
        Measure with subdomain_data corresponding to the given entities and tags, i.e. ``out[i]`` returns the integral meassure over all entities with tag ``i``.
    """
    if tags is None: tags = np.arange(len(entities))
    assert len(entities) == len(tags)
    
    full_tags = len(entities)*[None]
    for i in range(len(entities)):
        full_tags[i] = np.full_like(entities[i], tags[i])
    
    if len(entities) > 0:
        entities = np.concatenate(entities)
        full_tags = np.concatenate(full_tags)
    
    meshtags = mesh.meshtags(domain, tdim, entities, full_tags)
    
    return ufl.Measure(integral_type, domain=domain, subdomain_data=meshtags)

def interpolate_function(space: fem.FunctionSpace, func: np.ndarray | sp.sparse.sparray | Number | fem.Constant | fem.Function | fem.Expression | object) -> fem.Function | fem.Constant:
    r""" Interpolate a given function into a FEniCSx function space.
    
    As FEniCSx has one heck of a mess of different objects which makes it very difficult to work in a unified syntax, this function tries to extend the functionality of FEniCSx to interpolate a wide range of different objects into a FEniCSx function space by a coherent function call and thus provide one function to rule them all.
    
    Args:
        space:
            FEniCSx function space into which to interpolate.
        func:
            The function to interpolate. This can be a scalar or a discrete vector, a FEniCSx Constant or Function or any object that can be cast into `dolfinx.fem.Expression` or can be interpolated using `dolfinx.fem.Function.interpolate`.
    """
    
    if sp.sparse.issparse(func): func = func.toarray()
    if isinstance(func, Number) or isinstance(func, np.ndarray):
        func = fem.Constant(space.mesh, np.double(func))
    if isinstance(func, fem.Constant):
        return func
    if isinstance(func, fem.Function):
        if func.function_space == space:
            return func
        else:
            ip_data = fem.create_interpolation_data(space, func.function_space, space.mesh.topology.original_cell_index)
            func_ = fem.Function(space)
            func_.interpolate_nonmatching(func, space.mesh.topology.original_cell_index, ip_data)
            return func_
            
    if type(func).__module__.startswith("ufl"):
        func = fem.Expression(func, space.element.interpolation_points)
    func_ = fem.Function(space)
    func_.interpolate(func)
    return func_

def dirichletbc(space: fem.FunctionSpace, u, dofs: np.ndarray[int]) -> fem.DirichletBC:
    r""" Create a FEniCSx DirichletBC from the given boundary values and dofs.
    
    Again, the main problem are the many different objects in FEniCSx and one would need to handle each of them separately, e.g. to create a boundary condition from a `dolfinx.fem.Constant`, one needs to call ``dolfinx.fem.dirichletbc(u, dofs, space)``, while for a `dolfinx.fem.Function`, one needs to call ``dolfinx.fem.dirichletbc(u, dofs)``. This functions aimes at providing a unified interface.
    
    Args:
        space:
            FEniCSx function space on which the boundary condition is defined.
        u:
            The boundary values of the Dirichlet boundary condition. This can be any object supported by `interpolate_function`.
        dofs:
            List of dofs on which the boundary condition is applied as one would pass to `dolfinx.fem.dirichletbc`.
            
    Returns:
        A FEniCSx DirichletBC object representing the given boundary condition.
    """
    if isinstance(u, fem.Function):
        if u.function_space != space:
            raise ValueError("Function space of boundary values does not match the function space of the boundary condition.")
    
    u = interpolate_function(space, u)
    if isinstance(u, fem.Function):
        return fem.dirichletbc(u, dofs)
    else:
        return fem.dirichletbc(u, dofs, space)

def change_element(space: fem.FunctionSpace, shape: int =None, degree: int = None, add_degree: int = 0, discontinuous: bool = None) -> fem.FunctionSpace:
    r""" Create a new FEniCSx function space with with a modified element.
    
    This is a helper function to create a new function space with a modified element, e.g. to increase the degree of the element by 1 or to change the shape of the element from scalar to vectorial.
    
    Args:
        space:
            The original function space.
        shape:
            The shape of the new element, e.g. ``(gdim,)`` for a vectorial element. If None, the shape of the original element is used.
        degree:
            The degree of the new element. If None, the degree of the original element is used.
        add_degree:
            Increase the degree of the original element by this value. This is added to the degree given by the ``degree`` argument, i.e. the degree of the new element is ``degree + add_degree`` if ``degree`` is not None and ``original_degree + add_degree`` if ``degree`` is None.
        discontinuous:
            Whether the new element should be discontinuous. If None, the same discontinuity as the original element is used, except that if the original element is continuous and the new degree is 0, the new element is set to be discontinuous, as there is no continuous element of degree 0.
        
    Returns:
        A new FEniCSx function space with the modified element.
    """
    el = space.ufl_element().basix_element
    if shape is None:
        shape = space.ufl_element().reference_value_shape
    if degree is None:
        degree = el.degree
    degree += add_degree
    if discontinuous is None:
        discontinuous = el.discontinuous
        if not discontinuous and degree == 0:
            discontinuous = True
    
    el_new = basix.ufl.element(family=el.family, cell=el.cell_type, degree=degree, lagrange_variant=el.lagrange_variant, dpc_variant=el.dpc_variant, discontinuous=discontinuous, shape=shape, dof_ordering=el.dof_ordering, dtype=el.dtype)
    
    return fem.functionspace(space.mesh, el_new)


def plot_pyvista(u: np.ndarray, space: fem.FunctionSpace, name: str, plotter: pv.Plotter, scale: tuple[float,float,float] = (1,1,1)):
    try:
        # Standard case: nodal data (CG, DG with degree > 0, ...)
        cells, types, x = plot.vtk_mesh(space)
        grid = pv.UnstructuredGrid(cells, types, x)
        u_arr = np.asarray(u)
        if u_arr.size != grid.n_points:
            raise ValueError(
                f"Expected {grid.n_points} point values, got {u_arr.size}."
            )
        grid.point_data["u"] = u_arr
        grid.set_active_scalars("u")
        mesh_to_plot = grid.warp_by_scalar()
    except RuntimeError as err:
        if "cellwise constants" not in str(err):
            raise
        # DG0 case: build topology from mesh and attach values as cell data.
        cells, types, x = plot.vtk_mesh(space.mesh)
        grid = pv.UnstructuredGrid(cells, types, x)
        u_arr = np.asarray(u)
        if u_arr.size != grid.n_cells:
            raise ValueError(
                "Cellwise constant plotting expects one value per cell "
                f"({grid.n_cells}), got {u_arr.size}."
            )
        grid.cell_data["u"] = u_arr
        grid.set_active_scalars("u")
        grid_sep = grid.separate_cells()
        grid_sep = grid_sep.cell_data_to_point_data(pass_cell_data=True)
        grid_sep.set_active_scalars("u")
        mesh_to_plot = grid_sep.warp_by_scalar()
        # mesh_to_plot = grid

    # plotter.add_mesh(mesh_to_plot, show_edges=False).scale = scale
    # plotter.show_grid(xtitle='x1', ytitle='x2', ztitle='u(x)')    

    import vtk


    camera = plotter.renderer.GetActiveCamera()
    transform = vtk.vtkTransform()
    transform.Scale(*scale)
    camera.SetModelTransformMatrix(transform.GetMatrix())
    
    plotter.add_mesh(mesh_to_plot, show_edges=False)
    plotter.show_grid(xtitle='x1', ytitle='x2', ztitle='u(x)')
        
    plotter.camera.Azimuth(180)
    if name is not None: plotter.add_text(name)
    
    
def isclose(x: np.ndarray, reference_points: np.ndarray, *args, **kwargs) -> np.ndarray[bool]:
    r"""Check which points in x are numerically close to at least on target point in at least one coordinate.
    
    Args:
        x : 
            Coordinate values to test. Expected shape ``(n_dims, n_points)``
            or ``(n_points,)`` if ``n_dims = 1``.
        reference_points : 
            Reference values to test against. Expected shape ``(n_refs, n_dims)``,
            or ``(n_dims,)`` if ``n_refs = 1``, or scalar if ``n_dims = n_refs = 1``.
        \*args, \*\*kwargs :
            Additional arguments passed to `numpy.isclose` for the actual closeness check (e.g. ``rtol``, ``atol``).
        
    Returns:
        Array of shape ``(n_points,)``, where ``output[i]`` is ``True`` if ``x[j,i]`` is close to ``reference_points[k,j]`` for at least on ``k``, i.e. if the point is close to at least one reference point in at least one dimension.
    """
    
    x = np.asarray(x)
    reference_points = np.atleast_1d(np.asarray(reference_points))
    
    if x.ndim == 1:
        x = x.reshape(1,-1)
        
    if reference_points.ndim == 1:
        if x.shape[0] == 1:
            reference_points = reference_points.reshape(-1,1)
        else:
            reference_points = reference_points.reshape(1,-1)
    
    assert reference_points.ndim == 2
    assert x.ndim == 2
    assert reference_points.shape[1] == x.shape[0]
    
    return np.logical_or.reduce([np.isclose(xi, di, *args, **kwargs) for d in reference_points for (xi,di) in zip(x,d)])


def get_interpolation_points(U: fem.FunctionSpace) -> np.ndarray:
    r"""Get the interpolation points of a FEniCSx function space.
    
    When using `dolfinx.fem.Function.interpolate`, the interpolated function is evaluated at several interpolation points that depend on the underlying function space. This function returns the interpolation points of a given FEniCSx function space.
    """
    y = [None]
    def __get_points(x):
        y[0] = x.copy()
        dummy = np.zeros(U.value_shape).reshape(-1,1)
        return np.zeros((dummy.shape[0],x.shape[1]))
    fem.Function(U).interpolate(__get_points)
    return y[0]


def assemble_matrix(B: ufl.Form | AffineObject[Mu, ufl.Form]) -> csr_array | AffineLinear[Mu, csr_array]:
    r"""Assemble the matrix of a (parametric) bilinear form.
    
    Args:
        B:
            (Parametric) bilinear form.
            
    Returns:
        Assembled matrix representation as a (parametric) sparse array.
    """
    
    assemble = lambda B: csr_array(fem.assemble_matrix(fem.form(B)).to_scipy())
    if isinstance(B, AffineObject):
        return AffineLinear(B.compress().apply2data(assemble))
    else:
        return assemble(B)


def assemble_vector(l: ufl.Form | AffineObject[Mu, ufl.Form]) -> np.ndarray | AffineLinear[Mu,np.ndarray]:
    r"""Assemble the vector of a (parametric) linear form.
    
    Args:
        l:
            (Parametric) linear form.
            
    Returns:
        Assembled vector representation as a (parametric) numpy array.
    """
    
    assemble = lambda l: fem.assemble_vector(fem.form(l)).array
    if isinstance(l, AffineObject):
        return AffineLinear(l.compress().apply2data(assemble))
    else:
        return assemble(l)
    
def _make_points_3d(msh: ufl.Mesh, points: np.ndarray):
    if points.shape[1] != 3:
        if points.shape[1] != msh.geometry.dim:
            raise ValueError(f"Points have wrong dimension {points.shape}, should be either shape=(npoints,3) or shape=(npoints,{msh.geometry.dim}).")
        points = np.hstack( (points, np.zeros((points.shape[0], 3 - points.shape[1]))) )
    return points

def point_cells(msh: ufl.Mesh, points: np.ndarray) -> list[np.int32]:
    r"""Return cells that contain the coordinates in points.

    Args:
        msh:
            Mesh containing all cells.
        points:
            (3D) coordinates of points of interest.
   
    Returns:
        List of cells that contain the coordinates in points.
    """

    points = _make_points_3d(msh, points)
    
    cells = []
    # 1. Build a bounding-box tree and find the cell containing the point
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)
    for i in range(len(points)):
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
        else:
            raise ValueError("Point is outside the mesh.")
    
    return cells

def point_evaluation(func: fem.Function, points: np.ndarray[np.float64], cells: list[np.int32] = None) -> np.ndarray:
    r"""Point evaluate a function.

    Args:
        func:
            Function to be evaluated.
        points:
            (3D) coordinates of points of interest.
        cells:
            The cells contaning the coordinates in points.
   
    Returns:
        Array of function values at the coordinates in points.
    """

    points = _make_points_3d(func.function_space.mesh, points)
    if cells is None:
        cells = point_cells(func.function_space.mesh, points)
    values = func.eval(points, cells)
    return values.reshape(-1)

def point_functional(space: fem.FunctionSpace, points: np.ndarray[np.float64], cells: list[np.int32] = None) -> np.ndarray:
    r"""Compute linear functional for the point evaluation of a function.
    
    Args:
        func:
            Function to be evaluated.
        points:
            (3D) coordinates of points of interest.
        cells:
            The cells contaning the coordinates in points.

    Returns:
        Linear operator as 2D array.
    """

    points = _make_points_3d(space.mesh, points)
    if cells is None:
        cells = point_cells(space.mesh, points)
        
    m = points.shape[0]
    n = space.dofmap.index_map.size_global
    
    L = np.empty((m, n))
    
    e = fem.Function(space)
    e.x.array[:] = 0.0
    for i in range(n):
        e.x.array[i] = 1.0
        L[:,i] = point_evaluation(e, points, cells)
        e.x.array[i] = 0.0
    return L



def projection_matrices(U1: fem.FunctionSpace, U2: fem.FunctionSpace, both: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    r"""Compute projection matrices between two function spaces.
    
    The two function spaces :math:`U_1` and :math:`U_2` can be defined on different meshes (e.g. a coarse and a fine mesh) and / or have different elements (e.g. different polynomial degrees). The projection matrix :math:`P_{12}: U_1 \to U_2` interpolates a discrete function in :math:`U_1` into :math:`U_2`, while :math:`P_{21}: U_2 \to U_1` projects a discrete function in :math:`U_2` into :math:`U_1`, where :math:`P_{21} := = (P_{12}^T P_{12})^{-1} P_{12}^T` is the Moore–Penrose pseudoinverse of :math:`P_{12}`.
    
    Thus, :math:`P_{21}` only exists, if :math:`P_{12}` has full column rank, i.e. if the functions in :math:`U_1` are linearly independent when interpolated into :math:`U_2` (e.g. if :math:`U_1 \subsetU_2`).
    
    This function is in particular usefull, if you want to embedd a coarse function space into a fine function space, as it is the case e.g. in context of gemetric multigrid methods. In this case,  :math:`U_1` should be the coarse function space and :math:`U_2` the fine function space.
    
    Args:
        U1:
            First function space (coarse).
        U2:
            Second function space (fine).
        both:
            If True, also compute :math:`P_{21}`. If False, compute only :math:`P_{12}`.

    Returns
    -------
    P_12
        Projection matrix from :math:`U_1` to :math:`U_2`.
    P_21
        Projection matrix from :math:`U_2` to :math:`U_1`. (only if ``both=True``)
    """
    
    u_from = fem.Function(U1)
    u_to   = fem.Function(U2)
    P_12 = np.empty((U2.dofmap.index_map.size_global, U1.dofmap.index_map.size_global), dtype=np.float64)
    
    data = fem.create_interpolation_data(U2, U1, U2.mesh.topology.original_cell_index)
    
    for i in range(U1.dofmap.index_map.size_global):
        u_from.x.array[:] = 0.0
        u_from.x.array[i] = 1.0
        u_to.interpolate_nonmatching(u_from, U2.mesh.topology.original_cell_index, data)
        P_12[:,i] = u_to.x.array[:]
    
    if both:
        P_21 = np.linalg.solve(P_12.T @ P_12, P_12.T)
        return P_12, P_21
    else:
        return P_12