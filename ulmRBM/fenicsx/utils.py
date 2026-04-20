r""" Utility functions for FEniCSx, not directly related to model order reduction.

Functions
----------------
.. autosummary::
   :toctree: generated/
   
    create_measure
    interpolate_function
    dirichletbc
    change_element
    plot_pyvista
    isclose
"""


import numpy as np
import scipy as sp
from numbers import Number

from dolfinx import mesh, fem, plot
import ufl
import basix.ufl

import pyvista as pv


__all__ = [
    'create_measure',
    'interpolate_function',
    'dirichletbc',
    'change_element',
    'plot_pyvista',
    'isclose',
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
    
    As FEniCSx as one heck of a mess of different objects which makes it very difficult to work with, this function tries to extend the functionality of FEniCSx to interpolate a wide range of different objects into a FEniCSx function space and thus provide one function to rule them all.
    
    Args:
        space:
            FEniCSx function space into which to interpolate.
        func:
            The function to interpolate. This can be a scalar or a discrete vector, a FEniCSx Constant or Function or any object that can be cast into `dolfinx.fem.Expression` or can be interpolated using `dolfinx.fem.Function.interpolate`.
    """
    
    if sp.sparse.issparse(func): func = func.toarray()
    if isinstance(func, Number) or isinstance(func, np.ndarray):
        func = fem.Constant(space.mesh, func)
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


def plot_pyvista(u: np.ndarray, space: fem.FunctionSpace, name: str, plotter: pv.Plotter):
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

    mesh_to_plot.rotate_z(180, inplace=True)
    plotter.add_mesh(mesh_to_plot, show_edges=False)
    plotter.show_grid(xtitle='x1', ytitle='x2', ztitle='u(x)')
    plotter.add_text(name)
    
    
def isclose(x: np.ndarray, reference_points: np.ndarray, *args, **kwargs) -> np.ndarray[bool]:
    """Check which points in x are numerically close to at least on target point in at least one coordinate.
    
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