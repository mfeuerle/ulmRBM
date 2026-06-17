import numpy as np
import scipy as sp
from numbers import Number

from dolfinx import mesh, fem, plot, geometry
import ufl
import basix.ufl

import pyvista as pv
import matplotlib.pyplot as plt


def plot_endtime_pyvista(u_np: np.ndarray, time_axis: int ,space: fem.FunctionSpace, name: str):
    # try:
    msh = space.mesh
    comm = msh.comm

    assert(msh.geometry.dim==2)

    # V is the FunctionSpace your solution lives in
    u = fem.Function(space)

    # If u_np matches u_fun.x.array length (includes ghosts), this works directly:
    u.x.array[:] = u_np
    u.x.scatter_forward()

    N = 400
    eps = 1e-10

    if time_axis==0:
        xmax = msh.geometry.x[:, 0].max()
        ymin = msh.geometry.x[:, 1].min()
        ymax = msh.geometry.x[:, 1].max()

        ys = np.linspace(ymin, ymax, N)
        pts = np.c_[np.full(N, xmax - eps), ys]
        pts3 = np.c_[pts, np.zeros(N)]  # (N,3) for dolfinx geometry
    elif time_axis==1:
        xmax = msh.geometry.x[:, 1].max()
        ymin = msh.geometry.x[:, 0].min()
        ymax = msh.geometry.x[:, 0].max()

        ys = np.linspace(ymin, ymax, N)
        pts = np.c_[ys, np.full(N, xmax - eps)]
        pts3 = np.c_[pts, np.zeros(N)]  # (N,3) for dolfinx geometry

    

    bb = geometry.bb_tree(msh, msh.topology.dim)
    cands = geometry.compute_collisions_points(bb, pts3)
    cells = geometry.compute_colliding_cells(msh, cands, pts3)

    vals_local = np.full(N, np.nan)
    for i in range(N):
        if len(cells.links(i)) > 0:
            cell = cells.links(i)[0]
            vals_local[i] = u.eval(pts3[i], cell)[0]

    # Gather and combine (take non-nan)
    Vall = comm.gather(vals_local, root=0)
    if comm.rank == 0:
        vals = np.nan * np.ones(N)
        for v in Vall:
            mask = ~np.isnan(v)
            vals[mask] = v[mask]

        plt.plot(ys, vals, label=name)

        # Standard case: nodal data (CG, DG with degree > 0, ...)
        # cells, types, x = plot.vtk_mesh(space)
        # grid = pv.UnstructuredGrid(cells, types, x)
        # u_arr = np.asarray(u)
        # if u_arr.size != grid.n_points:
        #     raise ValueError(
        #         f"Expected {grid.n_points} point values, got {u_arr.size}."
        #     )
        # grid.point_data["u"] = u_arr
        # grid.set_active_scalars("u")
        # mesh_to_plot = grid.warp_by_scalar(factor=10)
    # except RuntimeError as err:
    #     if "cellwise constants" not in str(err):
    #         raise

        # DG0 case: build topology from mesh and attach values as cell data.
        # cells, types, x = plot.vtk_mesh(space.mesh)
        # grid = pv.UnstructuredGrid(cells, types, x)
        # u_arr = np.asarray(u)
        # if u_arr.size != grid.n_cells:
        #     raise ValueError(
        #         "Cellwise constant plotting expects one value per cell "
        #         f"({grid.n_cells}), got {u_arr.size}."
        #     )
        # grid.cell_data["u"] = u_arr
        # grid.set_active_scalars("u")
        # grid_sep = grid.separate_cells()
        # grid_sep = grid_sep.cell_data_to_point_data(pass_cell_data=True)
        # grid_sep.set_active_scalars("u")
        # mesh_to_plot = grid_sep.warp_by_scalar()
        # mesh_to_plot = grid

    # mesh_to_plot.rotate_z(180, inplace=True)
    # plotter.add_mesh(mesh_to_plot, show_edges=False)
    # plotter.show_grid(xtitle='x1', ytitle='x2', ztitle='u(x)')
    # plotter.add_text(name)

def eval_int_u_T(u_np: np.ndarray, time_axis: int ,space: fem.FunctionSpace):
    # try:
    msh = space.mesh
    comm = msh.comm

    assert(msh.geometry.dim==2)

    # V is the FunctionSpace your solution lives in
    u = fem.Function(space)

    # If u_np matches u_fun.x.array length (includes ghosts), this works directly:
    u.x.array[:] = u_np
    u.x.scatter_forward()

    N = 400
    eps = 1e-10

    if time_axis==0:
        xmax = msh.geometry.x[:, 0].max()
        ymin = msh.geometry.x[:, 1].min()
        ymax = msh.geometry.x[:, 1].max()

        ys = np.linspace(ymin, ymax, N)
        pts = np.c_[np.full(N, xmax - eps), ys]
        pts3 = np.c_[pts, np.zeros(N)]  # (N,3) for dolfinx geometry
    elif time_axis==1:
        xmax = msh.geometry.x[:, 1].max()
        ymin = msh.geometry.x[:, 0].min()
        ymax = msh.geometry.x[:, 0].max()

        ys = np.linspace(ymin, ymax, N)
        pts = np.c_[ys, np.full(N, xmax - eps)]
        pts3 = np.c_[pts, np.zeros(N)]  # (N,3) for dolfinx geometry

    

    bb = geometry.bb_tree(msh, msh.topology.dim)
    cands = geometry.compute_collisions_points(bb, pts3)
    cells = geometry.compute_colliding_cells(msh, cands, pts3)

    vals_local = np.full(N, np.nan)
    for i in range(N):
        if len(cells.links(i)) > 0:
            cell = cells.links(i)[0]
            vals_local[i] = u.eval(pts3[i], cell)[0]

    # Gather and combine (take non-nan)
    Vall = comm.gather(vals_local, root=0)
    if comm.rank == 0:
        vals = np.zeros(N)
        for v in Vall:
            mask = ~np.isnan(v)
            vals[mask] = v[mask]

        # Compute integral via trapezoidal rule
        c = np.ones(N)
        c[0] = 0.5
        c[-1] = 0.5
        return np.dot(vals,c)*(ymax-ymin)/N
