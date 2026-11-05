"""h-refinement of meshes with curved (higher order) geometry, and transfer of functions from a mesh
to its refinement.

The refinement of DOLFINx (:func:`dolfinx.mesh.refine`, with the algorithm of Plaza and Carey,
"Local refinement of simplicial grids based on the skeleton", Applied Numerical Mathematics 32(2),
2000, doi:10.1016/S0168-9274(99)00022-7) only supports meshes with straight (affine) cells. A mesh
of geometry order :math:`k` is refined by

1. creating the straight mesh with the same topology (:func:`dolfinx.fem.interpolate_geometry`),
2. refining the straight mesh, which gives the parent cell of each new cell,
3. creating the new mesh of order :math:`k` with straight cells, and
4. moving its nodes to the curved geometry: the coordinates of the original mesh are a vector
   Lagrange function of degree :math:`k` (in the coordinate space of the mesh) on the straight
   mesh, which is interpolated into the coordinate space of the new mesh with
   :func:`interpolate_from_parent`.

The new cells are nested in the curved cells of the original mesh, and the boundary of the
refined mesh is the (curved) boundary of the original mesh.
"""

import typing
from dataclasses import dataclass

import basix
import basix.ufl
import dolfinx
import numpy as np
import numpy.typing as npt

#: Maps the coordinates of points on the boundary, with shape ``(3, num_points)``, to the closest
#: points on the exact boundary, with shape ``(3, num_points)`` (or ``(gdim, num_points)``)
BoundaryProjection = typing.Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]


@dataclass(frozen=True)
class RefinedMesh:
    """A refined mesh, and its relation to the original mesh."""

    #: The refined mesh
    mesh: dolfinx.mesh.Mesh
    #: The (local) parent cell in the original mesh of each owned cell of the refined mesh
    parent_cells: npt.NDArray[np.int32]
    #: The facet tags transferred to the refined mesh (if facet tags were given)
    facet_tags: dolfinx.mesh.MeshTags | None


def interpolate_from_parent(
    u: dolfinx.fem.Function,
    u_fine: dolfinx.fem.Function,
    parent_cells: npt.NDArray[np.int32],
):
    """Interpolate a function into a function on a refined mesh, where each cell of the refined
    mesh lies in a known parent cell.

    The interpolation points of each owned cell of the refined mesh are evaluated in its parent
    cell, which is on the same process (the refinement is not redistributed). The points are not
    searched for, and there is no communication but a scatter of the ghost values. As each point
    is evaluated in the parent cell of the cell it belongs to, this also works for discontinuous
    functions, where a point on the boundary of a parent cell is ambiguous.

    Note:
        Collective.

    Args:
        u: The function on the original mesh.
        u_fine: The function on the refined mesh to interpolate into.
        parent_cells: The (local) parent cell in the mesh of ``u`` of each owned cell of the mesh
            of ``u_fine``, see :class:`RefinedMesh`.
    """
    V = u_fine.function_space
    mesh = V.mesh
    tdim = mesh.topology.dim
    num_owned = mesh.topology.index_map(tdim).size_local
    if len(parent_cells) != num_owned:
        raise ValueError("Every owned cell of the refined mesh needs a parent cell")
    cells = np.arange(num_owned, dtype=np.int32)
    x = dolfinx.cpp.fem.interpolation_coords(
        V.element._cpp_object, mesh.geometry._cpp_object, cells
    ).T
    num_points = V.element.interpolation_points.shape[0]
    rank = np.full(x.shape[0], mesh.comm.rank, dtype=np.int32)
    dtype = mesh.geometry.x.dtype
    cls = getattr(dolfinx.cpp.geometry, f"PointOwnershipData_float{8 * np.dtype(dtype).itemsize}")
    # All points are owned by this process, in the parent cell of the cell they belong to
    data = dolfinx.geometry.PointOwnershipData(
        cls(
            rank,
            rank,
            np.ascontiguousarray(x, dtype=dtype).reshape(-1),
            np.ascontiguousarray(np.repeat(parent_cells, num_points), dtype=np.int32),
        )
    )
    u_fine.interpolate_nonmatching(u, cells, data)
    u_fine.x.scatter_forward()


def _coordinate_space(mesh: dolfinx.mesh.Mesh, cmap, msh: dolfinx.mesh.Mesh):
    """The vector Lagrange space of the coordinate element ``cmap`` of ``mesh`` on ``msh``."""
    element = basix.ufl.element(
        "Lagrange",
        mesh.basix_cell(),
        cmap.degree,
        basix.LagrangeVariant(cmap.variant),
        shape=(mesh.geometry.dim,),
    )
    return dolfinx.fem.functionspace(msh, element)


def refine_curved(
    mesh: dolfinx.mesh.Mesh,
    cells: npt.NDArray[np.int32] | None = None,
    facet_tags: dolfinx.mesh.MeshTags | None = None,
    boundary_projection: BoundaryProjection | None = None,
) -> RefinedMesh:
    """Refine a mesh of any geometry order with the algorithm of Plaza and Carey (see the module
    documentation).

    Note:
        Collective. Only simplex meshes can be refined.

    Args:
        mesh: The mesh to refine.
        cells: The (local) cells to refine. Further cells are refined to keep the mesh conforming.
            If ``None``, every cell is refined.
        facet_tags: Facet tags to transfer to the refined mesh.
        boundary_projection: If given, the nodes of the refined mesh on the boundary are moved to
            the exact boundary with this map. Otherwise, the boundary of the refined mesh is the
            boundary of ``mesh``. The map must leave nodes that are on the exact boundary in
            place, such as all nodes of ``mesh`` if it was created with the same map.

    Returns:
        The refined mesh, ghosted over shared facets, the parent cells and the facet tags.
    """
    tdim, gdim = mesh.topology.dim, mesh.geometry.dim
    cmap = mesh.geometry.cmaps[0]
    cell_type = mesh.topology.cell_type
    variant = basix.LagrangeVariant(cmap.variant)
    straight = dolfinx.fem.interpolate_geometry(
        mesh, dolfinx.fem.coordinate_element(cell_type, 1, dtype=mesh.geometry.x.dtype)
    )

    # The coordinates of the mesh, as a function on the straight mesh (with the same topology,
    # and hence the same dofmap)
    V = _coordinate_space(mesh, cmap, mesh)
    X = dolfinx.fem.Function(V)
    X.interpolate(lambda x: x[:gdim])
    V_straight = _coordinate_space(mesh, cmap, straight)
    assert np.array_equal(V.dofmap.list, V_straight.dofmap.list)
    X_straight = dolfinx.fem.Function(V_straight)
    X_straight.x.array[:] = X.x.array

    straight.topology.create_entities(1)
    edges = None
    if cells is not None:
        edges = dolfinx.mesh.compute_incident_entities(straight.topology, cells, tdim, 1)
    if facet_tags is not None:
        straight.topology.create_entities(tdim - 1)
        straight.topology.create_connectivity(tdim - 1, tdim)
    fine_straight, parent_cell, parent_facet = dolfinx.mesh.refine(
        straight,
        edges,
        option=dolfinx.mesh.RefinementOption.parent_cell_and_facet,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
    )
    assert parent_cell is not None
    fine = dolfinx.fem.interpolate_geometry(
        fine_straight,
        dolfinx.fem.coordinate_element(
            cell_type, cmap.degree, variant, dtype=mesh.geometry.x.dtype
        ),
    )

    # The parent of each owned cell, in the final order of the cells of the refined mesh
    cell_map = fine.topology.index_map(tdim)
    num_owned = cell_map.size_local
    original = np.asarray(fine.topology.original_cell_index, dtype=np.int64)
    original = (original[:num_owned] - cell_map.local_range[0]).astype(np.int32)
    parents = np.ascontiguousarray(parent_cell[original], dtype=np.int32)

    # Move the nodes of the refined mesh to the curved geometry. The nodes of a cell are in the
    # order of the dofs of a cell of the coordinate space.
    X_fine = dolfinx.fem.Function(_coordinate_space(mesh, cmap, fine_straight))
    interpolate_from_parent(X_straight, X_fine, parents)
    values = X_fine.x.array.reshape(-1, gdim)
    if boundary_projection is not None:
        _project_boundary(X_fine, boundary_projection)
    nodes = fine.geometry.dofmaps[0].reshape(-1)
    fine.geometry.x[nodes, :gdim] = values[X_fine.function_space.dofmap.list.reshape(-1)]

    fine.topology.create_connectivity(tdim - 1, tdim)
    fine.topology.create_connectivity(tdim, tdim - 1)
    fine_tags = None
    if facet_tags is not None:
        assert parent_facet is not None
        num_facets = dolfinx.cpp.mesh.cell_num_entities(cell_type, tdim - 1)
        parent_facets = parent_facet.reshape(-1, num_facets)[original]
        fine_tags = _transfer_facet_tags(facet_tags, mesh, fine, parents, parent_facets)
    return RefinedMesh(fine, parents, fine_tags)


def _transfer_facet_tags(
    facet_tags: dolfinx.mesh.MeshTags,
    mesh: dolfinx.mesh.Mesh,
    fine: dolfinx.mesh.Mesh,
    parents: npt.NDArray[np.int32],
    parent_facets: npt.NDArray[np.int8],
) -> dolfinx.mesh.MeshTags:
    """Tag the facets of the owned cells of the refined mesh that lie in a tagged facet of the
    parent cell. (:func:`dolfinx.mesh.transfer_meshtag` does not support ghosted meshes.)

    Args:
        facet_tags: The facet tags of the original mesh.
        mesh: The original mesh.
        fine: The refined mesh.
        parents: The parent cell of each owned cell of the refined mesh.
        parent_facets: For each owned cell of the refined mesh and each of its local facets, the
            local facet of the parent cell containing it, or ``-1`` if it is inside the parent
            cell.
    """
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    fine.topology.create_connectivity(tdim, tdim - 1)
    num_facets = parent_facets.shape[1]
    coarse_c_to_f = mesh.topology.connectivity(tdim, tdim - 1).array.reshape(-1, num_facets)
    fine_c_to_f = fine.topology.connectivity(tdim, tdim - 1).array.reshape(-1, num_facets)
    coarse_map = mesh.topology.index_map(tdim - 1)
    values = np.full(coarse_map.size_local + coarse_map.num_ghosts, -1, dtype=np.int32)
    values[facet_tags.indices] = facet_tags.values

    cells, local = np.nonzero(parent_facets >= 0)
    coarse = coarse_c_to_f[parents[cells], parent_facets[cells, local]]
    tagged = values[coarse] >= 0
    facets = fine_c_to_f[cells[tagged], local[tagged]]
    facets, unique = np.unique(facets, return_index=True)
    return dolfinx.mesh.meshtags(
        fine, tdim - 1, facets.astype(np.int32), values[coarse][tagged][unique]
    )


def _project_boundary(X: dolfinx.fem.Function, projection: BoundaryProjection):
    """Move the coordinates in ``X`` (a function in the coordinate space) on the boundary with
    ``projection``. The dofs on the owned exterior facets are communicated to their owners, which
    project them, and the ghosts are updated."""
    V = X.function_space
    mesh = V.mesh
    tdim, gdim = mesh.topology.dim, mesh.geometry.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, facets)
    index_map = V.dofmap.index_map
    marker = dolfinx.la.vector(index_map, 1, dtype=np.int32)
    marker.array[dofs] = 1
    marker.scatter_reverse(dolfinx.la.InsertMode.add)
    owned = np.flatnonzero(marker.array[: index_map.size_local] > 0)
    values = X.x.array.reshape(-1, gdim)
    points = np.zeros((3, len(owned)), dtype=values.dtype)
    points[:gdim] = values[owned].T
    values[owned] = np.asarray(projection(points))[:gdim].T
    X.x.scatter_forward()
