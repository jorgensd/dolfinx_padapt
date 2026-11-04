"""Approach 2: restrict the dofmaps of N parent spaces to the cells of each degree."""

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl

from .base import BlockApproach


def restrict_function_space(
    V: dolfinx.fem.FunctionSpace, cells: npt.NDArray[np.int32]
) -> tuple[dolfinx.fem.FunctionSpace, npt.NDArray[np.int32]]:
    """Restrict a function space to the degrees of freedom in the closure of ``cells``.

    The returned space lives on the same mesh, but only has degrees of freedom associated with
    ``cells``, so it must only be used with integration entities (and interpolation) over
    ``cells``.

    An interior facet kernel always tabulates both cells of the facet, also on an interface
    where the space only lives on one side. The dofmap of a cell outside ``cells`` that shares
    a facet with ``cells`` therefore points to the dofs of its neighbour inside ``cells``.
    The contributions to these dofs are identically zero (the restriction to the outside cell
    never appears in the form), and as they are added into an existing cell block the sparsity
    pattern is unchanged. Rows of all other cells are ``-1``.

    Args:
        V: The parent space.
        cells: Local (owned and ghosted) cells to restrict to.

    Returns:
        The restricted space and the map from its (local) dofs to the dofs of ``V``.
    """
    parent_dofmap = V.dofmap.list
    dofs = np.unique(parent_dofmap[cells].reshape(-1)).astype(np.int32)
    parent_map = V.dofmap.index_map
    sub_map, sub_to_parent, _ = dolfinx.common.create_sub_index_map(parent_map, dofs)
    assert len(sub_to_parent) == len(dofs)

    parent_to_sub = np.full(parent_map.size_local + parent_map.num_ghosts, -1, dtype=np.int32)
    parent_to_sub[sub_to_parent] = np.arange(len(sub_to_parent), dtype=np.int32)
    sub_dofmap = np.full_like(parent_dofmap, -1)
    sub_dofmap[cells] = parent_to_sub[parent_dofmap[cells]]

    # Point cells across the boundary of the restriction to the dofs of their neighbour
    topology = V.mesh.topology
    tdim = topology.dim
    f_to_c = topology.connectivity(tdim - 1, tdim)
    inside = np.zeros(len(parent_dofmap), dtype=bool)
    inside[cells] = True
    interior = np.flatnonzero(np.diff(f_to_c.offsets) == 2)
    c0 = f_to_c.array[f_to_c.offsets[interior]]
    c1 = f_to_c.array[f_to_c.offsets[interior] + 1]
    crossing = inside[c0] != inside[c1]
    c0, c1 = c0[crossing], c1[crossing]
    inner = np.where(inside[c0], c0, c1)
    outer = np.where(inside[c0], c1, c0)
    sub_dofmap[outer] = sub_dofmap[inner]

    cpp_dofmap = dolfinx.cpp.fem.DofMap(
        V.dofmap.dof_layout._cpp_object,
        sub_map._cpp_object,
        V.dofmap.index_map_bs,
        dolfinx.cpp.graph.AdjacencyList_int32(sub_dofmap),
        V.dofmap.bs,
    )
    cpp_space = type(V._cpp_object)(V.mesh._cpp_object, V.element._cpp_object, cpp_dofmap)
    return dolfinx.fem.FunctionSpace(V.mesh, V.ufl_element(), cpp_space), sub_to_parent


class RestrictedApproach(BlockApproach):
    """Spaces ``V_i`` of degree ``i`` on the whole mesh, restricted to the cells of degree ``i``."""

    name = "restricted"

    def _ufl_spaces(self) -> dict[tuple[int, str], ufl.FunctionSpace]:
        fields = self.problem.fields
        if len(fields) != 1 or fields[0].shape != ():
            raise NotImplementedError(
                "Restricted dofmaps are only implemented for scalar problems, use the submesh "
                "approach"
            )
        # The parent spaces are both used for compilation and as the spaces to restrict
        self.parent_spaces = {
            key: dolfinx.fem.functionspace(self.mesh, el) for key, el in self.elements.items()
        }
        return self.parent_spaces

    def _create_spaces(self, active):
        spaces = {}
        for i in active:
            cells = self.degrees.cells(i, include_ghosts=True)
            for name in self._field_names:
                key = (i, name)
                spaces[key], _ = restrict_function_space(self.parent_spaces[key], cells)
        return spaces, []

    def _interpolate(self, u, u_i, degree):
        cells = self.degrees.cells(degree)
        u.interpolate(u_i, cells0=cells, cells1=cells)
