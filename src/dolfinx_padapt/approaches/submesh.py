"""Approach 3: one submesh (and function space) per polynomial degree."""

import dolfinx
import numpy as np
import ufl

from .base import BlockApproach


class SubmeshApproach(BlockApproach):
    """Spaces ``V_i`` of degree ``i`` on the submesh of cells with degree ``i``.

    The forms are compiled once with the spaces on *template* domains (one per degree, with the
    coordinate element of the parent mesh). At every update the submeshes are extracted and the
    forms are created over the parent mesh with the cell entity maps.
    """

    name = "submesh"

    @property
    def submeshes(self) -> dict[int, tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap]]:
        """The submesh of each degree, and its map to the cells of the parent mesh."""
        return self._submeshes

    def _ufl_spaces(self):
        coordinate_element = self._mesh.ufl_domain().ufl_coordinate_element()
        # One template domain per degree, shared by the fields of that degree
        domains = {i: ufl.Mesh(coordinate_element) for i in self._params.degrees}
        return {
            (i, name): ufl.FunctionSpace(domains[i], element)
            for (i, name), element in self._elements.items()
        }

    def _create_spaces(self, degrees):
        tdim = self._mesh.topology.dim
        self._submeshes = {}
        spaces = {}
        entity_maps = []
        for i in degrees:
            cells = self._degrees.cells(i, include_ghosts=True)
            submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(self._mesh, tdim, cells)
            self._submeshes[i] = (submesh, cell_map)
            for name in self._field_names:
                spaces[(i, name)] = dolfinx.fem.functionspace(submesh, self._elements[(i, name)])
            entity_maps.append(cell_map)
        return spaces, entity_maps

    def _interpolate(self, u, u_i, degree):
        tdim = self._mesh.topology.dim
        submesh, cell_map = self._submeshes[degree]
        sub_cells = np.arange(submesh.topology.index_map(tdim).size_local, dtype=np.int32)
        parent_cells = cell_map.sub_topology_to_topology(sub_cells, inverse=False)
        u.interpolate(u_i, cells0=sub_cells, cells1=parent_cells)
