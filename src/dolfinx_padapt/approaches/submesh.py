"""Approach 3: one submesh (and function space) per polynomial degree."""

import basix.ufl
import dolfinx
import numpy as np
import ufl

from .base import BlockApproach


class SubmeshApproach(BlockApproach):
    """Spaces ``V_i`` of degree ``i`` on the submesh of cells with degree ``i``.

    The forms are compiled once with the spaces on *template* domains (one per degree, with the
    coordinate element of the parent mesh). At every update the submeshes are extracted and the
    forms are created over the parent mesh with the cell entity maps.

    A field on the boundary (see :class:`~dolfinx_padapt.problems.Field`) of degree ``i`` lives
    on a submesh of the facets with its marker whose cell has degree ``i``.
    """

    name = "submesh"

    @property
    def submeshes(self) -> dict[int, tuple[dolfinx.mesh.Mesh, dolfinx.mesh.EntityMap]]:
        """The submesh of each degree, and its map to the cells of the parent mesh."""
        return self._submeshes

    def _ufl_spaces(self):
        coordinate_element = self._mesh.ufl_domain().ufl_coordinate_element()
        cmap = self._mesh.geometry.cmaps[0]
        tdim = self._mesh.topology.dim
        facet_coordinate_element = basix.ufl.element(
            "Lagrange",
            basix.cell.sub_entity_type(self._mesh.basix_cell(), tdim - 1, 0),
            cmap.degree,
            basix.LagrangeVariant(cmap.variant),
            shape=(self._mesh.geometry.dim,),
        )
        # One template domain per degree, shared by the fields of that degree, and one per degree
        # and boundary marker for the fields on the boundary
        spaces = {}
        domains = {}
        for field in self._problem.fields:
            for i in self._params.degrees:
                if field.boundary is None:
                    key = (i,)
                    element = coordinate_element
                else:
                    key = (i, field.boundary)
                    element = facet_coordinate_element
                domain = domains.setdefault(key, ufl.Mesh(element))
                spaces[(i, field.name)] = ufl.FunctionSpace(domain, self._elements[(i, field.name)])
        return spaces

    def _create_spaces(self, degrees):
        tdim = self._mesh.topology.dim
        topology = self._mesh.topology
        f_to_c = topology.connectivity(tdim - 1, tdim)
        cell_degrees = self._degrees.cell_degrees
        self._submeshes = {}
        self._facet_submeshes = {}
        spaces = {}
        entity_maps = []
        for i in degrees:
            cells = self._degrees.cells(i, include_ghosts=True)
            submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(self._mesh, tdim, cells)
            self._submeshes[i] = (submesh, cell_map)
            entity_maps.append(cell_map)
            for field in self._problem.fields:
                if field.boundary is None:
                    mesh_i = submesh
                else:
                    if (i, field.boundary) not in self._facet_submeshes:
                        # The facets with the marker of the field whose cell has degree i
                        facets = self._facet_tags.find(field.boundary)
                        facets = facets[cell_degrees[f_to_c.array[f_to_c.offsets[facets]]] == i]
                        facet_mesh, facet_map, _, _ = dolfinx.mesh.create_submesh(
                            self._mesh, tdim - 1, facets.astype(np.int32)
                        )
                        self._facet_submeshes[(i, field.boundary)] = (facet_mesh, facet_map)
                        entity_maps.append(facet_map)
                    mesh_i = self._facet_submeshes[(i, field.boundary)][0]
                key = (i, field.name)
                spaces[key] = dolfinx.fem.functionspace(mesh_i, self._elements[key])
        return spaces, entity_maps

    def _interpolate(self, u, u_i, degree):
        tdim = self._mesh.topology.dim
        submesh, cell_map = self._submeshes[degree]
        sub_cells = np.arange(submesh.topology.index_map(tdim).size_local, dtype=np.int32)
        parent_cells = cell_map.sub_topology_to_topology(sub_cells, inverse=False)
        u.interpolate(u_i, cells0=sub_cells, cells1=parent_cells)
