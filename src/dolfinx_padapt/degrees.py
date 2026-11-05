"""Cell-wise polynomial degree field and the integration entities derived from it."""

import typing

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt

IntegrationEntities = dict[dolfinx.fem.IntegralType, list[tuple[int, npt.NDArray[np.int32]]]]


def pair_tag(i: int, j: int, pmax: int) -> int:
    """Interior facet tag for the interface between cells of degree ``i`` and ``j``."""
    i, j = min(i, j), max(i, j)
    return i * (pmax + 1) + j


def boundary_tag(marker: int, degree: int, pmax: int) -> int:
    """Exterior facet tag for the facets with a given (non-negative) marker of the cells with a
    given degree. Distinct from the tags ``0, ..., pmax`` of all exterior facets of each degree."""
    assert marker >= 0
    return (marker + 1) * (pmax + 1) + degree


def degree_pairs(pmin: int, pmax: int, continuous: bool = False) -> list[tuple[int, int]]:
    """All ``(i, j)``, ``i <= j``, interior facet combinations of the coupling terms.

    For continuous spaces, the facets interior to a single degree region carry no coupling
    terms (the jumps vanish), so ``(i, i)`` is left out. The error indicator integrates over all
    combinations, as the flux jumps do not vanish.
    """
    return [
        (i, j)
        for i in range(pmin, pmax + 1)
        for j in range(i, pmax + 1)
        if not (continuous and i == j)
    ]


class DegreeField:
    """Polynomial degree per cell, stored both as a DG-0 function and as cell tags.

    The value of a cell tag is the polynomial degree of that cell.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, pmin: int, pmax: int):
        self._mesh = mesh
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        mesh.topology.create_connectivity(tdim, tdim - 1)
        self._pmin = pmin
        self._pmax = pmax
        self._Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
        self._p = dolfinx.fem.Function(self._Q, name="p")
        self._p.x.array[:] = pmin
        self._cell_to_dof = self._Q.dofmap.list[:, 0]
        self._update_tags()

    @property
    def mesh(self) -> dolfinx.mesh.Mesh:
        """The mesh."""
        return self._mesh

    @property
    def pmin(self) -> int:
        """The lowest degree."""
        return self._pmin

    @property
    def pmax(self) -> int:
        """The highest degree."""
        return self._pmax

    @property
    def p(self) -> dolfinx.fem.Function:
        """The degree of each cell as a DG-0 function."""
        return self._p

    @property
    def ct(self) -> dolfinx.mesh.MeshTags:
        """The degree of each cell as cell tags (owned and ghost cells)."""
        return self._ct

    @property
    def num_owned_cells(self) -> int:
        return self._mesh.topology.index_map(self._mesh.topology.dim).size_local

    @property
    def cell_degrees(self) -> npt.NDArray[np.int32]:
        """Degree of every cell (owned and ghost) on the process."""
        return self._ct.values

    def _update_tags(self):
        self._p.x.scatter_forward()
        num_cells = len(self._cell_to_dof)
        self._ct = dolfinx.mesh.meshtags(
            self._mesh,
            self._mesh.topology.dim,
            np.arange(num_cells, dtype=np.int32),
            np.rint(self._p.x.array[self._cell_to_dof]).astype(np.int32),
        )

    def set_degrees(self, degrees: npt.NDArray[np.int32]):
        """Set the degree of all owned cells."""
        degrees = np.asarray(degrees)
        assert np.all((self._pmin <= degrees) & (degrees <= self._pmax))
        self._p.x.array[self._cell_to_dof[: self.num_owned_cells]] = degrees
        self._update_tags()

    def refine(self, cells: npt.NDArray[np.int32]) -> int:
        """Increase the degree of the given owned cells by one (capped at ``pmax``).

        Note:
            Collective.

        Returns:
            The number of cells (globally) whose degree changed.
        """
        return self.change(cells, np.empty(0, dtype=np.int32))

    def coarsen(self, cells: npt.NDArray[np.int32]) -> int:
        """Decrease the degree of the given owned cells by one (bounded below by ``pmin``).

        Note:
            Collective.

        Returns:
            The number of cells (globally) whose degree changed.
        """
        return self.change(np.empty(0, dtype=np.int32), cells)

    def change(self, refine: npt.NDArray[np.int32], coarsen: npt.NDArray[np.int32]) -> int:
        """Increase the degree of the owned cells ``refine`` and decrease the degree of the owned
        cells ``coarsen`` by one, within ``[pmin, pmax]``.

        Note:
            Collective.

        Returns:
            The number of cells (globally) whose degree changed.

        Raises:
            ValueError: If a cell is not owned by the process, or is both refined and coarsened
                (on any process).
        """
        # The checks are reduced before raising, such that all processes raise and none waits in
        # a collective. A ghost cell would be counted as changed, and then be overwritten by its
        # owner in the scatter.
        num_owned = self.num_owned_cells
        not_owned = any(np.any((cells < 0) | (cells >= num_owned)) for cells in (refine, coarsen))
        overlap = np.intersect1d(refine, coarsen).size > 0
        local = np.array([not_owned, overlap], dtype=np.int8)
        flags = np.zeros_like(local)
        self._mesh.comm.Allreduce(local, flags, op=MPI.MAX)
        if flags[0]:
            raise ValueError("Only the degree of owned cells can be changed")
        if flags[1]:
            raise ValueError("A cell cannot be both refined and coarsened")
        old = self._p.x.array.copy()
        refine = np.asarray(refine, dtype=np.int32)
        coarsen = np.asarray(coarsen, dtype=np.int32)
        refine_dofs = self._cell_to_dof[refine]
        coarsen_dofs = self._cell_to_dof[coarsen]
        self._p.x.array[refine_dofs] = np.minimum(old[refine_dofs] + 1, self._pmax)
        self._p.x.array[coarsen_dofs] = np.maximum(old[coarsen_dofs] - 1, self._pmin)
        changed = np.union1d(refine_dofs, coarsen_dofs)
        num_changed = int(np.sum(self._p.x.array[changed] != old[changed]))
        self._update_tags()
        return self._mesh.comm.allreduce(num_changed, op=MPI.SUM)

    def cells(self, degree: int, include_ghosts: bool = False) -> npt.NDArray[np.int32]:
        """Local indices of cells with the given degree."""
        n = len(self.cell_degrees) if include_ghosts else self.num_owned_cells
        return np.flatnonzero(self.cell_degrees[:n] == degree).astype(np.int32)

    def integration_entities(
        self,
        facet_tags: dolfinx.mesh.MeshTags | None = None,
        markers: typing.Sequence[int] = (),
    ) -> IntegrationEntities:
        """Integration entities (in parent mesh numbering) for every compiled subdomain id.

        - ``dx(i)``: cells of degree ``i``
        - ``ds(i)``: exterior facets of cells of degree ``i``
        - ``ds(boundary_tag(m, i))``: exterior facets with marker ``m`` (in ``facet_tags``) of cells
          of degree ``i``, for each of the ``markers``
        - ``dS(pair_tag(i, j))``: facets between cells of degree ``i`` and ``j``, where the
          ``"+"`` restriction is the cell with the lowest degree (also ``i == j``, which the
          coupling terms of continuous spaces do not use, but the error indicator does).

        The interior facets are computed once with
        :func:`dolfinx.fem.compute_integration_domains`, and grouped by the degrees of their two
        cells.

        Args:
            facet_tags: Markers of the exterior facets.
            markers: The markers (in ``facet_tags``) with their own exterior facet integrals.

        Raises:
            ValueError: If the mesh is not ghosted with ``GhostMode.shared_facet``.

        Note:
            Collective. All tags are present on every process (possibly with no entities).
        """
        topology = self._mesh.topology
        degrees = self.cell_degrees
        cell_type = dolfinx.fem.IntegralType.cell
        ext_type = dolfinx.fem.IntegralType.exterior_facet
        int_type = dolfinx.fem.IntegralType.interior_facet

        entities: IntegrationEntities = {cell_type: [], ext_type: [], int_type: []}
        boundary = dolfinx.mesh.exterior_facet_indices(topology)
        ext = dolfinx.fem.compute_integration_domains(ext_type, topology, boundary)
        ext = ext.reshape(-1, 2)
        for i in range(self._pmin, self._pmax + 1):
            entities[cell_type].append((i, self.cells(i)))
            ext_i = ext[degrees[ext[:, 0]] == i]
            entities[ext_type].append((i, np.ascontiguousarray(ext_i.reshape(-1))))
        for marker in markers:
            assert facet_tags is not None
            facets = np.intersect1d(facet_tags.find(marker), boundary).astype(np.int32)
            ext_m = dolfinx.fem.compute_integration_domains(ext_type, topology, facets)
            ext_m = ext_m.reshape(-1, 2)
            for i in range(self._pmin, self._pmax + 1):
                ext_mi = ext_m[degrees[ext_m[:, 0]] == i]
                tag = boundary_tag(marker, i, self._pmax)
                entities[ext_type].append((tag, np.ascontiguousarray(ext_mi.reshape(-1))))

        # All interior facets owned by this process. Both cells of such a facet must be on the
        # process, which requires a mesh ghosted with GhostMode.shared_facet. Otherwise, an owned
        # facet on the interface between two processes only has one (local) cell, and is not on
        # the boundary.
        tdim = topology.dim
        f_to_c = topology.connectivity(tdim - 1, tdim)
        num_owned_facets = topology.index_map(tdim - 1).size_local
        num_cells = np.diff(f_to_c.offsets)[:num_owned_facets]
        one_cell = np.flatnonzero(num_cells == 1)
        not_ghosted = np.setdiff1d(one_cell, boundary).size > 0
        if self._mesh.comm.allreduce(not_ghosted, op=MPI.LOR):
            raise ValueError("Interior facet integrals require a mesh with GhostMode.shared_facet")
        interior = np.flatnonzero(num_cells == 2).astype(np.int32)
        data = dolfinx.fem.compute_integration_domains(int_type, topology, interior).reshape(-1, 4)

        # Each facet is (cell_0, local_facet_0, cell_1, local_facet_1). Put the cell with the
        # lowest degree on the "+" side, and group the facets by the degrees of their cells.
        swap = degrees[data[:, 0]] > degrees[data[:, 2]]
        data[swap] = data[swap][:, [2, 3, 0, 1]]
        degree_plus = degrees[data[:, 0]]
        degree_minus = degrees[data[:, 2]]
        for i, j in degree_pairs(self._pmin, self._pmax):
            facets = data[(degree_plus == i) & (degree_minus == j)]
            entities[int_type].append((pair_tag(i, j, self._pmax), facets.reshape(-1)))

        for integral_type in entities:
            entities[integral_type] = [
                (tag, np.ascontiguousarray(ents, dtype=np.int32))
                for tag, ents in sorted(entities[integral_type], key=lambda x: x[0])
            ]
        return entities
