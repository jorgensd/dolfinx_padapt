"""Cell-wise polynomial degree field and the integration entities derived from it."""

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import scifem

IntegrationEntities = dict[dolfinx.fem.IntegralType, list[tuple[int, npt.NDArray[np.int32]]]]


def pair_tag(i: int, j: int, pmax: int) -> int:
    """Interior facet tag for the interface between cells of degree ``i`` and ``j``."""
    i, j = min(i, j), max(i, j)
    return i * (pmax + 1) + j


def degree_pairs(pmin: int, pmax: int, continuous: bool) -> list[tuple[int, int]]:
    """All ``(i, j)``, ``i <= j``, interior facet combinations that are integrated over.

    For continuous spaces, the facets interior to a single degree region carry no jump terms.
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
        self.mesh = mesh
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        mesh.topology.create_connectivity(tdim, tdim - 1)
        self.pmin = pmin
        self.pmax = pmax
        self.Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
        self.p = dolfinx.fem.Function(self.Q, name="p")
        self.p.x.array[:] = pmin
        self._cell_to_dof = self.Q.dofmap.list[:, 0]
        self._update_tags()

    @property
    def num_owned_cells(self) -> int:
        return self.mesh.topology.index_map(self.mesh.topology.dim).size_local

    @property
    def cell_degrees(self) -> npt.NDArray[np.int32]:
        """Degree of every cell (owned and ghost) on the process."""
        return self.ct.values

    def _update_tags(self):
        self.p.x.scatter_forward()
        num_cells = len(self._cell_to_dof)
        self.ct = dolfinx.mesh.meshtags(
            self.mesh,
            self.mesh.topology.dim,
            np.arange(num_cells, dtype=np.int32),
            np.rint(self.p.x.array[self._cell_to_dof]).astype(np.int32),
        )

    def set_degrees(self, degrees: npt.NDArray[np.int32]):
        """Set the degree of all owned cells."""
        degrees = np.asarray(degrees)
        assert np.all((self.pmin <= degrees) & (degrees <= self.pmax))
        self.p.x.array[self._cell_to_dof[: self.num_owned_cells]] = degrees
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
        """
        old = self.p.x.array.copy()
        refine_dofs = self._cell_to_dof[refine]
        coarsen_dofs = self._cell_to_dof[coarsen]
        self.p.x.array[refine_dofs] = np.minimum(old[refine_dofs] + 1, self.pmax)
        self.p.x.array[coarsen_dofs] = np.maximum(old[coarsen_dofs] - 1, self.pmin)
        changed = np.union1d(refine_dofs, coarsen_dofs)
        num_changed = int(np.sum(self.p.x.array[changed] != old[changed]))
        self._update_tags()
        return self.mesh.comm.allreduce(num_changed, op=MPI.SUM)

    def active_degrees(self) -> list[int]:
        """Degrees used by at least one cell on any process."""
        local = np.zeros(self.pmax + 1, dtype=np.int32)
        local[np.unique(self.cell_degrees[: self.num_owned_cells])] = 1
        active = np.zeros_like(local)
        self.mesh.comm.Allreduce(local, active, op=MPI.MAX)
        return [int(d) for d in np.flatnonzero(active)]

    def cells(self, degree: int, include_ghosts: bool = False) -> npt.NDArray[np.int32]:
        """Local indices of cells with the given degree."""
        n = len(self.cell_degrees) if include_ghosts else self.num_owned_cells
        return np.flatnonzero(self.cell_degrees[:n] == degree).astype(np.int32)

    def integration_entities(self, continuous: bool = False) -> IntegrationEntities:
        """Integration entities (in parent mesh numbering) for every compiled subdomain id.

        - ``dx(i)``: cells of degree ``i``
        - ``ds(i)``: exterior facets of cells of degree ``i``
        - ``dS(pair_tag(i, j))``: facets between cells of degree ``i`` and ``j``, where the
          ``"+"`` restriction is the cell with the lowest degree.

        The interface facets are found with :func:`scifem.find_interface`, and ordered with
        :func:`scifem.compute_interface_data`.

        Note:
            Collective. All tags are present on every process (possibly with no entities).
        """
        topology = self.mesh.topology
        degrees = self.cell_degrees
        cell_type = dolfinx.fem.IntegralType.cell
        ext_type = dolfinx.fem.IntegralType.exterior_facet
        int_type = dolfinx.fem.IntegralType.interior_facet

        entities: IntegrationEntities = {cell_type: [], ext_type: [], int_type: []}
        boundary = dolfinx.mesh.exterior_facet_indices(topology)
        ext = dolfinx.fem.compute_integration_domains(ext_type, topology, boundary)
        ext = ext.reshape(-1, 2)
        for i in range(self.pmin, self.pmax + 1):
            entities[cell_type].append((i, self.cells(i)))
            ext_i = ext[degrees[ext[:, 0]] == i]
            entities[ext_type].append((i, np.ascontiguousarray(ext_i.reshape(-1))))

        # Interfaces between different degrees, "+" is the lowest degree
        cross_facets = []
        for i, j in degree_pairs(self.pmin, self.pmax, continuous=True):
            facets = scifem.find_interface(self.ct, (i,), (j,))
            cross_facets.append(facets)
            data = scifem.compute_interface_data(self.ct, facets)
            entities[int_type].append((pair_tag(i, j, self.pmax), data.reshape(-1)))

        if not continuous:
            # find_interface(ct, (i,), (i,)) returns all interior facets of the cells with
            # degree i, also those shared with a cell of another degree.
            all_cross = np.unique(np.concatenate(cross_facets)) if cross_facets else []
            for i in range(self.pmin, self.pmax + 1):
                facets = scifem.find_interface(self.ct, (i,), (i,))
                facets = np.setdiff1d(facets, all_cross).astype(np.int32)
                data = scifem.compute_interface_data(self.ct, facets)
                entities[int_type].append((pair_tag(i, i, self.pmax), data.reshape(-1)))

        for integral_type in entities:
            entities[integral_type] = [
                (tag, np.ascontiguousarray(ents, dtype=np.int32))
                for tag, ents in sorted(entities[integral_type], key=lambda x: x[0])
            ]
        return entities
