"""Second order PacMan mesh generated with GMSH."""

from mpi4py import MPI

import dolfinx
import gmsh

#: First Dirichlet eigenvalue of the Laplacian on the PacMan domain (j_{2/3,1}^2)
EXACT_EIGENVALUE = 3.375610652693620492628**2


def create_pacman(
    comm: MPI.Comm, h: float = 0.25, h_corner: float | None = None, order: int = 2, rank: int = 0
) -> dolfinx.mesh.Mesh:
    """Create the unit disk with the quadrant x>0, y<0 removed.

    This is the domain of the `amr.py` NetGen demo in the DOLFINx tutorial, with a re-entrant
    corner of angle 3π/2 at the origin.

    Args:
        comm: Communicator to distribute the mesh over.
        h: Characteristic cell size.
        h_corner: Cell size at the re-entrant corner. If ``None`` the mesh is quasi-uniform.
        order: Order of the geometry.
        rank: Rank generating the mesh with GMSH.

    Returns:
        The mesh, ghosted over shared facets (required for interior facet integrals).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if comm.rank == rank:
        gmsh.model.add("pacman")
        disk = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        quadrant = gmsh.model.occ.addRectangle(0, -1, 0, 1, 1)
        domain, _ = gmsh.model.occ.cut([(2, disk)], [(2, quadrant)])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [tag for _, tag in domain], tag=1, name="domain")
        boundary = gmsh.model.getBoundary(domain, oriented=False)
        gmsh.model.addPhysicalGroup(1, [tag for _, tag in boundary], tag=2, name="boundary")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        if h_corner is not None:
            corner = [
                tag
                for _, tag in gmsh.model.getEntities(0)
                if max(abs(c) for c in gmsh.model.getValue(0, tag, [])) < 1e-12
            ]
            distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance, "PointsList", corner)
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", h_corner)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.5)
            gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)

    mesh_data = dolfinx.io.gmsh.model_to_mesh(
        gmsh.model, comm, rank, gdim=2, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    gmsh.finalize()

    mesh = mesh_data.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, tdim - 1)
    return mesh
