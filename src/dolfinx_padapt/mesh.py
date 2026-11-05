"""Second order meshes generated with GMSH: the PacMan domain and a half disk."""

from mpi4py import MPI

import dolfinx
import gmsh
import numpy as np

#: First Dirichlet eigenvalue of the Laplacian on the PacMan domain (j_{2/3,1}^2)
EXACT_EIGENVALUE = 3.375610652693620492628**2


def create_pacman(
    comm: MPI.Comm, h: float = 0.25, h_corner: float | None = None, order: int = 2, rank: int = 0
) -> dolfinx.mesh.Mesh:
    r"""Create the unit disk with the quadrant :math:`x > 0, y < 0` removed.

    This is the domain of the `amr.py` NetGen demo in the DOLFINx tutorial, with a re-entrant
    corner of angle :math:`3\pi/2` at the origin.

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


def pacman_boundary_projection(x: np.ndarray) -> np.ndarray:
    """Project points on the boundary of the PacMan domain (:func:`create_pacman`) to the exact
    boundary: the points on the arc are moved to the unit circle, and the points on the straight
    edges are left in place.

    Args:
        x: Points on the (discrete) boundary, with shape ``(3, num_points)``.

    Returns:
        The projected points.
    """
    tol = 1e-10
    straight = ((np.abs(x[1]) < tol) & (x[0] >= -tol)) | ((np.abs(x[0]) < tol) & (x[1] <= tol))
    radius = np.linalg.norm(x[:2], axis=0)
    projected = x.copy()
    arc = ~straight & (radius > tol)
    projected[:2, arc] /= radius[arc]
    return projected


#: Marker of the flat (top) boundary of :func:`create_half_disk`
HALF_DISK_FLAT = 1
#: Marker of the curved (bottom) boundary of :func:`create_half_disk`
HALF_DISK_CURVED = 2


def create_half_disk(
    comm: MPI.Comm,
    radius: float = 0.5,
    centre: float = 1.0,
    h: float = 0.06,
    order: int = 2,
    rank: int = 0,
) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
    r"""Create the lower half of the disk with the given radius and centre :math:`(0, c)`, the
    domain of the Signorini problem in the FEniCS workshop.

    Args:
        comm: Communicator to distribute the mesh over.
        radius: The radius of the disk.
        centre: The :math:`y`-coordinate :math:`c` of the centre of the disk.
        h: Characteristic cell size.
        order: Order of the geometry.
        rank: Rank generating the mesh with GMSH.

    Returns:
        The mesh, ghosted over shared facets (required for interior facet integrals), and the
        facet tags, with :data:`HALF_DISK_FLAT` on the flat boundary and
        :data:`HALF_DISK_CURVED` on the curved boundary.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if comm.rank == rank:
        gmsh.model.add("half_disk")
        disk = gmsh.model.occ.addDisk(0, centre, 0, radius, radius)
        top = gmsh.model.occ.addRectangle(-radius, centre, 0, 2 * radius, 1.1 * radius)
        domain, _ = gmsh.model.occ.cut([(2, disk)], [(2, top)])
        gmsh.model.occ.synchronize()
        flat, curved = [], []
        for dim, tag in gmsh.model.getBoundary(domain, oriented=False):
            length = gmsh.model.occ.getMass(dim, tag)
            (curved if np.isclose(length, np.pi * radius) else flat).append(tag)
        gmsh.model.addPhysicalGroup(2, [tag for _, tag in domain], tag=1, name="domain")
        gmsh.model.addPhysicalGroup(1, flat, tag=HALF_DISK_FLAT, name="flat")
        gmsh.model.addPhysicalGroup(1, curved, tag=HALF_DISK_CURVED, name="curved")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
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
    assert mesh_data.facet_tags is not None
    return mesh, mesh_data.facet_tags


def half_disk_boundary_projection(radius: float = 0.5, centre: float = 1.0):
    """The projection of points on the boundary of the half disk (:func:`create_half_disk`) to
    the exact boundary: the points on the curved boundary are moved to the circle, and the points
    on the flat boundary are left in place.

    Args:
        radius: The radius of the disk.
        centre: The :math:`y`-coordinate of the centre of the disk.

    Returns:
        The projection, see :data:`dolfinx_padapt.refinement.BoundaryProjection`.
    """

    def projection(x: np.ndarray) -> np.ndarray:
        flat = np.abs(x[1] - centre) < 1e-10
        offset = x[:2] - np.array([[0.0], [centre]])
        distance = np.linalg.norm(offset, axis=0)
        projected = x.copy()
        curved = ~flat & (distance > 1e-10)
        projected[:2, curved] = (
            np.array([[0.0], [centre]]) + radius * offset[:, curved] / distance[curved]
        )
        return projected

    return projection
