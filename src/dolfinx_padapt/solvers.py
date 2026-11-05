"""Solvers of assembled systems: linear systems (PETSc KSP) and generalised eigenvalue problems
(SLEPc). The block approaches solve linear and nonlinear problems with
:class:`dolfinx.fem.petsc.LinearProblem` and :class:`dolfinx.fem.petsc.NonlinearProblem`."""

from petsc4py import PETSc
from slepc4py import SLEPc


def solve_linear(
    A: PETSc.Mat,
    b: PETSc.Vec,
    nullspace: PETSc.NullSpace | None = None,
    petsc_options: dict | None = None,
) -> PETSc.Vec:
    """Solve :math:`Ax = b` with a PETSc KSP, where :math:`A` may have a (constant) nullspace.

    Args:
        A: The matrix.
        b: The right hand side.
        nullspace: The nullspace of ``A``, removed from ``b`` and the solution.
        petsc_options: PETSc options of the KSP (by default a direct solver, MUMPS).

    Returns:
        The solution.
    """
    if petsc_options is None:
        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    ksp = PETSc.KSP().create(A.getComm())
    prefix = f"dolfinx_padapt_solve_{id(ksp)}_"
    options = PETSc.Options()
    try:
        ksp.setOperators(A)
        ksp.setOptionsPrefix(prefix)
        options.prefixPush(prefix)
        for key, value in petsc_options.items():
            options[key] = value
        options.prefixPop()
        ksp.setFromOptions()
        if nullspace is not None:
            A.setNullSpace(nullspace)
            nullspace.remove(b)
        x = A.createVecRight()
        ksp.solve(b, x)
        reason = ksp.getConvergedReason()
    finally:
        options.prefixPush(prefix)
        for key in petsc_options:
            del options[key]
        options.prefixPop()
        ksp.destroy()
    if reason <= 0:
        x.destroy()
        raise RuntimeError(f"The linear solver did not converge ({reason})")
    if nullspace is not None:
        nullspace.remove(x)
    return x


def solve_eigenproblem(A: PETSc.Mat, M: PETSc.Mat, tol: float = 1e-12) -> tuple[float, PETSc.Vec]:
    """Compute the eigenpair of :math:`Ax = \\lambda Mx` with the eigenvalue closest to zero
    (shift-and-invert, MUMPS)."""
    E = SLEPc.EPS().create(A.getComm())
    try:
        E.setOperators(A, M)
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        E.setDimensions(1, SLEPc.DECIDE)
        E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        E.setTarget(0.0)
        E.setTolerances(tol, 500)
        ST = E.getST()
        ST.setType(SLEPc.ST.Type.SINVERT)
        ST.setShift(0.0)
        ksp = ST.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        E.solve()
        if E.getConverged() < 1:
            raise RuntimeError(f"The eigenvalue solver did not converge ({E.getConvergedReason()})")
        x = A.createVecRight()
        lam = E.getEigenpair(0, x)
    finally:
        E.destroy()
    return float(lam.real), x
