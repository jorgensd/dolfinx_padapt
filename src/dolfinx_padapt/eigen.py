"""Smallest eigenpair of the generalised eigenvalue problem A x = λ M x."""

from petsc4py import PETSc
from slepc4py import SLEPc


def solve_eigenproblem(A: PETSc.Mat, M: PETSc.Mat, tol: float = 1e-12) -> tuple[float, PETSc.Vec]:
    """Compute the eigenpair with eigenvalue closest to zero (shift-and-invert, MUMPS)."""
    E = SLEPc.EPS().create(A.getComm())
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
    assert E.getConverged() >= 1, "Eigenvalue solver did not converge"
    x = A.createVecRight()
    lam = E.getEigenpair(0, x)
    E.destroy()
    return float(lam.real), x
