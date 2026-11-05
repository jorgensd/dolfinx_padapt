from mpi4py import MPI


def all_ranks(condition, comm: MPI.Comm = MPI.COMM_WORLD) -> bool:
    """Whether a (process local) condition holds on every process.

    Assert this instead of a local condition, such that a failure on one process fails the test
    on all processes. Otherwise, the other processes continue into the collective operations of
    the next test, and the test suite hangs.
    """
    return bool(comm.allreduce(bool(condition), op=MPI.LAND))
