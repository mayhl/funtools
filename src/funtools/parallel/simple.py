from dask.distributed import Client
from collections.abc import Callable

# TODO: Redo cluster
from .cluster import DASK_SCHEDULER_ADDRESS


def simple(
    self,
    funcs: list[Callable] | Callable,
    args: list[tuple] | tuple | None,
    kwargs: list[dict] | None = None,
) -> list:
    """Executes embarrassingly simple parallel tasks using dask"""
    client = Client(DASK_SCHEDULER_ADDRESS)  # , asynchronous=True)
    futures = []

    nc = len(funcs) if isinstance(funcs, list) else 1

    if args is None:
        na = 1
        args: tuple = ()
    else:
        na = len(args) if isinstance(args, list) else 1

    if kwargs is None:
        nk = 1
        kwargs: dict = {}
    else:
        nk = len(kwargs) if issubclass(kwargs, dict) else 1

    are_same = (nc == nk) and (nc == na)

    if are_same and na == 1:
        raise TypeError("At least one of funcs, args, or kwargs must be a list")

    n = max(nc, nk, na)

    if not nc == n or nc == 1:
        raise TypeError(f"Length of funcs, {nc:d}, does not match total, {n:d}")

    if not na == n or na == 1:
        raise TypeError(f"Length of args, {na:d}, does not match total, {n:d}")

    if not nk == n or nk == 1:
        raise TypeError(f"Length of kwargs, {nk:d}, does not match total, {n:d}")

    if nc == 1:
        funcs = [funcs] * n
    if nk == 1:
        kwargs = [kwargs] * n
    if na == 1:
        args = [args] * n

    for (
        f,
        a,
        k,
    ) in zip(funcs, args, kwargs):
        future = client.submit(f, *a, **k)
        futures.append(future)

    return client.gather(futures)
