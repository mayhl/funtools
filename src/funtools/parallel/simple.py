from collections.abc import Callable
from multiprocessing import Pool

from tqdm.notebook import tqdm


def simple(
    funcs: list[Callable] | Callable,
    args: list[tuple] | tuple | None,
    kwargs: list[dict] | None = None,
    n_procs: int = 1,
    desc: str | None = None,
) -> list:
    """Executes embarrassingly simple parallel tasks using dask"""
    #   client = Client(DASK_SCHEDULER_ADDRESS)  # , asynchronous=True)
    #  futures = []

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
        nk = len(kwargs) if isinstance(kwargs, list) else 1

    are_same = (nc == nk) and (nc == na)

    if are_same and na == 1:
        raise TypeError("At least one of funcs, args, or kwargs must be a list")

    n = max(nc, nk, na)

    if not (nc == n) and not (nc == 1):
        raise TypeError(f"Length of funcs, {nc:d}, does not match total, {n:d}")

    if not (na == n) and not (na == 1):
        raise TypeError(f"Length of args, {na:d}, does not match total, {n:d}")

    if not (nk == n) and (not nk == 1):
        raise TypeError(f"Length of kwargs, {nk:d}, does not match total, {n:d}")

    if nc == 1:
        funcs = [funcs] * n
    if nk == 1:
        kwargs = [kwargs] * n
    if na == 1:
        args = [args] * n

    if desc is None:
        p_bar = None
    else:
        p_bar = tqdm(total=n, desc=desc)

    # Setting up callback function for updating progress bar required for integration with pool.apply_async
    if p_bar is not None:

        def update_progress_bar(*dummy):
            p_bar.update()

    else:

        def update_progress_bar(*dummy):
            pass

    arg_list = zip(funcs, args, kwargs)

    if n_procs > 1:
        with Pool(n_procs) as pool:
            jobs = [
                pool.apply_async(f, args=a, kwds=k, callback=update_progress_bar)
                for f, a, k in arg_list
            ]
            results = [j.get() for j in jobs]
    else:

        results = []

        for f, a, k in arg_list:
            results.append(f(*a, **k))
            update_progress_bar()

    return results


def recursive(
    callback: Callable,
    funcs: list[Callable] | Callable,
    args: list[tuple] | tuple | None,
    kwargs: list[dict] | None = None,
    n_procs: int = 1,
    desc: str | None = None,
) -> list:
    """Executes embarrassingly simple parallel tasks using dask"""
    #   client = Client(DASK_SCHEDULER_ADDRESS)  # , asynchronous=True)
    #  futures = []

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
        nk = len(kwargs) if isinstance(kwargs, list) else 1

    are_same = (nc == nk) and (nc == na)

    if are_same and na == 1:
        raise TypeError("At least one of funcs, args, or kwargs must be a list")

    n = max(nc, nk, na)

    if not (nc == n) and not (nc == 1):
        raise TypeError(f"Length of funcs, {nc:d}, does not match total, {n:d}")

    if not (na == n) and not (na == 1):
        raise TypeError(f"Length of args, {na:d}, does not match total, {n:d}")

    if not (nk == n) and (not nk == 1):
        raise TypeError(f"Length of kwargs, {nk:d}, does not match total, {n:d}")

    if nc == 1:
        funcs = [funcs] * n
    if nk == 1:
        kwargs = [kwargs] * n
    if na == 1:
        args = [args] * n

    if desc is None:
        desc = "Jobs"

    p_bar = tqdm(total=n, desc=desc)

    # Setting up callback function for updating progress bar required for integration with pool.apply_async
    if p_bar is not None:

        def update_progress_bar(*dummy):
            p_bar.update()

    else:

        def update_progress_bar(*dummy):
            pass

    def _callback(return_vals):
        level, *return_vals = return_vals
        is_done, return_vals = callback(level, return_vals)

        if is_done:
            results.append(return_vals)
            update_progress_bar()
            return

        funcs = [functools.partial(f, *a, **k) for f, a, k in return_vals]
        p_bar.total += len(funcs)

        print(f"total: {p_bar.total:d}")
        update_progress_bar()

        jobs = [
            pool.apply_async(_level_wrapper, args=(f, 0), callback=_callback)
            for f in funcs
        ]

    arg_list = zip(funcs, args, kwargs)

    import functools

    if n_procs > 1:

        funcs = [functools.partial(f, *a, **k) for f, a, k in arg_list]
        results = []
        with Pool(n_procs) as pool:
            jobs = [
                pool.apply_async(_level_wrapper, args=(f, 0), callback=_callback)
                for f in funcs
            ]
            results = [j.get() for j in jobs]
    else:

        results = []

        for f, a, k in arg_list:
            results.append(f(*a, **k))
            update_progress_bar()
    return results


def _level_wrapper(func: Callable, level: int):
    return level + 1, func()
