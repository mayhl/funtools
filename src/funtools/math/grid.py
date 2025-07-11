import numpy as np


def linspace(s0: float, s1: float, n: int, mode: str = "centered") -> np.ndarray:
    mmap = {
        "centered": (False, lambda dx: dx / 2),
        "left": (False, lambda dx: 0),
        "right": (False, lambda dx: dx),
        "border": (True, lambda dx: 0),
    }

    e, o = mmap[mode]
    o = o((s1 - s0) / n)
    return np.linspace(s0 + o, s1 + o, n, endpoint=e)


def linspace2d(
    x0: float, x1: float, nx: int, y0: float, y1: float, ny: int, mode: str = "centered"
) -> tuple[np.ndarray, np.ndarray]:
    x = linspace(x0, x1, nx, mode)
    y = linspace(y0, y1, ny, mode)
    return x, y


def rectilinear(n: int, ds: float, s0: float = 0, mode: str = "centered") -> np.ndarray:
    mmap = {"centered": (0, 1 / 2), "left": (0, 0), "right": (0, 1), "border": (1, 0)}
    ep, o = mmap[mode]
    return (np.arange(n + ep) + o) * ds + s0


def rectilinear2d(
    nx: int,
    dx: float,
    ny: int,
    dy: float,
    x0: float = 0,
    y0: float = 0,
    mode: str = "centered",
) -> tuple[np.ndarray, np.ndarray]:
    x = rectilinear(nx, dx, x0, mode)
    y = rectilinear(ny, dy, y0, mode)
    return x, y


# Generated linear space for some given spacing centered on some range
def nearest_linspace(s0: float, s1: float, ds: float, mode="centered") -> np.ndarray:
    l = s1 - s0
    n = int(np.floor(l / ds))
    if n < 1:
        n = 1

    mmap = {
        "centered": (False, lambda dx: dx / 2),
        "left": (False, lambda dx: 0),
        "right": (False, lambda dx: dx),
        "border": (True, lambda dx: 0),
    }

    # Centering grid on domain

    e, o = mmap[mode]
    o = o(ds) + (l - n * ds) / 2
    return np.linspace(s0 + o, s1 + o, n, endpoint=e)


def even_divide_slices(num: int, div: int, off: int = 0) -> list[slice]:
    """Returns indices after dividing range into mostly even subsizes"""
    sub, rem = divmod(num, div)
    grps = [sub + (1 if i < rem else 0) for i in range(div)]

    idxs = [(off, off + grps[0])] * div
    for i in range(1, div):
        idxs[i] = (idxs[i - 1][1], idxs[i - 1][1] + grps[i])

    return [slice(i0, i1) for i0, i1 in idxs]


def even_divide_subgrid_slices(data: np.ndarray, target_1d_length: int):
    n, m = data.shape
    nb, mb = [round(s / target_1d_length) for s in [n, m]]
    sy, sx = [even_divide_slices(*it) for it in [(n, nb), (m, mb)]]
    sx, sy = [s.flatten() for s in np.meshgrid(sx, sy)]
    return sy, sx, nb, mb


def nearest_linspace2d(
    x0: float,
    x1: float,
    dx: float,
    y0: float,
    y1: float,
    dy: float,
    mode: str = "centered",
) -> tuple[np.ndarray, np.ndarray]:
    x = nearest_linspace(x0, x1, dx, mode)
    y = nearest_linspace(y0, y1, dy, mode)
    return x, y


def flat_meshgrid(*args, **kwargs) -> tuple[list[np.ndarray], list[int]]:
    aargs = np.meshgrid(*args, **kwargs)
    shp = list(aargs[0].shape)
    aargs = [ss.flatten() for ss in aargs]
    return aargs, shp
