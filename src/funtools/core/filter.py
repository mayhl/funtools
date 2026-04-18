from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def maskScatter(
    x: NDArray, y: NDArray, bounds: tuple[float | None, ...] | None = None
) -> NDArray | slice:

    if bounds is None:
        return slice(None, None)

    x0, y0, x1, y1 = bounds

    filt = None
    if not x0 is None:
        tfilt = x0 <= x
        filt = tfilt if filt is None else filt & tfilt

    if not x1 is None:
        tfilt = x <= x1
        filt = tfilt if filt is None else filt & tfilt

    if not y0 is None:
        tfilt = y0 <= y
        filt = tfilt if filt is None else filt & tfilt

    if not y1 is None:
        tfilt = y <= y1
        filt = tfilt if filt is None else filt & tfilt

    assert not filt is None
    return np.argwhere(filt)


def maskMesh(
    x: NDArray,
    y: NDArray,
    bounds: tuple[float | None, ...] | None = None,
    stride: tuple[int, int] | int = 1,
) -> tuple[slice, slice] | NDArray:

    if isinstance(stride, int):
        sx = sy = stride
    else:
        sx, sy = stride

    if bounds is None:
        return slice(None, None, sy), slice(None, None, sx)

    else:

        indices = maskScatter(x[::sx], y[::sy])
        assert not isinstance(indices, slice)
        indices[:, 0] *= sx
        indices[:, 1] *= sy
        return indices


def maskStructured(
    x: NDArray,
    y: NDArray,
    bounds: tuple[float | None, ...] | None = None,
    stride: tuple[int, int] | int = 1,
) -> tuple[slice, slice]:

    if bounds is None:
        return slice(None, None), slice(None, None)

    x0, y0, x1, y1 = bounds

    if x0 is None:
        i0 = 0
    else:
        i0 = np.argmin(np.abs(x - x0))

    if x1 is None:
        i1 = x.size
    else:
        i1 = np.argmin(np.abs(x - x1)) + 1

    if y0 is None:
        j0 = 0
    else:
        j0 = np.argmin(np.abs(y - y0))

    if y1 is None:
        j1 = y.size
    else:
        j1 = np.argmin(np.abs(y - y1)) + 1

    if isinstance(stride, int):
        sx = sy = stride
    else:
        sx, sy = stride

    sx = slice(i0, i1, sx)
    sy = slice(j0, j1, sy)
    return sx, sy
