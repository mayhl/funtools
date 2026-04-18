from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from funtools.model.stations import Station, Stations


def getTransectIndices(fpath: str, n_transects: int) -> NDArray:

    sta = Stations(fpath)
    n = len(sta._items)

    print(n % n_transects, n, n_transects)
    assert n % n_transects == 0, n

    m = n // n_transects

    return np.arange(n).reshape((n_transects, m))


from ..io.input import Input
from ..math.grid import even_divide_slices
from ..parallel.multi import simple as eparallel


def loadStations2(fpath: str, idxs: NDArray, nprocs: int):

    input = Input.fromFile(fpath)
    input.setupCompleted(fpath)

    sta = input.stations

    x, y = input.grid.data.nodes

    subidx = sta.getIndices()[idxs, :]
    i, j = [subidx[:, k] for k in [2, 1]]

    x = x[i]
    y = y[j]
    h = input.output.readBathy()[j, i]

    n = idxs.size
    m = 2 * nprocs
    if n < m:
        args = [np.array([i + 1]) for i in idxs]
    else:
        args = [idxs[s] + 1 for s in even_divide_slices(n, m)]

    args = eparallel(sta.read, nprocs, args)
    t, data = zip(*args)

    t = t[0]
    keys = list(data[0].keys())

    data = {k: np.concatenate([d[k] for d in data], axis=0) for k in keys}

    return t, x, y, h, data["eta"]
    # return t, data


def loadStations(fpath: str, idxs: NDArray):

    sta = Stations(fpath)

    def parse(s: Station):
        t, eta = s.eta_timeseries()
        return t, s.x, s.y, s.h, eta

    data = [parse(sta._items[i]) for i in idxs]
    t, *args = zip(*data)

    t = t[0]

    x, y, h, eta = [np.stack(s) for s in args]

    i = np.argwhere(np.diff(t) <= 0) + 1

    if len(i) > 0:
        i = i[0][0]

        t = t[:i]
        eta = eta[:, :i]
    return t, x, y, h, eta


def interpolate(t: NDArray, data: NDArray, dt_target: float) -> tuple[NDArray, NDArray]:
    nt = int((t[-1] - t[0]) // dt_target)
    ti = np.arange(nt + 1) * dt_target
    datai = interp1d(t, data, axis=-1)(ti)
    return ti, datai


def computeHmo(energy: NDArray) -> NDArray:
    return 4 * np.sqrt(energy)


def computeEnergy(freq: NDArray, spec: NDArray, freq_cut=[]) -> NDArray | list[NDArray]:

    i_cut = [np.argmin(np.abs(fc - freq)) for fc in freq_cut]
    istart = np.insert(i_cut, 0, 0).astype(int)
    iend = np.append(i_cut, len(freq)).astype(int)
    slices = [slice(*a) for a in zip(istart, iend)]

    energy = [np.trapezoid(spec[:, s], freq[s]) for s in slices]

    if len(slices) == 1:
        return energy[0]
    else:
        return energy
