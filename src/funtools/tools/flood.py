from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from re import A

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS, proj

from funtools.math.projection import PolygonProjection, RotationProjection

from ..math.geometry import Polygon
from ..math.grid import even_divide_slices
from ..model.simulation import Simulation
from ..parallel.simple import simple as eparallel
from ..ui import plot


@dataclass
class BaseData:

    def save(self, fpath: str):
        with open(fpath, "wb") as fh:
            pickle.dump(asdict(self), fh)

    @classmethod
    def load(cls, fpath: str) -> BaseData:
        with open(fpath, "rb") as fh:
            return cls(**pickle.load(fh))


@dataclass
class TotalFlood(BaseData):
    time: NDArray
    flood: NDArray
    total_area: float


def _getBatches(sim: Simulation, n_procs: int, t_min=None, t_max=None):

    steps, time = sim.data.getTimeSteps(t_min=t_min, t_max=t_max)

    if n_procs == 1:
        slices = [slice(i, i + 1) for i in range(steps.size)]
    else:
        nbatches = n_procs * 4
        n = steps.size
        m = n / nbatches
        if m < 2:
            nbatches = n_procs * 2
        slices = even_divide_slices(steps.size, nbatches)

    steps = [steps[s] for s in slices]
    return time, steps


def computeTotal(fpath: str, n_procs: int = 1, desc: str | None = None):

    # if isinstance(sim, str | Path):
    sim = Simulation(fpath)

    bathy = sim.data.read_bathy()
    land_mask = bathy > 0

    cell_area = sim.data.dx * sim.data.dy
    total_land = float(np.sum(land_mask) * cell_area)

    time, steps = _getBatches(sim, n_procs)
    args = [(fpath, s) for s in steps]

    args = eparallel(_computeTotal, args, n_procs=n_procs, desc=desc)
    data = np.concatenate(args)
    data = data*cell_area
    data = TotalFlood(
        time=time,
        flood=data,
        total_area=total_land,
    )

    return data


def _computeTotal(fpath: str, idxs: NDArray) -> NDArray:

    sim = Simulation(fpath)
    bathy = sim.data.read_bathy()
    land_mask = bathy > 0

    def _load(k: int) -> float:
        mask = sim.data.read_step("mask", k) == 1
        return np.sum(np.bitwise_and(land_mask, mask))

    return np.array([_load(k) for k in idxs])


def plotTotal(data: TotalFlood):

    scale = 1/3600
    tlim = t0, t1 = data.time[0]*scale, data.time[-1]*scale

    ylim = y0, y1 = 0, data.total_area

    gbl_opts = dict(
        frame_width=600,
        frame_height=250,
        xlabel="Time (hr)",
        xlim=tlim,
        ylabel="Area Flooding (%)",
        fontsize=plot.getFontSize(14, 13, 12),
        show_grid=True,
    )

    line_opts = dict()

    _data = (data.time*scale, (data.flood / y1) * 100)
    plt_flood = plot.curve(_data).opts(**line_opts)

    line_opts = dict(line_dash="dashed")

    _data = [(t0, y1), (t1, y1)]
    plt_max = plot.curve(_data)

    plt = plt_flood  # * plt_max

    return plt.opts(**gbl_opts)


@dataclass
class AreaFlood(BaseData):

    x: NDArray
    y: NDArray
    bounds: tuple[float, float, float, float]
    time: NDArray
    mask: NDArray
    percent: NDArray
    poly: Polygon


def computeArea(
    fpath: str, n_procs: int = 1, desc: str | None = None, t_min=None, t_max=None
):

    func = np.any
    func = np.sum
    sim = Simulation(fpath)

    time, steps = _getBatches(sim, n_procs, t_min=t_min, t_max=t_max)

    args = [(fpath, s, func) for s in steps]

    args = eparallel(_computeMask, args, n_procs=n_procs, desc=desc)
    # mask = func(np.stack(args), axis=0)

    mask, cum = [np.stack(s) for s in zip(*args)]

    mask = mask.any(axis=0)
    cum = cum.sum(axis=0)

    bathy = sim.data.read_bathy()

    mask[bathy < 0] = False

    cum = cum / time.size
    cum[~mask] = np.nan

    x = sim.data.x
    y = sim.data.y
    poly = Polygon.from_equipartition(x, y, mask, n_procs=n_procs)
    return AreaFlood(
        x=x, y=y, bounds=sim.data.bounds, time=time, mask=mask, percent=cum, poly=poly
    )


def _computeMask(fpath: str, idxs: NDArray, func) -> tuple[NDArray, NDArray]:

    sim = Simulation(fpath)

    def _load(k: int) -> float:
        return sim.data.read_step("mask", k) == 1

    data = np.stack([_load(k) for k in idxs])

    return data.any(axis=0), data.sum(axis=0)


def _getPolygonProjection(fpath: str, code: int, poly: Polygon) -> PolygonProjection:

    crs = CRS.from_user_input(code)
    proj = PolygonProjection(crs, None, poly)

    with open(fpath, "r") as fh:
        kwargs = json.load(fh)
        proj._rot_proj = RotationProjection.from_dict(kwargs)

    proj._items["rot"] = proj._items["src"]

    del proj._items["src"]

    return proj


def plotArea(data: AreaFlood, opts: dict = {}, fpath: str = None, code: int = None):

    if not fpath is None and not code is None:
        pproj = _getPolygonProjection(fpath, code, data.poly)

        key = "mrc"
        pproj.project("rot", key)
        poly = pproj._items[key]
    else:
        poly = self.poly

    plt = plot.polygons(poly).opts(**opts)

    tiles = plot.Tiles("EsriStreet")

    return tiles.merge(plt)
