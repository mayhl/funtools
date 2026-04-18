from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from netCDF4 import Dataset, Variable
from numpy.typing import NDArray

from funtools.io.input.grid import Grid
from funtools.io.input.output import Output, Stations, Time

from .input import Input


class LogConverter:
    """Converts FUNWAVE log files into NetCDF4 2D char array (# lines x max line length)."""

    def __init__(self, fpath: str | Path) -> None:

        with open(fpath, "r") as fh:
            lines = fh.read().splitlines()

        # Computing maximum line length
        nc = np.max([len(l) for l in lines])
        self._shape = shp = len(lines), nc

        # Initializing 2D array of blank chars and
        # converting each lines to 1D char array
        data = np.full(shp, "", dtype="U")
        for i, l in enumerate(lines):
            l = list(l)
            data[i, : len(l)] = l

        self._log_array = data

    def write(self, nc: Dataset) -> None:
        """Writes log file to netCDF4 Dataset"""
        nc.createDimension("log_lines", self._shape[0])
        nc.createDimension("log_length", self._shape[1])
        log = nc.createVariable("log", "S1", ("log_lines", "log_length"))
        log[:] = self._log_array


def _writeAttrs(
    nc: Dataset,
    name: str,
    datatype: str,
    dims: tuple[str],
    attrs: Variable,
    **extra,
) -> None:

    var = nc.createVariable(name, datatype, dims)
    extra.update(asdict(attrs))
    extra = {k: d for k, d in extra.items() if not d is None}

    rmkeys = ["latex"]
    for k in rmkeys:
        if k in extra:
            del extra[k]
    var.setncatts(extra)
    return var


class DimensionsWriter:

    def __init__(self, grid: Grid, time: Time, output: Output) -> None:

        self.__grid = grid
        self.__time = time
        self.__output = output

    def write(
        self, nc: Dataset, time: NDArray, grid_idxs: NDArray | None
    ) -> tuple[str, ...]:

        # TODO: Add option for native FORTRAN ordering?
        if grid_idxs is None:
            shp = ("time", "x", "y")
            nx = self.__grid.nx
            ny = self.__grid.ny
            xdim = ("x",)
            ydim = ("y",)
            nc.createDimension("x", nx)
            nc.createDimension("y", ny)
        else:
            shp = ("time", "sta_idx")
            nx = ny = grid_idxs.shape[0]
            xdim = ydim = ("sta_idx",)
            nc.createDimension("sta_idx", ny)

        nc.createDimension("time", None)

        tmeta = self.__time.getMeta()
        gmeta = self.__grid.getMeta()

        t = _writeAttrs(nc, "time", "f8", ("time",), tmeta)
        x = _writeAttrs(nc, "x", "f8", xdim, gmeta["x"])
        y = _writeAttrs(nc, "y", "f8", ydim, gmeta["y"])

        b = _writeAttrs(nc, "bathy", "f8", shp[1:], gmeta["bathy"])

        xx, yy = self.__grid.data.nodes
        bathy = self.__output.readBathy()

        if not grid_idxs is None:
            sta_idx = _writeAttrs(nc, "sta_idx", "i4", ydim, gmeta["sta_idx"])
            sta_idx[:] = grid_idxs[:, 0]
            i, j = [grid_idxs[:, k] for k in [2, 1]]
            xx = xx[i]
            yy = yy[j]
            bathy = bathy[j, i]

        x[:] = xx
        y[:] = yy
        t[:] = time
        b[:] = bathy.T
        return shp


class GroupWriter:

    def __init__(
        self, in_path: Path, input: Input, out_path: Path, is_group: bool
    ) -> None:
        self.__is_group = is_group
        self.__log = LogConverter(in_path.parent / "LOG.txt")
        self.__grid = DimensionsWriter(input.grid, input.time, input.output)
        self.__attrs = {f"input.{k}": d for k, d in input.toInputDict().items()}

        self.__path = out_path
        self.__nc_opts = {"format": "NETCDF4"}

        if self.__is_group:
            self.__nc = Dataset(out_path, "w", **self.__nc_opts)
            self.__nc.setncatts(self.__attrs)
            self.__log.write(self.__nc)

    def getGroup(
        self, name: str, suffix: str, time: NDArray, grid_idxs: NDArray | None = None
    ) -> tuple[Dataset, tuple[str, ...]]:

        if self.__is_group:
            nc = self.__nc.createGroup(name)

        else:
            name = self.__path.name
            if not suffix == "":
                name = name.replace(".nc", f"_{suffix}.nc")

            fpath = self.__path.parents[0] / name
            nc = Dataset(fpath, "w", **self.__nc_opts)
            nc.setncatts(self.__attrs)
            self.__log.write(nc)

        shp = self.__grid.write(nc, time, grid_idxs)
        return nc, shp

    def closeGroup(self, grp: Dataset):
        if not self.__is_group:
            grp.close()

    def close(self):
        if self.__is_group:
            self.__nc.close()


def convertTo(
    in_path: str | Path,
    out_path: str | Path,
    is_group: bool = True,
    field_suffix: str = "",
    mean_suffix: str = "mean",
    station_suffix: str = "stations",
) -> None:

    datatype = "f8"
    in_path = Path(in_path)
    input = Input.fromFile(in_path)
    input.setupCompleted(in_path)

    out_path = Path(out_path)

    gbl = GroupWriter(in_path, input, out_path, is_group)

    meta = input.output.getMeta()
    readers = input.output.getReadIterators()
    t = input.time.getAllTime()
    suffix = {"field": field_suffix, "mean": mean_suffix}

    for k in suffix.keys():
        if t[k].size == 0:
            continue
        grp, shp = gbl.getGroup(k, suffix[k], t[k])

        _writeFields(grp, readers[k], meta[k], datatype, shp)

        gbl.closeGroup(grp)

    if input.stations.hasStations():

        sta = input.stations
        t = sta.getTime()
        idxs = sta.getIndices()
        grp, shp = gbl.getGroup("stations", station_suffix, t, idxs)

        idxs = idxs[:, 0]
        meta = sta.getMeta()

        _writeStations(grp, sta, idxs, meta, datatype, shp)

        gbl.closeGroup(grp)

    gbl.close()


from tqdm.notebook import tqdm

from ..math.grid import even_divide_slices
from ..parallel.multi import simple as eparallel


def _writeStations(
    nc: Dataset,
    station: Stations,
    indices: NDArray,
    meta: dict[str, Variable],
    datatype: str,
    shape: tuple[str, ...],
) -> None:

    vars = {k: _writeAttrs(nc, k, datatype, shape, d) for k, d in meta.items()}

    n = indices.shape[0]
    m = 4
    nb = 4
    nprocs = 192
    subn = n / m
    subm = round(subn / nb)

    def subdived(sl: slice, m: int):
        n = sl.stop - sl.start
        return even_divide_slices(n, m, sl.start)

    sls = even_divide_slices(n, m)
    subsls = [subdived(sl, subm) for sl in sls]

    for sl, subsl in tqdm(list(zip(sls, subsls)), desc="Test"):
        args = [indices[it] for it in subsl]
        args = eparallel(station.read, nprocs, args)
        _, data = zip(*args)

        keys = list(data[0].keys())

        data = {k: np.concatenate([d[k] for d in data], axis=0) for k in keys}

        for k, var in vars.items():
            var[:, sl] = data[k].T


def _writeFields(
    nc: Dataset,
    readers: dict[str, list[Callable]],
    meta: dict[str, Variable],
    datatype: str,
    shape: tuple[str, ...],
) -> None:

    vars = {k: _writeAttrs(nc, k, datatype, shape, d) for k, d in meta.items()}

    for k, var in vars.items():

        for i, qread in tqdm(list(enumerate(readers[k])), desc=k):
            var[i, :, :] = qread().T
