from __future__ import annotations

from enum import Enum
from functools import partial
from os import PathLike
from pathlib import Path, PurePath
from re import A
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import NonNegativeInt, PositiveFloat

from .core import _IN_META_, _OUT_META_, Config, Metadata, Parameter, Variable
from .grid import Bathymetry, Grid


class Time(Config):

    _title: str = "Flags"

    t_end: PositiveFloat
    dt_plot: PositiveFloat
    dt_info: PositiveFloat
    t_plot_start: Optional[PositiveFloat] = None
    steady: Optional[PositiveFloat] = None
    dt_mean: Optional[PositiveFloat] = None

    class TimeMetadata(Metadata):

        t_end: Parameter = Parameter(
            name="total time", funwave_name="TOTAL_TIME", units="s"
        )
        dt_plot: Parameter = Parameter(
            name="plot interval", funwave_name="PLOT_INTV", units="s"
        )
        dt_info: Parameter = Parameter(
            name="screen interval", funwave_name="SCREEN_INTV", units="s"
        )

        t_plot_start: Parameter = Parameter(
            name="screen interval", funwave_name="SCREEN", units="s"
        )
        steady: Parameter = Parameter(
            name="Steady Time", funwave_name="STEADY_TIME", units="s"
        )

        dt_mean: Parameter = Parameter(
            name="Plot Mean Interval", funwave_name="T_INTV_mean", units="s"
        )

    meta: TimeMetadata = TimeMetadata()

    def initCompleted(self, **kwargs):
        spath: Path = kwargs["path"]

        tpath = spath / "time_dt.out"

        self.__t = t = np.loadtxt(tpath)[:, 0]
        self.__out_meta = Variable(**_OUT_META_["t"])

        if self.steady is None or self.dt_mean is None:
            self.__tm = np.array([])
            self.__is_mean = False
        else:
            t0 = self.steady
            dt = self.dt_mean
            t1 = self.t_end

            n = int((t1 - t0) / dt)
            self.__tm = np.arange(n) * dt + t0 + dt
            self.__is_mean = True

    def getTime(self) -> NDArray:
        assert not self.__t is None
        return self.__t

    def getMeanTime(self) -> NDArray:
        assert not self.__tm is None
        return self.__tm

    def getAllTime(self) -> dict[str, NDArray]:
        return {"field": self.__t, "mean": self.__tm}

    def getMeta(self) -> Variable:
        return self.__out_meta

    def hasMeanData(self) -> bool:
        return self.__is_mean


class OutputFlags(Config):

    _title: str = "Flags"

    age: Optional[bool] = False
    depth: Optional[bool] = False
    eta: Optional[bool] = True
    etamean: Optional[bool] = False
    fx: Optional[bool] = False
    fy: Optional[bool] = False
    gx: Optional[bool] = False
    gy: Optional[bool] = False
    hmax: Optional[bool] = False
    hmin: Optional[bool] = False
    hs: Optional[bool] = False
    mask9: Optional[bool] = False
    mask: Optional[bool] = True
    mfmax: Optional[bool] = False
    nu: Optional[bool] = False
    p: Optional[bool] = False
    q: Optional[bool] = False
    sourcex: Optional[bool] = False
    sourcey: Optional[bool] = False
    sxl: Optional[bool] = False
    sxr: Optional[bool] = False
    syl: Optional[bool] = False
    syr: Optional[bool] = False
    tmp: Optional[bool] = False
    u: Optional[bool] = False
    umax: Optional[bool] = False
    umean: Optional[bool] = False
    v: Optional[bool] = False
    vmean: Optional[bool] = False
    vormax: Optional[bool] = False

    class OutputFlagsMetadata(Metadata):
        nu: Parameter = Parameter(name="OUT_NU", funwave_name="OUT_NU")
        age: Parameter = Parameter(name="AGE", funwave_name="AGE")
        depth: Parameter = Parameter(name="DEPTH_OUT", funwave_name="DEPTH_OUT")
        u: Parameter = Parameter(name="U", funwave_name="U")
        v: Parameter = Parameter(name="V", funwave_name="V")
        eta: Parameter = Parameter(name="ETA", funwave_name="ETA")
        hmax: Parameter = Parameter(name="Hmax", funwave_name="Hmax")
        hmin: Parameter = Parameter(name="Hmin", funwave_name="Hmin")
        mfmax: Parameter = Parameter(name="MFmax", funwave_name="MFmax")
        umax: Parameter = Parameter(name="Umax", funwave_name="Umax")
        vormax: Parameter = Parameter(name="VORmax", funwave_name="VORmax")
        umean: Parameter = Parameter(name="Umean", funwave_name="Umean")
        vmean: Parameter = Parameter(name="Vmean", funwave_name="Vmean")
        etamean: Parameter = Parameter(name="ETAmean", funwave_name="ETAmean")
        mask: Parameter = Parameter(name="MASK", funwave_name="MASK")
        mask9: Parameter = Parameter(name="MASK9", funwave_name="MASK9")
        sxl: Parameter = Parameter(name="SXL", funwave_name="SXL")
        sxr: Parameter = Parameter(name="SXR", funwave_name="SXR")
        syl: Parameter = Parameter(name="SYL", funwave_name="SYL")
        syr: Parameter = Parameter(name="SYR", funwave_name="SYR")
        sourcex: Parameter = Parameter(name="SourceX", funwave_name="SourceX")
        sourcey: Parameter = Parameter(name="SourceY", funwave_name="SourceY")
        p: Parameter = Parameter(name="P", funwave_name="P")
        q: Parameter = Parameter(name="Q", funwave_name="Q")
        fx: Parameter = Parameter(name="Fx", funwave_name="Fx")
        fy: Parameter = Parameter(name="Fy", funwave_name="Fy")
        gx: Parameter = Parameter(name="Gx", funwave_name="Gx")
        gy: Parameter = Parameter(name="Gy", funwave_name="Gy")
        age: Parameter = Parameter(name="AGE", funwave_name="AGE")
        tmp: Parameter = Parameter(name="TMP", funwave_name="TMP")
        hs: Parameter = Parameter(name="WaveHeight", funwave_name="WaveHeight")

    meta: OutputFlagsMetadata = OutputFlagsMetadata()

    def getMap(self) -> dict[str, dict[str, str | list[str]]]:
        """Returns a dict mapping FUNWAVE output flags to output variable(s)."""

        def parse(map: dict) -> dict:
            return {getattr(self.meta, k).funwave_name: d for k, d in map.items()}

        return {k: parse(d) for k, d in self.__getMap().items()}

    def __getMap(self) -> dict[str, dict[str, str | list[str]]]:
        """Returns a dict mapping class output flags to output variable(s)."""
        return {
            "field": {"eta": "eta", "u": "u", "v": "v", "age": "age", "mask": "mask"},
            "mean": {
                "hs": "Hsig",
                "etamean": "etamean",
                "hs": ["Hsig", "Hrms", "Havg"],
                "umean": ["umean", "ulagm"],
                "vmean": ["vmean", "vlagm"],
            },
        }

    def initCompleted(self, **kwargs):

        def parse(map: dict) -> dict[str, Variable]:
            raw = [d for k, d in map.items() if getattr(self, k)]
            items = []
            for r in raw:
                if isinstance(r, list):
                    items.extend(r)
                else:
                    items.append(r)

            return {it: Variable(**_OUT_META_[it]) for it in items}

        self._out_meta = {k: parse(d) for k, d in self.__getMap().items()}

    def getMeta(self) -> dict[str, dict[str, Variable]]:
        return self._out_meta


class FieldOutputTypeEnum(Enum):
    ASCII = "ASCII"
    BINARY = "BINARY"


class Output(Config):
    _title: str = "Output"

    path: str = "output"
    type: FieldOutputTypeEnum = FieldOutputTypeEnum.ASCII
    breaking: bool = True
    flags: OutputFlags

    class OutputMetadata(Metadata):
        path: Parameter = Parameter(name="path", funwave_name="RESULT_FOLDER")
        breaking: Parameter = Parameter(name="breaking", funwave_name="SHOW_BREAKING")
        type: Parameter = Parameter(name="type", funwave_name="FIELD_IO_TYPE")

    meta: OutputMetadata = OutputMetadata()

    def model_post_init(self, context: Any, /) -> None:
        self.__out_meta = None

    def initCompleted(self, **kwargs):
        self.__path: Path = kwargs["path"]
        grid: Grid = kwargs["grid"]
        time: Time = kwargs["time"]
        bathy: Bathymetry = kwargs["bathy"]

        if bathy.isFile():
            self.__bathy = bathy.path
        else:
            self.__bathy = bathy.getGenerated()

        self.flags.initCompleted()

        self.__shape = grid.ny, grid.nx
        meta = self.getMeta()

        if len(meta["mean"]) > 0:
            assert (
                time.hasMeanData()
            ), "Mean outputs detected, but not STEADY_TIME or T_INTV_mean in input file."
        else:
            assert (
                not time.hasMeanData()
            ), "No mean outputs detect, but STEADY_TIME and T_INTV_mean specified input file."

        path = self.getPath()

        def parse(var):
            idxs = [int(it.name.split("_")[-1]) for it in path.glob(f"{var}_*")]
            if len(idxs) == 0:
                return 0

            idxs = np.sort(idxs)
            if idxs[-1] == 99999:
                idxs = idxs[:-1]

            assert np.all(np.diff(idxs) == 1)
            return idxs.size

        n = {k: np.array([parse(n) for n in d.keys()]) for k, d in meta.items()}

        for k, d in n.items():
            assert np.all(d == d[0]), f"{k} | {d} | {d[0]==d}"

        t = time.getAllTime()

        tinfo = {k: (d[0], t[k]) for k, d in n.items()}

        for k, (n, t) in tinfo.items():
            pass
            # assert t.size == n, f"{k} | {t.size} | {n}"

        self.__tinfo = {k: t for k, (_, t) in tinfo.items()}

    def getPath(self) -> Path:
        return self.__path / self.path

    def getMeta(self) -> dict[str, dict[str, Variable]]:

        if self.__out_meta is None:
            meta = self.flags.getMeta()

            # NOTE: Temporary check
            assert isinstance(self.__path, Path)
            found_titles = [
                "_".join(it.name.split("_")[:-1])
                for it in self.getPath().glob("*_00001")
            ]

            names_chk = list(meta["field"].keys()) + list(meta["mean"].keys())

            if not len(names_chk) == len(found_titles):
                missing = [n for n in found_titles if not n in names_chk]
                assert False, missing

            for n in names_chk:
                assert n in found_titles, n

            self.__out_meta = meta

        return self.__out_meta

    def _read(self, path: Path) -> NDArray:

        match self.type:
            case FieldOutputTypeEnum.ASCII:
                return self.__read_ascii(path)
            case FieldOutputTypeEnum.BINARY:
                return self.__read_binary(path)

        assert False, self.type

    def __read_ascii(self, path: Path) -> NDArray:
        return np.loadtxt(path)

    def __read_binary(self, path: Path) -> NDArray:
        return np.fromfile(path, dtype="<f8").reshape(self.__shape)

    def readBathy(self) -> NDArray:

        if isinstance(self.__bathy, PurePath):

            if self.flags.depth:
                return self.read("dep.out")
            else:
                return self.__read_ascii(self.__bathy)

        else:
            return self.__bathy

    def read(self, name: str) -> NDArray:
        return self._read(self.getPath() / name)

    def readIndex(self, name: str, index: int) -> NDArray:
        return self._read(self.getPath() / f"{name}_{index:05d}")

    def readTime(self, name: str, time: float) -> NDArray:

        pass

    def getReadIterators(self) -> dict[str, dict[str, list[Callable]]]:
        assert not self.__out_meta is None
        meta = self.__out_meta

        def cast(k: str, i: int, offset: int) -> Callable:
            return partial(self.readIndex, name=k, index=i + offset)

        def parse(
            d: dict[str, Variable], n: int, offset: int
        ) -> dict[str, list[Callable]]:
            return {k: [cast(k, i, offset) for i in range(n)] for k in d.keys()}

        offset = {"field": 0, "mean": 1}
        return {k: parse(d, self.__tinfo[k].size, offset[k]) for k, d in meta.items()}


class Stations(Config):

    _title: str = "Stations"

    n: Optional[NonNegativeInt] = None
    path: Optional[PathLike] = None
    dt: Optional[PositiveFloat] = None

    transects: Optional[Path] = None

    class StationsMetadata(Metadata):

        n: Parameter = Parameter(name="# stations", funwave_name="NumberStations")
        path: Parameter = Parameter(name="Path", funwave_name="STATIONS_FILE")
        transects: Parameter = Parameter(name="Path", funwave_name="STATIONS_MAP_FILE")

        dt: Parameter = Parameter(
            name="plot interval", funwave_name="PLOT_INTV_STATION"
        )
        # tmp: Parameter = Parameter(name="", funwave_name="")

    meta: StationsMetadata = StationsMetadata()

    def model_post_init(self, context: Any, /) -> None:
        self.__out_meta = None
        self.__n = None

    def hasStations(self) -> bool:

        if self.n is None:
            return False

        return self.n > 0

    def initCompleted(self, **kwargs):
        path: Path = kwargs["path"]
        output: Output = kwargs["output"]

        self.__path = output.getPath()
        if not self.hasStations():
            self.__idxs = np.zeros((0, 3))
            self.__t = np.array([])
            self.__n = 0
            return

        assert not self.path is None
        idxs = np.loadtxt(path / self.path).astype(int) - 1

        self.__idxs = np.concat(
            [np.arange(1, self.n + 1)[:, None], np.fliplr(idxs)], axis=1
        )

        self.__n = None
        k = self.__idxs[0, 0]
        t, data = self.read(k)

        self.__out_meta = {k: Variable(**_OUT_META_[k]) for k in data.keys()}

        i = np.argwhere(np.diff(t) <= 0) + 1

        if len(i) > 0:
            n = i[0][0]
        else:
            n = t.size

        self.__t = t[:n]
        self.__n = n

    def getMeta(self) -> dict[str, Variable]:
        assert not self.__out_meta is None
        return self.__out_meta

    def getIndices(self) -> NDArray:
        return self.__idxs

    def getTime(self) -> NDArray:
        return self.__t

    def read(self, index: int | list[int]) -> tuple[NDArray, dict[str, NDArray]]:

        if not isinstance(index, np.ndarray):
            return self._read(index)

        args = [self._read(i) for i in index]
        t = args[0][0]

        data = {k: np.stack([d[k] for _, d in args]) for k in args[0][1].keys()}

        return t, data

    def _read(self, index: int) -> tuple[NDArray, dict[str, NDArray]]:

        path = self.__path / f"sta_{index:04d}"
        data = np.loadtxt(path)

        if not self.__n is None:
            data = data[: self.__n, :]

        if data.shape[1] == 4:
            keys = ["eta", "u", "v"]

        else:
            raise NotImplementedError(data.shape[1])

        return data[:, 0], {k: data[:, i] for i, k in enumerate(keys, start=1)}
