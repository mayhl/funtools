from __future__ import annotations

from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union, get_origin

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    FiniteFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from pydantic.types import PathType

# TODO: Figure out good solution to allow path validation or skip
# Path = str


class Parameter(BaseModel):
    name: str
    funwave_name: str
    short_name: Optional[str] = None
    long_name: Optional[str] = None
    units: Optional[str] = None


class LinkedMetadata(BaseModel):

    def validate(self):
        """"""
        assert hasattr(self, "meta"), f"{self.__class__} missing meta attribute."

        meta = self.meta

        assert (
            meta.validate()
        ), f"{self.__class__} meta class, {meta.__class__}, is not valid."

        assert issubclass(
            meta.__class__, BaseModel
        ), f"{self.__class__} meta attribute is not a subclass of BaseModel."

        for f in self.model_fields_set:
            assert hasattr(meta, f), f"{self.__class__} meta missing {f} attribute."

        for f in meta.model_fields_set:
            assert hasattr(
                self, f
            ), f"{self.__init__} missing {f} attribute defined in meta."

        for f in self.model_fields_set:
            attr = getattr(self, f)
            if issubclass(attr.__class__, LinkedMetadata):
                attr.validate()

    def toInputDict(self) -> dict:

        items = [
            (n, v, issubclass(v.__class__, LinkedMetadata))
            for n, v in self
            if not v is None and not n == "meta"
        ]

        nested = [v.toInputDict() for n, v, f in items if f]

        assert hasattr(self, "meta")
        local = {getattr(self.meta, n).funwave_name: v for n, v, f in items if not f}

        for k, d in local.items():
            if issubclass(d.__class__, Enum):
                local[k] = d.value

            if issubclass(d.__class__, PathLike):
                local[k] = str(d)

            if issubclass(d.__class__, bool):
                local[k] = "T" if d else "F"
        for n in nested:
            local.update(n)

        return local

    @classmethod
    def _getNested(cls) -> tuple[list[tuple[str, type[LinkedMetadata]]], type[Any]]:
        """Returns nested fields as key/class type pairs, and meta field class type."""
        items = cls.model_fields

        assert "meta" in cls.model_fields
        MetaCls = cls.model_fields["meta"].annotation

        assert not MetaCls is None

        def parse(d) -> None | type[LinkedMetadata]:
            obj = d.annotation
            if get_origin(obj) is Union:
                obj = obj.__class__

            if issubclass(obj, LinkedMetadata):
                return obj
            else:
                return None

        nested = [(k, parse(d)) for k, d in items.items()]

        return [(k, d) for k, d in nested if not d is None], MetaCls

    @classmethod
    def getInputMap(cls) -> dict:
        """Returns dict mapping nested keys to all FUNWAVE inputs"""

        nested, MetaCls = cls._getNested()

        local = MetaCls.getInputMap()
        nested = [((k,), d.getInputMap()) for k, d in nested]
        nested = {k1 + k2: d for k1, sub in nested for k2, d in sub.items()}

        return {**local, **nested}

    @classmethod
    def getFileMap(cls) -> dict:
        """Returns dict mapping nested keys to FUNWAVE input file"""
        nested, MetaCls = cls._getNested()
        items = cls.model_fields.items()
        local = [k for k, d in items if d.annotation is Path]

        if len(local) > 0:
            meta = MetaCls()
            local = {(k,): getattr(meta, k).funwave_name for k in local}
        else:
            local = {}

        nested = [((k,), d.getFileMap()) for k, d in nested]
        nested = {k1 + k2: d for k1, sub in nested for k2, d in sub.items()}

        return {**local, **nested}


class Metadata(BaseModel):

    def validate(self):
        return not any(
            [isinstance(getattr(self, f), Parameter) for f in self.model_fields_set]
        )

    @classmethod
    def getInputMap(cls) -> dict:
        """Returns dict mapping local variables to FUNWAVE input."""
        meta = cls()
        return {(f,): getattr(meta, f).funwave_name for f in cls.model_fields}


class BathyTypeEnum(Enum):
    FLAT = "FLAT"
    SLOPE = "SLOPE"
    FILE = "DATA"


class Bathymetry(LinkedMetadata):

    input: BathyTypeEnum
    file: PathLike
    water_level: FiniteFloat = 0.0

    class BathymetryMetadata(Metadata):
        input: Parameter = Parameter(name="type", funwave_name="DEPTH_TYPE")

        file: Parameter = Parameter(name="file path", funwave_name="DEPTH_FILE")
        water_level: Parameter = Parameter(
            name="Water Level", funwave_name="WaterLevel"
        )

    meta: BathymetryMetadata = BathymetryMetadata()


class Boundary(LinkedMetadata):
    sponge: Sponge
    tide: Tide

    periodic: bool = False

    class BoundaryMetadata(Metadata):
        periodic: Parameter = Parameter(name="periodic y", funwave_name="PERIODIC")
        #: Parameter = Parameter(name="", funwave_name="")

    meta: BoundaryMetadata = BoundaryMetadata()


class Friction(LinkedMetadata):

    is_file: bool = False
    cd: NonNegativeFloat = 0.0

    class FrictionMetadata(Metadata):
        is_file: Parameter = Parameter(name="is file", funwave_name="Friction_Matrix")
        cd: Parameter = Parameter(name="Cd", funwave_name="Cd")
        #: Parameter = Parameter(name="", funwave_name="")
        pass

    meta: FrictionMetadata = FrictionMetadata()


class Grid(LinkedMetadata):
    dx: PositiveFloat
    dy: PositiveFloat

    nx: PositiveInt
    ny: PositiveInt

    path: Optional[PathLike] = None

    class GridMetadata(Metadata):
        dx: Parameter = Parameter(name="dx", funwave_name="DX", units="m")
        dy: Parameter = Parameter(name="dy", funwave_name="DY", units="m")
        nx: Parameter = Parameter(name="nx", funwave_name="Mglob")
        ny: Parameter = Parameter(name="ny", funwave_name="Nglob")
        path: Parameter = Parameter(name="proj path", funwave_name="ProjectionPath")

    meta: GridMetadata = GridMetadata()


class TimeSchemeEnum(Enum):
    RK4 = "Runge_Kutta"


class Numerics(LinkedMetadata):

    cfl: PositiveFloat = 0.5
    froude_cap: PositiveFloat = 3.5

    scheme: TimeSchemeEnum = TimeSchemeEnum.RK4
    min_depth: PositiveFloat = 0.01
    min_depth_frc: PositiveFloat = 0.01

    class NumericsMetadata(Metadata):
        cfl: Parameter = Parameter(name="CFL", funwave_name="CFL")
        scheme: Parameter = Parameter(name="Time Scheme", funwave_name="Time_Scheme")
        froude_cap: Parameter = Parameter(name="Froude Cap", funwave_name="FroudeCap")
        min_depth: Parameter = Parameter(name="Minimum Depth", funwave_name="MinDepth")
        min_depth_frc: Parameter = Parameter(
            name="Minimum Depth Friction", funwave_name="MinDepthFrc"
        )

    meta: NumericsMetadata = NumericsMetadata()


class OutputFlags(LinkedMetadata):

    age: bool = False
    depth: bool = False
    eta: bool = True
    etamean: bool = False
    fx: bool = False
    fy: bool = False
    gx: bool = False
    gy: bool = False
    hmax: bool = False
    hmin: bool = False
    hs: bool = False
    mask9: bool = False
    mask: bool = True
    mfmax: bool = False
    nu: bool = False
    p: bool = False
    q: bool = False
    sourcex: bool = False
    sourcey: bool = False
    sxl: bool = False
    sxr: bool = False
    syl: bool = False
    syr: bool = False
    tmp: bool = False
    u: bool = False
    umax: bool = False
    umean: bool = False
    v: bool = False
    vmean: bool = False
    vormax: bool = False

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


class FieldOutputTypeEnum(Enum):
    ASCII = "ASCII"
    BINARY = "BINARY"


class Output(LinkedMetadata):
    path: str = "output"
    type: FieldOutputTypeEnum = FieldOutputTypeEnum.ASCII
    breaking: bool = True
    flags: OutputFlags

    class OutputMetadata(Metadata):
        path: Parameter = Parameter(name="path", funwave_name="RESULT_FOLDER")
        breaking: Parameter = Parameter(name="breaking", funwave_name="SHOW_BREAKING")
        type: Parameter = Parameter(name="type", funwave_name="FIELD_IO_TYPE")

    meta: OutputMetadata = OutputMetadata()


class Parallel(LinkedMetadata):
    nx: PositiveInt
    ny: PositiveInt

    class ParallelMetadata(Metadata):
        nx: Parameter = Parameter(name="px", funwave_name="PX")
        ny: Parameter = Parameter(name="py", funwave_name="PY")

    meta: ParallelMetadata = ParallelMetadata()


class Physics(LinkedMetadata):

    dispersion: bool = True
    viscosity: bool = True

    gamma1: PositiveFloat
    gamma2: PositiveFloat
    gamma3: PositiveFloat
    beta: FiniteFloat
    swe_dep: PositiveFloat
    cbrk1: PositiveFloat = 0.45
    cbrk2: PositiveFloat = 0.35

    class PhysicsMetadata(Metadata):
        dispersion: Parameter = Parameter(name="dispersion", funwave_name="DISPERSION")
        viscosity: Parameter = Parameter(
            name="viscosity", funwave_name="VISCOSITY_BREAKING"
        )
        gamma1: Parameter = Parameter(name="gamma1", funwave_name="Gamma1")
        gamma2: Parameter = Parameter(name="gamma2", funwave_name="Gamma2")
        gamma3: Parameter = Parameter(name="gamma3", funwave_name="Gamma3")
        beta: Parameter = Parameter(name="beta ref", funwave_name="Beta_ref")
        swe_dep: Parameter = Parameter(name="beta ref", funwave_name="SWE_ETA_DEP")
        cbrk1: Parameter = Parameter(name="Cbkr1", funwave_name="Cbrk1")
        cbrk2: Parameter = Parameter(name="Cbkr2", funwave_name="Cbrk2")
        #: Parameter = Parameter(name="", funwave_name="")
        pass

    meta: PhysicsMetadata = PhysicsMetadata()


class Sponge(LinkedMetadata):
    west_width: NonNegativeFloat = 0.0
    east_width: NonNegativeFloat = 0.0
    north_width: NonNegativeFloat = 0.0
    south_width: NonNegativeFloat = 0.0

    diffussion: bool = False
    friction: bool = False
    direct: bool = False

    csp: PositiveFloat
    cd: PositiveFloat
    r: PositiveFloat
    a: PositiveFloat

    class SpongeMetaData(Metadata):
        west_width: Parameter = Parameter(
            name="west_width", funwave_name="Sponge_west_width"
        )
        east_width: Parameter = Parameter(
            name="east_width", funwave_name="Sponge_east_width"
        )
        south_width: Parameter = Parameter(
            name="south_width", funwave_name="Sponge_south_width"
        )
        north_width: Parameter = Parameter(
            name="north_width", funwave_name="Sponge_north_width"
        )
        diffussion: Parameter = Parameter(
            name="diffussion", funwave_name="DIFFUSION_SPONGE"
        )
        friction: Parameter = Parameter(name="friction", funwave_name="FRICTION_SPONGE")
        direct: Parameter = Parameter(name="direct", funwave_name="DIRECT_SPONGE")
        csp: Parameter = Parameter(name="csp", funwave_name="Csp")
        cd: Parameter = Parameter(name="Cd", funwave_name="CDsponge")
        r: Parameter = Parameter(name="R", funwave_name="R_sponge")
        a: Parameter = Parameter(name="R", funwave_name="A_sponge")
        #: Parameter = Parameter(name="", funwave_name="")

    meta: SpongeMetaData = SpongeMetaData()


class Stations(LinkedMetadata):

    n: NonNegativeInt = 0
    path: Optional[PathLike] = None
    dt: Optional[PositiveFloat] = None

    map: Optional[Path] = None

    class StationsMetadata(Metadata):

        n: Parameter = Parameter(name="# stations", funwave_name="NumberStations")
        path: Parameter = Parameter(name="Path", funwave_name="STATIONS_FILE")
        transects: Parameter = Parameter(name="Path", funwave_name="STATIONS_MAP_FILE")

        dt: Parameter = Parameter(
            name="plot interval", funwave_name="PLOT_INTV_STATION"
        )
        # tmp: Parameter = Parameter(name="", funwave_name="")

    meta: StationsMetadata = StationsMetadata()


class WavemakerTypeEnum(Enum):
    IRR = "WK_IRR"
    DATA = "WK_DATA2D"


class Wavemaker(LinkedMetadata):

    xc: PositiveFloat
    yc: PositiveFloat = 0.0
    y_width: PositiveFloat = 99999.0
    delta: PositiveFloat
    type: WavemakerTypeEnum

    cbrk: PositiveFloat
    depth: PositiveFloat
    file: Path

    class WavemakerMetadata(Metadata):
        xc: Parameter = Parameter(name="xc", funwave_name="Xc_WK")
        yc: Parameter = Parameter(name="yc", funwave_name="Yc_WK")
        y_width: Parameter = Parameter(name="y_width", funwave_name="Ywidth_WK")
        delta: Parameter = Parameter(name="delta", funwave_name="Delta_WK")
        type: Parameter = Parameter(name="type", funwave_name="WAVEMAKER")
        file: Parameter = Parameter(name="file path", funwave_name="WaveCompFile")
        depth: Parameter = Parameter(name="depth", funwave_name="DEP_WK")
        cbrk: Parameter = Parameter(name="cbrk", funwave_name="WAVEMAKER_Cbrk")
        # : Parameter = Parameter(name="", funwave_name="")

    meta: WavemakerMetadata = WavemakerMetadata()


class Tide(LinkedMetadata):

    type: bool = False

    class TideMetadata(Metadata):
        type: Parameter = Parameter(name="type", funwave_name="TIDAL_BC_ABS")
        #: Parameter = Parameter(name="", funwave_name="")
        pass

    meta: TideMetadata = TideMetadata()


class Time(LinkedMetadata):

    t_end: PositiveFloat
    dt_plot: PositiveFloat
    dt_info: PositiveFloat

    steady: PositiveFloat
    dt_mean: PositiveFloat

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

        steady: Parameter = Parameter(
            name="Steady Time", funwave_name="STEADY_TIME", units="s"
        )

        dt_mean: Parameter = Parameter(
            name="Plot Mean Interval", funwave_name="T_INTV_mean", units="s"
        )

    meta: TimeMetadata = TimeMetadata()


class Tmp(LinkedMetadata):

    class TmpMetadata(Metadata):
        #: Parameter = Parameter(name="", funwave_name="")
        pass

    meta: TmpMetadata = TmpMetadata()


class Input(LinkedMetadata):

    title: str = "FUNWAVE Simulation"
    parallel: Parallel
    grid: Grid
    bathy: Bathymetry
    time: Time
    boundary: Boundary
    wavemaker: Wavemaker
    stations: Stations
    numerics: Numerics
    physics: Physics
    friction: Friction
    output: Output

    class InputMetadata(Metadata):
        title: Parameter = Parameter(name="Title", funwave_name="TITLE")
        # path: Parameter = Parameter(name="DUMMY_PATH", funwave_name="DUMMY_PATH")

    meta: InputMetadata = InputMetadata()

    @classmethod
    def getInputMap(cls) -> dict:
        """Returns dict mapping FUNWAVE inputs to nested keys."""
        map = super().getInputMap()

        values = list(map.values())
        uvalues = np.unique(values)
        assert len(values) == len(uvalues), cls

        map = {d: k for k, d in map.items()}

        return map

    @classmethod
    def getFileMap(cls) -> dict:
        """Returns dict mapping FUNWAVE file inputs to nested keys."""
        map = super().getFileMap()
        return {d: k for k, d in map.items()}

    @classmethod
    def fromFile(cls, fpath: Path | str, runtime_path: Path | None = None) -> Input:
        """Initialize class from FUNWAVE input file."""
        # TODO: Replace old code

        fpath = Path(fpath)
        items = InputFile.from_file(fpath)._items

        # Checking input file keys against nested classes
        ignore_list = ["C_smg"]
        map = cls.getInputMap()
        for key in items:
            if not key in ignore_list:
                assert key in map, f"Key '{key}' not found in map."

        for k in ignore_list:
            del items[k]

        for k in cls.getFileMap():

            old_path = items[k]
            assert isinstance(old_path, str)

            if not Path(old_path).is_absolute():
                dpath = fpath.parent
                # TODO: Add runtime_path check
                new_path = (dpath / old_path).resolve()
                items[k] = str(new_path)

        # Swapping FUNWAVE key with nested class keys and attaching file value
        map = {d: items[k] for k, d in map.items() if k in items}

        # Expanding keys into nested dicts
        args = {}
        for keys, val in map.items():
            k = keys[0]
            if not k in args:
                args[k] = {}

            parent = args
            child = args[k]
            for k in keys[1:]:
                if not k in child:
                    child[k] = {}

                parent = child
                child = child[k]

            parent[k] = val

        input = Input(**args)

        return input


# NOTE: Due to nested models defined out of order
#       Fixes incomplete references
Boundary.model_rebuild()


class InputFile:

    def __init__(self):
        # Setting up default values
        self._items = dict(
            RESULT_FOLDER="output",
            Sponge_west_width=0.0,
            Sponge_east_width=0.0,
            Sponge_north_width=0.0,
            Sponge_south_width=0.0,
            CFL=0.5,
            #  Wc_WK=0.0,
        )

    def __getitem__(self, key: str) -> str | bool | float | int:
        return self._items[key]

    def __setitem__(self, key: str, val: str | bool | float | int) -> None:
        self._items[key] = val

    @classmethod
    def _split_first(cls, line: str, char: str) -> Tuple[str, str]:
        first, *second = line.split(char)
        second = char.join(second)
        return first, second

    @classmethod
    def _filter_comment(cls, line: str) -> Tuple[bool, str, str | None]:
        if "!" not in line:
            return False, line, None
        first, second = cls._split_first(line, "!")
        return True, first, second

    @classmethod
    def _parse_str(cls, val: str) -> str | int | bool | float:

        def cast_type(val, cast) -> str | int | bool | float | None:
            try:
                return cast(val)
            except ValueError:
                return None

        ival = cast_type(val, int)
        fval = cast_type(val, float)

        if ival is None and fval is None:
            if type(val) is str and len(val) == 1:
                if val[0] == "T":
                    return True
                if val[0] == "F":
                    return False

            return str(val)

        elif ival is not None and fval is not None:
            return ival if ival == fval else fval

        elif fval is not None:  # and ival is None
            return fval

        else:  # fval is None, ival is not None
            # Case should not be possible
            raise Exception("Unexpected State")

    @classmethod
    def from_file(cls, path: Path) -> InputFile:

        path = Path(path)
        if path.is_dir():
            path = path / "input.txt"
        input = InputFile()

        with open(path, "r") as fh:
            lines = fh.readlines()

        for line in lines:

            if not "=" in line:
                continue
            first, second = cls._split_first(line, "=")

            is_comment, name, _ = cls._filter_comment(first.strip())
            if is_comment:
                continue

            is_comment, val_str, _ = cls._filter_comment(second.strip())

            input[name] = cls._parse_str(val_str)

        return input

    def create_file(self, path: Path) -> None:

        # Map of parameters to category/headings
        # File ordering is determined by map
        _CATEGORY_MAP_ = {
            "General": ["TITLE"],
            "Parallel": ["PX", "PY"],
            "Grid": ["DX", "DY", "Mglob", "Nglob", "StretchGrid"],
            "Bathy": ["DEPTH_TYPE", "DEPTH_FILE", "WaterLevel"],
            "Time": ["TOTAL_TIME", "PLOT_INTV", "SCREEN_INTV"],
            "Hot Start": ["HOT_START", "INI_UVZ"],
            "Wave Maker": [
                "WAVEMAKER",
                "WAVE_DATA_TYPE",
                "DEP_WK",
                "Xc_WK",
                "Yc_WK",
                "FreqPeak",
                "FreqMin",
                "FreqMax",
                "Hmo",
                "GammaTMA",
                "Sigma_Theta",
                "Delta_WK",
                "EqualEnergy",
                "Nfreq",
                "Ntheta",
                "alpha_c",
                "Tperiod",
                "AMP_WK",
                "ThetaPeak",
            ],
            "Boundary Conditions": [
                "PERIODIC",
                "DIFFUSION_SPONGE",
                "FRICTION_SPONGE",
                "DIRECT_SPONGE",
                "Csp",
                "CDsponge",
                "Sponge_west_width",
                "Sponge_east_width",
                "Sponge_south_width",
                "Sponge_north_width",
                "R_sponge",
                "A_sponge",
            ],
            "Tidal Boundary Forcing": [
                "TIDAL_BC_GEN_ABS",
                "TideBcType",
                "TideWest_ETA",
                "TIDAL_BC_ABS",
                "TideWestFileName",
            ],
            "Numerics": [
                "Gamma1",
                "Gamma2",
                "Gamma3",
                "Beta_ref",
                "HIGH_ORDER",
                "CONSTRUCTION",
                "CFL",
                "FroudeCap",
                "MinDepth",
                "MinDepthFrc",
                "Time_Scheme",
            ],
            "Breaking": [
                "DISPERSION",
                "SWE_ETA_DEP",
                "SHOW_BREAKING",
                "VISCOSITY_BREAKING",
                "Cbrk1",
                "Cbrk2",
                "WAVEMAKER_Cbrk",
            ],
            "Friction": ["Friction_Matrix", "Cd", "Cd_file"],
            "Mixing": ["STEADY_TIME", "T_INTV_mean", "C_smg"],
            "Stations": ["NumberStations", "STATIONS_FILE", "PLOT_INTV_STATION"],
            "Output": [
                "FIELD_IO_TYPE",
                "DEPTH_OUT",
                "U",
                "V",
                "ETA",
                "Hmax",
                "Hmin",
                "MFmax",
                "Umax",
                "VORmax",
                "Umean",
                "Vmean",
                "ETAmean",
                "MASK",
                "MASK9",
                "SXL",
                "SXR",
                "SYL",
                "SYR",
                "SourceX",
                "SourceY",
                "P",
                "Q",
                "Fx",
                "Fy",
                "Gx",
                "Gy",
                "AGE",
                "TMP",
                "WaveHeight",
                "OUT_NU",
            ],
        }

        # Validating parameters are in map
        for key in self._items:
            is_found = False
            for subparams in _CATEGORY_MAP_.values():
                if key in subparams:
                    is_found = True
                    break
            if not is_found:
                raise Exception("Parameter '%s' has no category." % key)

        # Writing to file
        if path.is_dir():
            path = path / "input.txt"
        with open(path, "w") as fh:
            for category, subparams in _CATEGORY_MAP_.items():
                is_first = True

                for subparam in subparams:
                    if subparam in self._items:

                        # Only create banner if at least one parameter is found
                        if is_first:
                            fh.write(self._get_banner(category))
                            is_first = False

                        fh.write(self._get_parameter_line(subparam, self[subparam]))

    def _get_banner(self, title: str, indent: int = 10, max_length: int = 80) -> str:

        banner = "! " + "".join(["-"] * indent)
        banner += " " + title + " "
        n = max_length - len(banner)
        if n < 1:
            n = 1
        banner += "".join(["-"] * n)
        return "\n" + banner + "\n"

    def _get_parameter_line(self, name: str, value: str | bool | float | int) -> str:

        if isinstance(value, bool):
            value_str = "T" if value else "F"
        elif isinstance(value, int):
            value_str = "%d" % value
        elif isinstance(value, float):
            value_str = "%f" % value

            # Removing trailing 0's
            if value_str[-1] == "0":
                value_str = value_str.strip("0")
                if value_str[0] == ".":
                    value_str = "0" + value_str
                if value_str[-1] == ".":
                    value_str += "0"

        else:
            value_str = value

        return "%s = %s\n" % (name, value_str)

    def get_int(self, key: str) -> int:
        return self[key]  # type: ignore

    def get_flt(self, key: str) -> float:
        return self[key]  # type: ignore

    def get_str(self, key: str) -> str:
        return self[key]  # type: ignore

    def get_bool(self, key: str) -> bool:
        return self[key]  # type: ignore
