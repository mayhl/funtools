from __future__ import annotations

from pathlib import Path

import numpy as np

from funtools.io.input import output

from .boundary import Boundary, Wavemaker
from .core import Config, Metadata, Parameter
from .grid import Bathymetry, Grid, Parallel
from .numerics import Friction, Numerics, Physics
from .output import Output, Stations, Time


class Input(Config):

    _title: str = "Main"

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

    def setupCompleted(self, path: str | Path):
        """Setups of simulations"""

        path = Path(path)
        if path.is_file():
            path = path.parent

        self.time.initCompleted(path=path)

        self.output.initCompleted(
            path=path, time=self.time, grid=self.grid, bathy=self.bathy
        )

        self.stations.initCompleted(path=path, output=self.output)

    @classmethod
    def getInputMap(cls) -> dict:
        """Returns dict mapping FUNWAVE inputs to nested keys."""
        map = super().getInputMap()

        values = list(map.values())
        uvalues = np.unique(values)
        assert len(values) == len(uvalues), np.sort(values)

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

        print(cls.getFileMap())
        # for k in cls.getFileMap():
        #
        #     old_path = items[k]
        #     assert isinstance(old_path, str)
        #
        #     if not Path(old_path).is_absolute():
        #         dpath = fpath.parent
        #         # TODO: Add runtime_path check
        #         new_path = (dpath / old_path).resolve()
        #         items[k] = str(new_path)
        #
        #         print(new_path)
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


# NOTE: Old code to remove, using old input parsing function for now
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
