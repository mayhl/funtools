from __future__ import annotations
from pathlib import Path
from typing import Tuple

from abc import ABC, abstractmethod


class _Parameter(ABC):

    def __init__(self, name: str, units: str, help: str, is_required: bool = False):

        self._name = name
        self._units = units
        self._help = help
        self._default = None
        self._value = None

        self._is_required = is_required

        self._requirements = {}
        self._hooks = []

    @property
    def has_default(self) -> bool:
        return not self._default is None

    @property
    def is_required(self) -> bool:
        return self.is_required

    def is_default(self) -> bool:
        if not self.has_default:
            return False
        return self._value is None

    @abstractmethod
    def format_funwave(self) -> str:
        pass

    def to_driver(self) -> str:
        return "%s = %s" % (self._name, self.format_funwave)


class IntParameter(_Parameter):

    def __init__(
        self, name: str, units: str, help: str, default_value: None | int = None
    ):
        super().__init__(name, units, help)
        self._default = default_value

    @property
    def value(self) -> None | int:

        if self._value is None:
            return self._default if self.has_default else None

        return self._value

    @value.setter
    def value(self, value: str | int):
        if isinstance(value, str):
            value = int(value)
        self._value = value

    def format_funwave(self) -> str:
        return "%d" % self.value


class CompilerParameter(_Parameter):
    pass


class Category:

    def __init__(self) -> None:
        self._items = {}
        self._subcategories = {}


class Test:

    def __init__(self):

        self._items = {}

        def _add_int(name: str, units: str, help: str) -> IntParameter:
            self._items[name] = IntParameter(name, units, help)
            return self._items[name]

        _add_int("Mglob", "", "")


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
            Wc_WK=0.0,
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
