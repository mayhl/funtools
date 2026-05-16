from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import FiniteFloat, NonNegativeFloat, PositiveFloat, PositiveInt

from .core import Config, Metadata, Parameter

# TODO: Figure out good solution to allow path validation or skip
# Path = str


class WavemakerTypeEnum(Enum):
    IRR = "WK_IRR"
    DATA = "WK_DATA2D"


class Wavemaker(Config):
    _title: str = "Wavemaker"

    xc: Optional[PositiveFloat]
    yc: Optional[PositiveFloat] = 0.0
    y_width: Optional[PositiveFloat] = 99999.0
    delta: Optional[PositiveFloat] = None
    hmo: Optional[PositiveFloat] = None
    freq_min: Optional[PositiveFloat] = None
    freq_peak: Optional[PositiveFloat] = None
    freq_max: Optional[PositiveFloat] = None
    gamma: Optional[PositiveFloat] = None
    sigma_theta: Optional[PositiveFloat] = None
    theta_peak: Optional[FiniteFloat] = None
    type: Optional[WavemakerTypeEnum] = None
    nfreq: Optional[PositiveInt] = None
    ntheta: Optional[PositiveInt] = None
    equal_energy: Optional[bool] = None
    cbrk: Optional[PositiveFloat] = None
    depth: Optional[PositiveFloat]
    path: Optional[Path] = None

    class WavemakerMetadata(Metadata):
        xc: Parameter = Parameter(name="xc", funwave_name="Xc_WK")
        yc: Parameter = Parameter(name="yc", funwave_name="Yc_WK")
        y_width: Parameter = Parameter(name="y_width", funwave_name="Ywidth_WK")
        delta: Parameter = Parameter(name="delta", funwave_name="Delta_WK")
        type: Parameter = Parameter(name="type", funwave_name="WAVEMAKER")
        hmo: Parameter = Parameter(name="hmo", funwave_name="Hmo")
        freq_peak: Parameter = Parameter(name="freq peak", funwave_name="FreqPeak")
        freq_min: Parameter = Parameter(name="freq min", funwave_name="FreqMin")
        freq_max: Parameter = Parameter(name="freq max", funwave_name="FreqMax")
        nfreq: Parameter = Parameter(name="Nfreq", funwave_name="Nfreq")
        ntheta: Parameter = Parameter(name="Ntheta", funwave_name="Ntheta")
        gamma: Parameter = Parameter(name="gamma TMA", funwave_name="GammaTMA")
        sigma_theta: Parameter = Parameter(
            name="sigma theta", funwave_name="Sigma_Theta"
        )
        theta_peak: Parameter = Parameter(
            name="Peak Direction", funwave_name="ThetaPeak"
        )
        equal_energy: Parameter = Parameter(
            name="equal energy", funwave_name="EqualEnergy"
        )
        path: Parameter = Parameter(name="file path", funwave_name="WaveCompFile")
        depth: Parameter = Parameter(name="depth", funwave_name="DEP_WK")
        cbrk: Parameter = Parameter(name="cbrk", funwave_name="WAVEMAKER_Cbrk")
        # : Parameter = Parameter(name="", funwave_name="")

    meta: WavemakerMetadata = WavemakerMetadata()


class Sponge(Config):
    _title: str = "Sponge"
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


class Tide(Config):

    _title: str = "Tide"

    type: bool = False

    class TideMetadata(Metadata):
        type: Parameter = Parameter(name="type", funwave_name="TIDAL_BC_ABS")
        #: Parameter = Parameter(name="", funwave_name="")
        pass

    meta: TideMetadata = TideMetadata()


class Boundary(Config):
    _title: str = "Boundary Conditions"
    periodic: bool = False
    sponge: Sponge
    tide: Tide

    class BoundaryMetadata(Metadata):
        periodic: Parameter = Parameter(name="periodic y", funwave_name="PERIODIC")
        #: Parameter = Parameter(name="", funwave_name="")

    meta: BoundaryMetadata = BoundaryMetadata()
