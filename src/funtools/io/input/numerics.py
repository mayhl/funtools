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

from .core import Config, Metadata, Parameter


class Friction(Config):

    _title: str = "Friction"
    is_file: bool = False
    cd: NonNegativeFloat = 0.0

    class FrictionMetadata(Metadata):
        is_file: Parameter = Parameter(name="is file", funwave_name="Friction_Matrix")
        cd: Parameter = Parameter(name="Cd", funwave_name="Cd")
        #: Parameter = Parameter(name="", funwave_name="")
        pass

    meta: FrictionMetadata = FrictionMetadata()


class TimeSchemeEnum(Enum):
    RK4 = "Runge_Kutta"


class Numerics(Config):

    _title: str = "Numerics"
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


class Physics(Config):

    _title: str = "Physics"
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
