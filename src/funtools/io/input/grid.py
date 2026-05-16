from __future__ import annotations

from enum import Enum
from os import PathLike
from typing import Optional

from numpy.typing import NDArray
from pydantic import FiniteFloat, NonNegativeFloat, PositiveFloat, PositiveInt

from ...core import types as gtypes
from ...core.data import DataClass, LinkedGrids, Process
from ...core.projection import LinkedProjections, ProjectionsEnum
from ...math.grid import rectilinear2d
from .core import _IN_META_, _OUT_META_, Config, Metadata, Parameter, Variable


class BathyTypeEnum(Enum):
    FLAT = "FLAT"
    SLOPE = "SLOPE"
    DATA = "DATA"


class Bathymetry(Config):

    _title: str = "Bathymetry"
    input: BathyTypeEnum
    path: PathLike
    water_level: FiniteFloat = 0.0

    class BathymetryMetadata(Metadata):
        input: Parameter = Parameter(name="type", funwave_name="DEPTH_TYPE")

        path: Parameter = Parameter(name="file path", funwave_name="DEPTH_FILE")
        water_level: Parameter = Parameter(
            name="Water Level", funwave_name="WaterLevel"
        )

    meta: BathymetryMetadata = BathymetryMetadata()

    def isFile(self) -> bool:
        return self.input == BathyTypeEnum.DATA

    def getGenerated(self) -> NDArray:
        assert not self.__bathy is None
        return self.__bathy

    def initCompleted(self, **kwargs):

        if self.isFile():
            self.__bathy = None
            return

        grid: Grid = kwargs["Grid"]
        x, y = grid.data.nodes

        raise NotImplementedError()


class Grid(Config):
    _title: str = "Grid"
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

    def model_post_init(self, context: Any) -> None:

        self.__grid = None
        self.__proj = None

        self.__out_meta = {
            k: Variable(**_OUT_META_[k]) for k in ["x", "y", "sta_idx", "bathy"]
        }

    @property
    def data(self) -> gtypes.Structured2D | LinkedGrids[gtypes.Structured2D]:

        if self.__grid is None:
            x, y = rectilinear2d(self.nx, self.dx, self.ny, self.dy)

            xnodes, ynodes = rectilinear2d(
                self.nx, self.dx, self.ny, self.dy, mode="border"
            )
            x = gtypes.Structured1D(x, xnodes)
            y = gtypes.Structured1D(y, ynodes)

            grid = gtypes.Structured2D(x, y)

            if not self.__proj is None:
                grid = LinkedGrids[gtypes.Structured2D](
                    rotated=grid,
                    original=ProjectionsEnum.ROTATION,
                    projection=self.__proj,
                )

            self.__grid = grid

        return self.__grid

    def getMeta(self) -> dict[str, Variable]:
        return self.__out_meta


class Parallel(Config):

    _title: str = "Parallel"
    nx: PositiveInt
    ny: PositiveInt

    class ParallelMetadata(Metadata):
        nx: Parameter = Parameter(name="px", funwave_name="PX")
        ny: Parameter = Parameter(name="py", funwave_name="PY")

    meta: ParallelMetadata = ParallelMetadata()
