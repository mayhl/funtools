from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from numpy.typing import NDArray

from funtools.core import projection

from . import types as gtypes
from .projection import LinkedProjections, ProjectionsEnum, RotationProjection

DataS = TypeVar("DataStore")
DataG = TypeVar("DataGrid")


@dataclass
class LinkedGrids(Generic[DataS]):
    source: Optional[DataS] = None
    target: Optional[DataS] = None
    geospatial: Optional[DataS] = None
    mercator: Optional[DataS] = None
    rotated: Optional[DataS] = None
    original: ProjectionsEnum = ProjectionsEnum.SOURCE
    projection: Optional[LinkedProjections] = None

    def getOriginal(self) -> DataS:
        """Returns original data"""
        return getattr(self, self.original.name.lower())

    def setOriginal(self, data: DataS) -> None:
        """Sets original data"""
        setattr(self, self.original.name.lower(), data)

    def set(self, data: DataS, key: str | ProjectionsEnum) -> None:
        if isinstance(key, str):
            key = ProjectionsEnum(key)
        setattr(self, key.name.lower(), data)

    def get(self, key: str | ProjectionsEnum) -> DataS:
        """Returns data if not none, else throws assertion error."""

        if isinstance(key, str):
            key = ProjectionsEnum[key]

        item = getattr(self, key.name.lower())
        assert (
            not item is None
        ), "Attempting to access None data. Data needs to be transformed first."
        return item

    def getProjection(self, pkey: ProjectionsEnum | str) -> DataS:

        if isinstance(pkey, str):
            pkey = ProjectionsEnum[pkey]
        assert isinstance(pkey, ProjectionsEnum)

        match pkey:
            case ProjectionsEnum.GEOSPATIAL:
                data = self.geospatial

            case ProjectionsEnum.MERCATOR:
                data = self.mercator

            case ProjectionsEnum.ROTATION:
                data = self.rotated

            case ProjectionsEnum.SOURCE:
                data = self.source

            case ProjectionsEnum.TARGET:
                data = self.target

            case _:
                assert False, f"Unknown Enum {pkey}."

        assert (
            not data is None
        ), f"{pkey.value.capitalize()} data has not been initialized yet."

        return data

    def getProjection(
        self, source: str | ProjectionsEnum, target: str | ProjectionsEnum
    ) -> ProjectionIterator | None:
        """ "Returns LinkedProjections object based on keys"""
        assert not source == target

        source = ProjectionsEnum(source)
        target = ProjectionsEnum(target)

        PE = ProjectionsEnum

        __KEYS__ = [PE.MERCATOR, PE.GEOSPATIAL, PE.SOURCE, PE.TARGET, PE.ROTATION]
        __MAP__ = {k: i for i, k in enumerate(__KEYS__)}

        src_idx = __MAP__[source]
        trg_idx = __MAP__[target]

        is_inverse = src_idx > trg_idx

        if not is_inverse:
            source, target = target, source
            src_idx, trg_idx = trg_idx, src_idx

        args = (target, source)

        if target == ProjectionsEnum.ROTATION and self.rotation is None:
            raise ValueError("Rotation projection not initalized.")

        if target == ProjectionsEnum.MERCATOR:
            assert not self.rotation is None
        if source == ProjectionsEnum.MERCATOR:
            assert not self.rotation is None

        match args:
            case (PE.MERCATOR, PE.GEOSPATIAL):
                projs = [self.mercator]

            case (PE.MERCATOR, PE.SOURCE):
                projs = [self.mercator, self.source]

            case (PE.MERCATOR, PE.TARGET):
                projs = [self.mercator, self.target]

            case (PE.MERCATOR, PE.ROTATION):
                projs = [self.mercator, self.target, self.rotation]

            case (PE.GEOSPATIAL, PE.SOURCE):
                projs = [self.source]

            case (PE.GEOSPATIAL, PE.TARGET):
                projs = [self.target]

            case (PE.GEOSPATIAL, PE.ROTATION):
                projs = [self.target, self.rotation]

            case (PE.SOURCE, PE.TARGET):
                projs = [self.src2trg]

            case (PE.SOURCE, PE.ROTATION):
                projs = [self.src2trg, self.rotation]

            case (PE.TARGET, PE.ROTATION):
                projs = [self.rotation]

            case _:
                raise ValueError(f"Unsupported conversion from {source} to {target}")

        # Hacky edge cases removal
        projs = [p for p in projs if not p is None]

        return ProjectionIterator(projs, is_inverse)


@dataclass
class Vector:
    x: NDArray
    y: NDArray


@dataclass
class DataClass:
    grid: LinkedGrids[gtypes.Scatter] | gtypes.Scatter
    data: dict[str, NDArray | Vector] | NDArray | Vector
    time: NDArray
    key_map: dict[str, int]
    name: str


@dataclass
class Spectra(DataClass):
    freq: gtypes.Structured1D


@dataclass
class Spectra2D(Spectra):
    dire: gtypes.Structured1D


@dataclass
class Process(ABC):

    @abstractmethod
    def execute(self, **kwargs) -> DataClass | list[DataClass]:
        pass

    def getConfig(self) -> dict:
        pass
