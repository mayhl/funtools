from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from numpy.typing import NDArray

from ..grid import types as gtypes
from ..math.projection2 import LinkedProjections, ProjectionsEnum

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


@dataclass
class Static(Generic[DataG, DataS]):
    grid: LinkedGrids[DataG] | DataG
    data: dict[str, DataS] | DataS
    projection: Optional[LinkedProjections] = None

    def __post_init__(self) -> None:

        data, self._data = Static._parse(self.data)
        self.data = LinkedGrids[dict[str, DataS]]()
        self.data.setOriginal(data)

    @classmethod
    def _parse(
        cls, data: Any
    ) -> tuple[dict[str, DataS], LinkedGrids[dict[str, DataS]]]:

        if not isinstance(data, dict):
            if not isinstance(data, list):
                data = [data]
            data = {str(i): d for i, d in enumerate(data)}

        return data, LinkedGrids[dict[str, DataS]]()

    @classmethod
    def create(
        cls,
        grid: LinkedGrids[DataG],
        data: Any,
        key: ProjectionsEnum | str,
        projection: LinkedProjections | None = None,
    ) -> Static:

        data, ldata = Static._parse(data)
        ldata.set(data, key)
        return Static(grid=grid, data=ldata, projection=projection)

    def get(
        self, pkey: ProjectionsEnum | str | None = None, dkey: str | None = None
    ) -> dict[str, DataS] | DataS:

        if self.projection is None:

            assert (not pkey is None) and (
                not dkey is None
            ), "No projection set, can not get by two keys."

            key = dkey if pkey is None else pkey

            assert isinstance(key, str)
            return self.data.getOriginal()[key]

        else:

            if dkey is None:
                assert not pkey is None
                return self.data.get(pkey)
            else:
                assert not pkey is None
                return self.data.getProjection(pkey)[dkey]

    def __getitem__(self, key: ProjectionsEnum | str) -> dict[str, DataS] | DataS:
        """Returns all transformed data as key/data pairs if projection is set,
        otherwise provides key access to all original data."""

        if self.projection is None:
            assert not isinstance(key, ProjectionsEnum)
            return self.data.getOriginal()[key]
        else:
            return self.data.getProjection(key)

    def applyProjection(
        self, source: ProjectionsEnum | str, target: ProjectionsEnum | str
    ) -> None:

        assert (
            not self.projection is None
        ), "No projections added, can not project any data."

        pass

@dataclass
class Timeseries(Generic[DataG, DataS]):
    grid: LinkedGrids[DataG]
    data: dict[str, DataS]
    time: NDArray
    projection: Optional[LinkedProjections] = None



@dataclass
class Reader2D:

    key: str
    def __getitem__(self, key) -> NDArray:









# class TimeseriesData(DataMixin[DataG], Generic[DataG,DataS]):
#     t: NDArray[Shape["* x"], float]
#     data: LinkedGrids[NDArray[Shape["* x"], DataS]]

Structured2D = Static[gtypes.Structured2D, NDArray]
# Structured2DTimeseries =TimeseriesData[ftypes.Structured2D, NDArray[Shape["* x"], float]]
