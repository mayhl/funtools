import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from funtools.core.data import Data
from funtools.core.projection import LinkedProjections

from ..core import data as dtypes
from ..core import types as gtypes
from ..io.funwave import InputConfig
from ..io.input import Input


class Core:

    def __init__(self, input: Core | Path | str | Input) -> None:

        if issubclass(input.__class__, Core):

            keys = ["input", "projection"]
            for k in keys:
                setattr(self, k, getattr(input, k))

            return

        if not isinstance(input, Input):
            input = Input.fromFile(input)

        self.input = input
        self.grid = None

        path = self.input.grid.path
        if path is None:
            self.projection = None
        else:
            kwargs = json.loads(path)
            self.projection = LinkedProjections(**kwargs)


        # stations - List[dict[int|str, int | List[int]]]
        # transects  List[dict[int|str, int | List[int]]]
        # arrays - List[]



    def areStations(self) -> bool:
        return self.input.stations.n > 0

    def isStationMap(self) -> bool: 
        return self.input.stations.map is None



class TimeseriesBase(Core):

    def __init__(self, input: Core | Path | str | Input) -> None:
        super().__init__(input)

        self.indices = []

    def read(self, key: str) -> dtypes.Data:
        pass

    def readVector(self, key: str) -> dtypes: Timeseries:
        pass

class CollectionMixin:

    def __init__(self) -> None:
        self.__items: list[TimeseriesBase] = []
        self.__key_map: dict[str, int] = {}

    def __getitem__(self, key: str | int) -> TimeseriesBase:
        if isinstance(key, str):
            key = self.__key_map[key]
        return self.__items[key]


class Stations:

    def __init__(self, input: Path | str | InputConfig, is_forced_2d=False) -> None:

        if not isinstance(input, InputConfig):
            input = InputConfig(input, is_forced_2d)

        config = input 

        self.data = config.stations.reader

    def read(self, keys: list[str] | str):

        self.d


class Transects:
    pass


class Simulation:



    def __init__(self, input: Path | str, is_forced_2d: bool = False) -> None:

        # self.core = Core(path)
        config = InputConfig(input, is_forced_2d)
        self.input = config

        self.fields = Fields(config)
        self.stations = Stations(config)
