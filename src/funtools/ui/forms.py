from __future__ import annotations

import os
from pathlib import Path

import panel as pn

from ..model.simulation import LocalSimulation, ProjectedSimulation


class SimulationBrowser:

    def __init__(self, fpath: str) -> None:

        self._selector = pn.widgets.FileSelector(fpath)
        pass

    @classmethod
    def from_home(cls) -> SimulationBrowser:
        fpath = os.environ["HOME"]
        return cls(fpath)

    @classmethod
    def from_work(cls) -> SimulationBrowser:
        fpath = os.environ["$WORKDIR"]
        return cls(fpath)

    def show(self):
        return self._selector

    def load(self, index: int = 0) -> LocalSimulation:

        fpath = Path(self._selector.value[index])

        return LocalSimulation(fpath.parent)
