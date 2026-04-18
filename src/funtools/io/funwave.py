from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
from dacite import from_dict
from numpy.typing import NDArray

from ..core import types as gtypes
from ..core.data import DataClass, LinkedGrids, Process
from ..core.projection import LinkedProjections, ProjectionsEnum
from ..math.grid import rectilinear2d
from .input import Input

# def readTimeFile(path: str):
