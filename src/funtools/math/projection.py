from __future__ import annotations, nested_scopes

from collections.abc import Callable
from functools import partial

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
import pyproj
import json
from pathlib import Path


class _Projection(ABC):
    """Interface for projecting between coordinates"""

    @abstractmethod
    def to_target(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def to_source(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def bounds_to_target(
        self, x0: float, y0: float, x1: float, y1: float, minimum_n: int = 100, **kwargs
    ) -> tuple[float, float, float, float]:
        """Returns outer bounds of target coordinates given source coordinates"""

        proj = partial(self.to_target, **kwargs)
        return self._proj_bounds(proj, x0, y0, x1, y1, minimum_n)

    def bounds_to_source(
        self, u0: float, v0: float, u1: float, v1: float, minimum_n: int = 100, **kwargs
    ) -> tuple[float, float, float, float]:
        """Return outter bounds of source coordinates given target coordinates"""

        proj = partial(self.to_source, **kwargs)
        return self._proj_bounds(proj, u0, v0, u1, v1, minimum_n)

    def _proj_bounds(
        self, func: Callable, x0: float, y0: float, x1: float, y1: float, minimum_n: int
    ) -> tuple[float, float, float, float]:
        """Returns outer bounds of projected coordinates given orginal coordinates"""

        xl = x1 - x0
        yl = y1 - y0

        ds = min(xl, yl) / minimum_n

        nx = int(xl // ds)
        ny = int(yl // ds)

        x = np.linspace(x0, x1, nx, endpoint=True)
        y = np.linspace(y0, y1, ny, endpoint=True)

        xb = np.concatenate([[x.min()] * ny, [x.max()] * ny, x, x])
        yb = np.concatenate([y, y, [y.min()] * nx, [y.max()] * nx])

        (u0, u1), (v0, v1) = ((s.min(), s.max()) for s in func(xb, yb))
        return u0, v0, u1, v1


class GeoProjection(_Projection):
    """Project Geographical Lon/Lat coordinates to some ESPG projections"""

    def __init__(self, target_espg: int) -> None:
        crs = pyproj.CRS.from_epsg(target_espg)
        self._proj = pyproj.Proj(crs)

    def to_source(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert projection coordinates to Geographical coordinates"""
        return self._proj(u, v, inverse=True)

    def to_target(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Geographical coordinates to projection coordinates"""
        return self._proj(x, y)


class MercatorProjection(_Projection):
    """Projects Geographical Lon/Lat coordinates to Mercator coordinates for Bokeh plotting"""

    def to_source(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts Geographical coodinates to Mercator coordinates"""
        x = u * 20037508.34 / 180
        y = np.log(np.tan((90 + v) * np.pi / 360)) / (np.pi / 180)
        y = y * 20037508.34 / 180
        return x, y

    def to_target(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts Mercator coodinates to Geographical coordinates"""
        u = x * (180.0 / 20037508.34)
        y = y / (20037508.34 / 180.0)
        v = np.arctan(np.exp(y * (np.pi / 180))) * 360 / np.pi - 90
        return u, v


class RotationProjection(_Projection):
    """Project from one coordinate system to another by rotation around some point of rotation and shifting coordinates"""

    def __init__(
        self,
        rotation_x0: float,
        rotation_y0: float,
        angle: float,
        offset_x: float,
        offset_y: float,
    ) -> None:

        self._rot_x0 = rotation_x0
        self._rot_y0 = rotation_y0
        self._angle = angle
        self._off_x = offset_x
        self._off_y = offset_y

    def to_source(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts rotated coordinates to original coordinates"""
        u = u + self._off_x
        v = v + self._off_y

        angle = self._angle
        x = u * np.cos(angle) + v * np.sin(angle)
        y = -u * np.sin(angle) + v * np.cos(angle)

        x = x + self._rot_x0
        y = y + self._rot_y0

        return x, y

    def to_target(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts original coordinates to rotated coordinates"""
        x = x - self._rot_x0
        y = y - self._rot_y0

        # NOTE: Negative sign
        angle = -self._angle
        u = x * np.cos(angle) + y * np.sin(angle)
        v = -x * np.sin(angle) + y * np.cos(angle)

        u = u - self._off_x
        v = v - self._off_y

        return u, v

    @classmethod
    def from_funwave_info(cls, fpath: str) -> RotationProjection:
        """Wrapper method for creating class from FUNWAVE JSON file"""
        with open(fpath, "r") as fh:
            trans_info = json.load(fh)

        off_x = trans_info["x_off"] - trans_info["x0_extend"]
        off_y = trans_info["y_off"] - float(trans_info["yl_blend"]) / 2

        rot_x0 = trans_info["x0_rot"]
        rot_y0 = trans_info["y0_rot"]

        angle = np.deg2rad(trans_info["angle"])

        return RotationProjection(rot_x0, rot_y0, angle, off_x, off_y)


class LinkedProjections(_Projection):
    """Class for chaining projections"""

    def __init__(self, source_name: str, projections: dict) -> None:

        self._projections = list(projections.values())
        keys = list(projections.keys())
        keys.insert(0, source_name)

        self._i = {k: i for i, k in enumerate(keys)}
        self._n = len(projections)

    def to_source(
        self,
        u: np.ndarray,
        v: np.ndarray,
        source: str | None = None,
        target: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        "Convert from final target coordinates to first source coordinates in projection list" ""

        i0 = 0 if source is None else self._i[source]
        i1 = self._n if target is None else self._i[target]

        x, y = u, v
        for proj in reversed(self._projections[i0:i1]):

            x, y = proj.to_source(x, y)

        return x, y

    def to_target(
        self,
        x: np.ndarray,
        y: np.ndarray,
        source: str | None = None,
        target: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        "Convert from first src coordinates to final target coordinates in projection list" ""

        i0 = 0 if source is None else self._i[source] - 1
        i0 = max(i0 - 1, 0)
        i1 = self._n if target is None else self._i[target]

        u, v = x, y
        for proj in self._projections[i0:i1]:

            u, v = proj.to_target(u, v)

        return u, v

    @classmethod
    def create_funwave(cls, espg_code: int, fpath: str) -> LinkedProjections:

        source_name = "bokeh"
        projections = {
            "geo": MercatorProjection(),
            "proj": GeoProjection(espg_code),
            "fun": RotationProjection.from_funwave_info(fpath),
        }

        return LinkedProjections(source_name, projections)
