from __future__ import annotations, nested_scopes

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS


@dataclass
class Angle:

    angle: float | NDArray
    is_azimuth: bool = False
    is_radians: bool = True

    def __post_init__(self) -> None:

        angle = self.angle
        if self.is_azimuth:
            if self.is_radians:
                angle = np.pi - angle
            else:
                angle = 90 - angle

        if not self.is_radians:
            angle = np.deg2rad(angle)

        self.__angle = angle

    @property
    def radians(self) -> float | NDArray:
        return self.__angle

    @property
    def degrees(self) -> float | NDArray:
        return np.rad2deg(self.__angle)

    @property
    def azimuth(self) -> float | NDArray:
        return 90 - np.rad2deg(self.__angle)

    @property
    def azimuth_radians(self) -> float | NDArray:
        return np.pi - self.__angle


class ProjectionsEnum(Enum):
    GEOSPATIAL = "geospatial"
    MERCATOR = "mercator"
    ROTATION = "rotation"
    SOURCE = "source"
    TARGET = "target"

    @classmethod
    def _missing_(cls, value: object) -> Any:

        assert isinstance(value, str)
        for member in cls:
            if member[:3] == value[:3]:
                return member

        return ValueError()


@dataclass
class ProjectionBase(ABC):
    """Interface for projecting between coordinates"""

    @abstractmethod
    def toTarget(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def toSource(self, u: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
        pass

    def vectorToTarget(
        self, x: NDArray, y: NDArray, ux: NDArray, uy: NDArray
    ) -> tuple[NDArray, NDArray]:
        return ux, uy

    def vectorToSource(
        self, p: NDArray, q: NDArray, vp: NDArray, vq: NDArray
    ) -> tuple[NDArray, NDArray]:
        return vp, vq


@dataclass
class GeoProjection(ProjectionBase):
    """Project Geographical Lon/Lat coordinates to some ESPG projections"""

    crs: CRS
    inverse: Optional[bool] = False

    def __post_init__(self):
        self._proj = pyproj.Proj(self.crs)

    def toSource(self, u: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
        """Convert projection coordinates to Geographical coordinates"""
        return self._proj(u, v, inverse=not self.inverse)

    def toTarget(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """Convert Geographical coordinates to projection coordinates"""
        return self._proj(x, y, inverse=self.inverse)


@dataclass
class GeoTransform(ProjectionBase):
    """Project between two ESPG Projections"""

    source_crs: CRS
    target_crs: CRS

    def __post_init__(self):
        self._src_proj = GeoProjection(self.source_crs)
        self._trg_proj = GeoProjection(self.target_crs)

    def toSource(self, u: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
        """Connverts source projection coordinates to target projection coordinates"""
        x, y = self._src_proj.toSource(u, v)
        return self._trg_proj.toTarget(x, y)

    def toTarget(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        u, v = self._trg_proj.toSource(x, y)
        return self._src_proj.toTarget(u, v)


@dataclass
class MercatorProjection(ProjectionBase):
    """Projects Geographical Lon/Lat coordinates to Mercator coordinates for Bokeh plotting"""

    def toSource(self, u: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
        """Converts Geographical coodinates to Mercator coordinates"""
        x = u * 20037508.34 / 180
        y = np.log(np.tan((90 + v) * np.pi / 360)) / (np.pi / 180)
        y = y * 20037508.34 / 180
        return x, y

    def toTarget(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """Converts Mercator coodinates to Geographical coordinates"""
        u = x * (180.0 / 20037508.34)
        y = y / (20037508.34 / 180.0)
        v = np.arctan(np.exp(y * (np.pi / 180))) * 360 / np.pi - 90
        return u, v


@dataclass
class RotationProjection(ProjectionBase):
    """Project from one coordinate system to another by rotation around some point of rotation and shifting coordinates"""

    rotation_x: float
    rotation_y: float

    angle: Angle
    offset_x: float
    offset_y: float
    length_x: float
    length_y: float

    def extend(
        self, west: float = 0, east: float = 0, south: float = 0, north: float = 0
    ) -> None:

        self.offset_x += west
        self.offset_y += south

    def toSource(self, u: NDArray, v: NDArray) -> tuple[NDArray, NDArray]:
        """Converts rotated coordinates to original coordinates"""

        u = u + self.offset_x
        v = v + self.offset_y

        angle = self.angle.radians
        x = u * np.cos(angle) + v * np.sin(angle)
        y = -u * np.sin(angle) + v * np.cos(angle)

        x = x + self.rotation_x
        y = y + self.rotation_y

        return x, y

    def toTarget(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """Converts original coordinates to rotated coordinates"""
        x = x - self.rotation_x
        y = y - self.rotation_y

        # NOTE: Negative sign
        angle = -self.angle.radians
        u = x * np.cos(angle) + y * np.sin(angle)
        v = -x * np.sin(angle) + y * np.cos(angle)

        u = u - self.offset_x
        v = v - self.offset_y

        return u, v

    def vectorToSource(
        self, p: NDArray, q: NDArray, vp: NDArray, vq: NDArray
    ) -> tuple[NDArray, NDArray]:

        mag = np.hypot(vp, vq)
        angle = np.arctan(vq, vp) + self.angle.radians

        ux = mag * np.cos(angle)
        uy = mag * np.sin(angle)
        return ux, uy

    def vectorToTarget(
        self, x: NDArray, y: NDArray, ux: NDArray, uy: NDArray
    ) -> tuple[NDArray, NDArray]:

        mag = np.hypot(ux, uy)
        angle = np.arctan(uy, ux) - self.angle.radians

        vp = mag * np.cos(angle)
        vq = mag * np.sin(angle)
        return vp, vq


@dataclass
class ProjectionIterator(ProjectionBase):
    """Class for chaining projections"""

    items: list[ProjectionBase]
    inverse: bool = False

    def toSource(
        self,
        u: NDArray,
        v: NDArray,
    ) -> tuple[NDArray, NDArray]:
        "Convert from final target coordinates to first source coordinates in projection list" ""

        if self.inverse:
            return self._toTarget(u, v)
        else:
            return self._toSource(u, v)

    def _toSource(
        self,
        u: NDArray,
        v: NDArray,
    ) -> tuple[NDArray, NDArray]:
        "Convert from final target coordinates to first source coordinates in projection list" ""

        x, y = u, v
        for proj in reversed(self.items):
            x, y = proj.toSource(x, y)
        return x, y

    def toTarget(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[NDArray, NDArray]:
        "Convert from first src coordinates to final target coordinates in projection list" ""

        if self.inverse:
            return self._toSource(x, y)
        else:
            return self._toTarget(x, y)

    def _toTarget(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[NDArray, NDArray]:
        "Convert from first src coordinates to final target coordinates in projection list" ""

        u, v = x, y
        for proj in self.items:
            u, v = proj.toTarget(u, v)
        return u, v


@dataclass
class LinkedProjections:
    source: GeoProjection
    target: Optional[GeoProjection] = None
    src2trg: Optional[GeoTransform] = None
    rotation: Optional[RotationProjection] = None

    def __post_init__(self) -> None:
        """Setting up internal projection"""
        self.mercator = MercatorProjection

    @classmethod
    def create(
        cls, source: Any, target: Any, rotation: RotationProjection | None = None
    ) -> LinkedProjections:

        def _parse(crs: Any) -> tuple[ProjectionBase | None, str | CRS, bool]:
            is_geo = False
            try:
                is_geo = ProjectionsEnum(crs) == ProjectionsEnum.GEOSPATIAL

            except Exception as e:
                raise e

            if is_geo:
                return None, crs, is_geo

            proj = GeoProjection(crs)

            return proj, crs, is_geo

        source, src_crs, is_src_geo = _parse(source)

        if target is None:
            is_trg_geo = False
            trg_crs = None
        else:
            target, trg_crs, is_trg_geo = _parse(target)

        ## Special cases
        if is_src_geo and is_trg_geo:
            src2trg = None

        elif is_src_geo:
            src2trg = GeoProjection(trg_crs)

        elif is_trg_geo:
            src2trg = GeoProjection(src_crs, inverse=True)  # Need inverse

        elif src_crs.equals(trg_crs) or trg_crs is None:
            src2trg = None

        else:
            # NOTE: Fix variable name or Class for trg/src mismatch
            src2trg = GeoTransform(trg_crs, src_crs)

        return LinkedProjections(source, target, src2trg, rotation)

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
