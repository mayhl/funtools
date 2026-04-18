from __future__ import annotations, nested_scopes

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from operator import xor
from os import EX_CANTCREAT
from pathlib import Path
from re import A
from typing import Any, List, Optional, Tuple

import numpy as np
import pyproj
import rasterio
import shapely
from numpy.typing import NDArray
from pyproj import CRS

from .geometry import Polygon


class _Projection(ABC):
    """Interface for projecting between coordinates"""

    @abstractmethod
    def to_target(self, x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def to_source(self, u: NDArray, v: NDArray) -> Tuple[NDArray, NDArray]:
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
        """Return outer bounds of source coordinates given target coordinates"""

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

        # print(xb, yb)
        (u0, u1), (v0, v1) = ((s.min(), s.max()) for s in func(xb, yb))
        return u0, v0, u1, v1

    def poly_to_source(self, poly):
        return self._transform_shape(poly, self.to_source)

    def poly_to_target(self, poly):
        return self._transform_shape(poly, self.to_target)

    def _transform_shape(self, poly, transform):

        ptype = type(poly)
        # Hacking working around for lines
        if ptype in [shapely.LineString, shapely.MultiLineString]:
            return self._transform_line(poly, transform)
        polys = [poly] if ptype is shapely.Polygon else poly.geoms
        polys = [self._transform_poly(p, transform) for p in polys]

        return shapely.union_all(polys)

    def _transform_line(self, line, transform):
        lines = [line] if isinstance(line, shapely.LineString) else line.geoms
        NewLine = lambda l: shapely.LineString(list(zip(*transform(*l.xy))))
        return shapely.union_all([NewLine(l) for l in lines])

    def _transform_poly(self, poly, transform):

        holes = [list(zip(*transform(*s.xy))) for s in poly.interiors]
        pts = list(zip(*transform(*poly.exterior.xy)))

        return shapely.Polygon(pts, holes=holes)

    #    @abstractmethod
    def to_dict(self) -> dict:
        """Return dict of init arguments"""


class GeoProjection(_Projection):
    """Project Geographical Lon/Lat coordinates to some ESPG projections"""

    def __init__(self, target_crs: Any, is_inverse: bool = False) -> None:

        self._crs = crs = target_crs

        if not isinstance(crs, CRS):
            crs = CRS.from_user_input(crs)
        self._proj = pyproj.Proj(crs)
        self._is_inverse = is_inverse

    def to_source(self, u: NDArray, v: NDArray) -> Tuple[NDArray, NDArray]:
        """Convert projection coordinates to Geographical coordinates"""
        return self._proj(u, v, inverse=~self._is_inverse)

    def to_target(self, x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
        """Convert Geographical coordinates to projection coordinates"""
        return self._proj(x, y, inverse=self._is_inverse)

    def to_dict(self) -> dict:
        return {"crs": self._crs}


class GeoTransform(_Projection):
    """Project between two ESPG Projections"""

    def __init__(self, source_crs: CRS | dict, target_crs: CRS | dict) -> None:

        self._src_crs = source_crs
        self._trg_crs = target_crs
        self._src_proj = GeoProjection(source_crs)
        self._trg_proj = GeoProjection(target_crs)

    def to_source(self, u: NDArray, v: NDArray) -> Tuple[NDArray, NDArray]:
        """Connverts source projection coordinates to target projection coordinates"""
        x, y = self._src_proj.to_source(u, v)
        return self._trg_proj.to_target(x, y)

    def to_target(self, x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
        u, v = self._trg_proj.to_source(x, y)
        return self._src_proj.to_target(u, v)

    def to_dict(self) -> dict:
        return {
            "source_crs": self._src_crs,
            "target_crs": self._trg_crs,
        }


# NOTE:: Switch to ESPG: 3857
class MercatorProjection(_Projection):
    """Projects Geographical Lon/Lat coordinates to Mercator coordinates for Bokeh plotting"""

    def to_source(self, u: NDArray, v: NDArray) -> Tuple[NDArray, NDArray]:
        """Converts Geographical coodinates to Mercator coordinates"""
        x = u * 20037508.34 / 180
        y = np.log(np.tan((90 + v) * np.pi / 360)) / (np.pi / 180)
        y = y * 20037508.34 / 180
        return x, y

    def to_target(self, x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
        """Converts Mercator coodinates to Geographical coordinates"""
        u = x * (180.0 / 20037508.34)
        y = y / (20037508.34 / 180.0)
        v = np.arctan(np.exp(y * (np.pi / 180))) * 360 / np.pi - 90
        return u, v

    def to_dict(self) -> dict:
        return {}


# NOTE:: Switch to ESPG: 3857
class MercatorProjectionV1(_Projection):
    """Projects Geographical Lon/Lat coordinates to Mercator coordinates for Bokeh plotting"""

    def __init__(self) -> None:
        # ESPG :3857 - WGS 84/ Pseudo-Meractor
        crs = CRS.from_epsg(3857)
        self._proj = pyproj.Proj(crs)

    def to_source(self, u: NDArray, v: NDArray) -> Tuple[NDArray, NDArray]:
        """Converts Geographical coodinates to Mercator coordinates"""
        return self._proj(u, v, inverse=~True)

    def to_target(self, x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
        """Converts Mercator coodinates to Geographical coordinates"""
        return self._proj(x, y, inverse=~False)

    def to_dict(self) -> dict:
        return {}


class RotationProjection(_Projection):
    """Project from one coordinate system to another by rotation around some point of rotation and shifting coordinates"""

    def __init__(
        self,
        rotation_x: float,
        rotation_y: float,
        angle: float,
        offset_x: float,
        offset_y: float,
        length_x: float,
        length_y: float,
    ) -> None:

        self._rotation_x = rotation_x
        self._rotation_y = rotation_y
        self._angle = angle
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._length_x = length_x
        self._length_y = length_y

    def to_dict(self) -> dict:

        keys = [
            "rotation_x",
            "rotation_y",
            "angle",
            "offset_x",
            "offset_y",
            "length_x",
            "length_y",
        ]

        return {k: getattr(self, f"_{k}") for k in keys}

    @classmethod
    def from_dict(cls, kwargs: dict) -> RotationProjection:
        return RotationProjection(**kwargs)

    def extend(
        self, west: float = 0, east: float = 0, south: float = 0, north: float = 0
    ) -> None:

        self._offset_x += west
        self._offset_y += south

    def to_source(self, u: NDArray, v: NDArray) -> Tuple[NDArray, NDArray]:
        """Converts rotated coordinates to original coordinates"""

        u = u + self._offset_x
        v = v + self._offset_y

        angle = self._angle
        x = u * np.cos(angle) + v * np.sin(angle)
        y = -u * np.sin(angle) + v * np.cos(angle)

        x = x + self._rotation_x
        y = y + self._rotation_y

        return x, y

    def to_target(self, x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
        """Converts original coordinates to rotated coordinates"""
        x = x - self._rotation_x
        y = y - self._rotation_y

        # NOTE: Negative sign
        angle = -self._angle
        u = x * np.cos(angle) + y * np.sin(angle)
        v = -x * np.sin(angle) + y * np.cos(angle)

        u = u - self._offset_x
        v = v - self._offset_y

        return u, v


class LinkedProjectionsV1(_Projection):
    """Class for chaining projections"""

    def __init__(self, source_name: str, projections: dict) -> None:

        self._projections = list(projections.values())
        keys = list(projections.keys())
        keys.insert(0, source_name)

        self._i = {k: i for i, k in enumerate(keys)}
        self._n = len(projections)

    def to_source(
        self,
        u: NDArray,
        v: NDArray,
        source: str | None = None,
        target: str | None = None,
    ) -> Tuple[NDArray, NDArray]:
        "Convert from final target coordinates to first source coordinates in projection list" ""

        i0 = 0 if source is None else self._i[source]
        i1 = self._n if target is None else self._i[target]

        x, y = u, v
        for proj in reversed(self._projections[i0:i1]):

            x, y = proj.to_source(x, y)

        return x, y

    def to_target(
        self,
        x: NDArray,
        y: NDArray,
        source: str | None = None,
        target: str | None = None,
    ) -> Tuple[NDArray, NDArray]:
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


class ProjectionIterator(_Projection):
    """Class for chaining projections"""

    def __init__(self, projections: list[_Projection], is_inverse=False) -> None:

        self._projections = projections
        self._is_inverse = is_inverse

    def to_source(
        self,
        u: NDArray,
        v: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        "Convert from final target coordinates to first source coordinates in projection list" ""

        if self._is_inverse:
            return self._to_target(u, v)
        else:
            return self._to_source(u, v)

    def _to_source(
        self,
        u: NDArray,
        v: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        "Convert from final target coordinates to first source coordinates in projection list" ""

        x, y = u, v
        for proj in reversed(self._projections):
            x, y = proj.to_source(x, y)
        return x, y

    def to_target(
        self,
        x: NDArray,
        y: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        "Convert from first src coordinates to final target coordinates in projection list" ""

        if self._is_inverse:
            return self._to_source(x, y)
        else:
            return self._to_target(x, y)

    def _to_target(
        self,
        x: NDArray,
        y: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        "Convert from first src coordinates to final target coordinates in projection list" ""

        u, v = x, y
        for proj in self._projections:
            u, v = proj.to_target(u, v)
        return u, v


class LinkedProjectionsV2(_Projection):
    """Class for chaining projections"""

    def __init__(self, projections: list[_Projection], is_inverse=False) -> None:

        self._projections = projections
        self._is_inverse = is_inverse

    def to_source(
        self,
        u: NDArray,
        v: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        "Convert from final target coordinates to first source coordinates in projection list" ""

        if self._is_inverse:
            return self._to_target(u, v)
        else:
            return self._to_source(u, v)

    def _to_source(
        self,
        u: NDArray,
        v: NDArray,
    ) -> Tuple[NDArray, NDArray]:

        x, y = u, v
        for proj in reversed(self._projections):
            x, y = proj.to_source(x, y)

        return x, y

    def to_target(
        self,
        x: NDArray,
        y: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        "Convert from first src coordinates to final target coordinates in projection list" ""

        if self._is_inverse:
            return self._to_source(x, y)
        else:
            return self._to_target(x, y)

    def _to_target(
        self,
        x: NDArray,
        y: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        "Convert from first src coordinates to final target coordinates in projection list" ""

        u, v = x, y
        for proj in self._projections:

            u, v = proj.to_target(u, v)

        return u, v

    @classmethod
    def create_funwave(cls, espg_code: int, fpath: str) -> LinkedProjectionsV2:

        import json

        with open(fpath, "r") as fh:
            kwargs = json.load(fh)

        source_name = "bokeh"
        projections = {
            "geo": MercatorProjection(),
            "proj": GeoProjection(espg_code),
            "fun": RotationProjection.from_dict(kwargs),
        }

        # return LinkedProjections(source_name, projections.values())
        return LinkedProjectionsV2(projections.values())


#
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

        with open(fpath, "r") as fh:
            kwargs = json.load(fh)

        source_name = "bokeh"
        projections = {
            "geo": MercatorProjection(),
            "proj": GeoProjection(espg_code),
            "fun": RotationProjection.from_dict(kwargs),
        }

        return LinkedProjections(source_name, projections)


class LinkedProjectionsV3:
    """Base class for projecting spatial data between coordinate systems"""

    # src - source coordinates of input
    # trg - target coordinates for output
    # mrc - mercator coordinates  for satellite tile imagery plot
    # geo - geospatial coordinates for explict lat/lon

    __KEYS__ = ["mrc", "geo", "src", "trg", "rot"]

    __MAP__ = {k: i for i, k in enumerate(__KEYS__)}

    def __init__(
        self,
        src_crs: Any,
        trg_crs: Any,
    ) -> None:

        self._items = {k: None for k in self.__KEYS__}

        def _parse(crs) -> tuple[_Projection | None, str | CRS, bool]:
            if isinstance(crs, str):
                is_geo = crs == "geo"
            else:
                is_geo = False

            if is_geo:
                proj = None
            else:
                crs = CRS(crs)
                proj = GeoProjection(crs)

            return proj, crs, is_geo

        self._src_proj, src_crs, is_src_geo = _parse(src_crs)

        if trg_crs is None:
            self._trg_proj = None
            is_trg_geo = False
        else:
            self._trg_proj, trg_crs, is_trg_geo = _parse(trg_crs)

        self._src_crs = src_crs
        self._trg_crs = trg_crs

        ## Special cases
        if is_src_geo and is_trg_geo:
            self._src2trg_proj = None

        elif is_src_geo:
            self._src2trg_proj = GeoProjection(trg_crs)

        elif is_trg_geo:
            self._src2trg_proj = GeoProjection(src_crs, is_inverse=True)  # Need inverse

        elif src_crs.equals(trg_crs) or trg_crs is None:
            self._src2trg_proj = None

        else:
            # NOTE: Fix variable name or Class for trg/src mismatch
            self._src2trg_proj = GeoTransform(trg_crs, src_crs)

        self._mrc_proj = MercatorProjection()
        self._rot_proj = None

    def get_projection(self, source: str, target: str) -> ProjectionIterator | None:
        """ "Returns LinkedProjections object based on keys"""
        assert not source == target
        assert source in self.__KEYS__
        assert target in self.__KEYS__

        src_idx = self.__MAP__[source]
        trg_idx = self.__MAP__[target]

        is_inverse = src_idx > trg_idx

        if not is_inverse:
            source, target = target, source
            src_idx, trg_idx = trg_idx, src_idx

        args = (target, source)

        if target == "rot" and self._rot_proj is None:
            raise ValueError("Rotation projection not initalized.")

        if target == "mrc":
            assert not self._rot_proj is None
        if source == "mrc":
            assert not self._rot_proj is None

        match args:

            case ("mrc", "geo"):
                projs = [self._mrc_proj]

            case ("mrc", "src"):
                projs = [self._mrc_proj, self._src_proj]

            case ("mrc", "trg"):
                projs = [self._mrc_proj, self._trg_proj]

            case ("mrc", "rot"):
                projs = [self._mrc_proj, self._trg_proj, self._rot_proj]

            case ("geo", "src"):
                projs = [self._src_proj]

            case ("geo", "trg"):
                projs = [self._trg_proj]

            case ("geo", "rot"):
                projs = [self._trg_proj, self._rot_proj]

            case ("src", "trg"):
                projs = [self._src2trg_proj]

            case ("src", "rot"):
                projs = [self._src2trg_proj, self._rot_proj]

            case ("trg", "rot"):
                projs = [self._rot_proj]

            case _:
                raise ValueError(f"Unsupported conversion from {source} to {target}")

        # Hacky edge cases removal
        projs = [p for p in projs if not p is None]

        return ProjectionIterator(projs, is_inverse)

    @abstractmethod
    def project(self, source: str, target: str) -> None:
        """Project data between two coordinate systems"""
        pass


class _InputObject:
    """Base class for projecting spatial data between coordinate systems"""

    # src - source coordinates of input
    # trg - target coordinates for output
    # mrc - mercator coordinates  for satellite tile imagery plot
    # geo - geospatial coordinates for explict lat/lon

    __KEYS__ = ["mrc", "geo", "src", "trg", "rot"]

    __MAP__ = {k: i for i, k in enumerate(__KEYS__)}

    #    __KEYS__ = ["rot" "src", "trg", "mrc", "geo"]

    ## src -> trg (if trg is None, assume src is trg)

    ## mrc -> geo -> trg -> fun

    def __getitem__(self, key: str) -> Any:
        assert key in self.__KEYS__
        return self._items[key]

    def is_same_src_crs(self, other: _InputObject) -> bool:
        return self._src_crs.equals(other._src_crs)

    def is_same_trg_crs(self, other: _InputObject) -> bool:
        return self._trg_crs.equals(other._trg_crs)

    def __init__(
        self,
        src_crs: Any,
        trg_crs: Any,
    ) -> None:

        self._items = {k: None for k in self.__KEYS__}

        def _parse(crs) -> tuple[_Projection | None, str | CRS, bool]:
            if isinstance(crs, str):
                is_geo = crs == "geo"
            else:
                is_geo = False

            if is_geo:
                proj = None
            else:
                if crs is None:
                    proj = None
                else:
                    if not isinstance(crs, CRS):
                        crs = CRS.from_user_input(crs)
                    proj = GeoProjection(crs)

            return proj, crs, is_geo

        self._src_proj, src_crs, is_src_geo = _parse(src_crs)
        self._trg_proj, trg_crs, is_trg_geo = _parse(trg_crs)

        self._src_crs = src_crs
        self._trg_crs = trg_crs

        ## Special cases
        if is_src_geo and is_trg_geo:
            self._src2trg_proj = None

        elif is_src_geo:
            self._src2trg_proj = GeoProjection(trg_crs)

        elif is_trg_geo:
            self._src2trg_proj = GeoProjection(src_crs, is_inverse=True)  # Need inverse

        elif src_crs.equals(trg_crs) or trg_crs is None:
            self._src2trg_proj = None
            self._trg_proj = self._src_proj

        else:
            # NOTE: Fix variable name or Class for trg/src mismatch
            self._src2trg_proj = GeoTransform(trg_crs, src_crs)

        self._mrc_proj = MercatorProjection()
        self._rot_proj = None

    def _get_projection(self, source: str, target: str) -> LinkedProjectionsV2 | None:
        """ "Returns LinkedProjections object based on keys"""
        assert not source == target
        assert source in self.__KEYS__
        assert target in self.__KEYS__

        src_idx = self.__MAP__[source]
        trg_idx = self.__MAP__[target]

        is_inverse = src_idx > trg_idx

        if not is_inverse:
            source, target = target, source
            src_idx, trg_idx = trg_idx, src_idx

        args = (target, source)

        if target == "rot" and self._rot_proj is None:
            raise ValueError("Rotation projection not initalized.")

        match args:

            case ("mrc", "geo"):
                projs = [self._mrc_proj]

            case ("mrc", "src"):
                projs = [self._mrc_proj, self._src_proj]

            case ("mrc", "trg"):
                projs = [self._mrc_proj, self._trg_proj]

            case ("mrc", "rot"):
                projs = [self._mrc_proj, self._trg_proj, self._rot_proj]

            case ("geo", "src"):
                projs = [self._src_proj]

            case ("geo", "trg"):
                projs = [self._trg_proj]

            case ("geo", "rot"):
                projs = [self._trg_proj, self._rot_proj]

            case ("src", "trg"):
                projs = [self._src2trg_proj]

            case ("src", "rot"):
                projs = [self._src2trg_proj, self._rot_proj]

            case ("trg", "rot"):
                projs = [self._rot_proj]

            case _:
                raise ValueError(f"Unsupported conversion from {source} to {target}")

        # Hacky edge cases removal
        projs = [p for p in projs if not p is None]

        if len(projs) > 0:
            return LinkedProjectionsV2(projs, is_inverse)
        else:
            return None

    @abstractmethod
    def project(self, source: str, target: str) -> None:
        """Project data between two coordinate systems"""
        pass


class ScatterProjection(_InputObject):
    """Class for projecting scatter/xyz type data between coordinate systems"""

    def __init__(self, src_crs: Any, trg_crs: Any, data: NDArray) -> None:

        super().__init__(src_crs, trg_crs)

        assert isinstance(data, np.ndarray)
        assert data.ndim == 2

        d, n = data.shape
        assert d >= 2

        self._shp = (2, n)

        self._items["src"] = data

    def __getitem__(self, key: str) -> NDArray:
        return super().__getitem__(key)

    def apply_filter(self, indices: NDArray) -> None:
        for k, d in self._items.items():

            if not d is None:
                if len(indices) > 0:
                    self._items[k] = d[:, indices]
                else:
                    self._items[k] = np.zeros((2, 0))

        self._shp = (2, indices.size)

    def project(self, source: str, target: str) -> None:
        """Project scatter/xyz data between two coordinate systems"""
        proj = self._get_projection(source, target)

        if proj is None:
            self._items[target] = self._items[source]
            return

        assert isinstance(proj, LinkedProjectionsV2)

        od = self[source]
        nd = np.zeros(self._shp)

        nd[0, :], nd[1, :] = proj.to_target(od[0, :], od[1, :])

        # nd[2:, :] = od[2:, :]

        self._items[target] = nd

    def copy(self, data: NDArray) -> ScatterProjection:

        new_obj = ScatterProjection(self._src_crs, self._trg_crs, data)
        new_obj._rot_proj = self._rot_proj
        return new_obj


class BoxProjection(_InputObject):
    """Class for projecting a bounding box between coordinate systems"""

    def __init__(
        self,
        src_crs: Any,
        trg_crs: Any,
        #        rot_pts: tuple[float, float],
        #        widths: tuple[float, float, float, float],
    ) -> None:
        super().__init__(src_crs, trg_crs)

        self._shp = (2, 11)

    def set_crossshore(
        self, key: str, pt1: tuple[float, float], pt2: tuple[float, float]
    ) -> None:
        x1, y1 = pt1
        x2, y2 = pt2

        self._items[key] = np.zeros(self._shp)

        self._items[key][0, :2] = [x1, x2]
        self._items[key][1, :2] = [y1, y2]

    def get_crossshore_center(self, key) -> None:
        x = self._items[key][0, :]
        self._items[key][0, 2] = (x[1] + x[0]) / 2

        y = self._items[key][1, :]
        self._items[key][1, 2] = (y[1] + y[0]) / 2

    def centerline(self, key):
        return self._items[key].T[:2, :]

    def boundary(self, key: str):
        return self._items[key].T[3:7, :]

    def padded_boundary(self, key: str):
        return self._items[key].T[7:, :]

    def center(self, key):
        return self._items[key].T[2, :]

    def construct_box(
        self,
        key: str,
        north_width: float = 0,
        south_width: float = 0,
        east_width: float = 0,
        west_width: float = 0,
        interpolation_width: float = 0,
        angle_tolerance: float = 0.1,
        length_tolerance: float = 1,
    ) -> None:

        x = self._items[key][0, :]
        self._items[key][0, 2] = (x[1] + x[0]) / 2

        y = self._items[key][1, :]
        self._items[key][1, 2] = (y[1] + y[0]) / 2
        self._items[key][0, :2]

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        angle = np.rad2deg(np.arctan2(dy, dx))
        angle = round(angle / angle_tolerance) * angle_tolerance
        angle += -90

        self._angle_deg = angle
        self._angle = np.deg2rad(angle)

        l = np.hypot(dx, dy)

        l = round(l / length_tolerance) * length_tolerance

        x0 = -west_width
        x1 = east_width
        y0 = -l / 2 - south_width
        y1 = l / 2 + north_width

        x0 = 0
        x1 = west_width + east_width
        y0 = 0
        y1 = l + south_width + north_width

        x0p = x0 - interpolation_width
        x1p = x1 + interpolation_width
        y0p = y0 - interpolation_width
        y1p = y1 + interpolation_width

        pt_rot = x[2], y[2]

        dx, dy = x1 - x0, y1 - y0

        x_off = -west_width
        y_off = -l / 2 - south_width
        proj = RotationProjection(*pt_rot, -self._angle, x_off, y_off, dx, dy)

        xb = np.array([x0, x1, x1, x0, x0p, x1p, x1p, x0p])
        yb = np.array([y0, y0, y1, y1, y0p, y0p, y1p, y1p])

        xb, yb = proj.to_source(xb, yb)

        self[key][0, 3:] = xb
        self[key][1, 3:] = yb

        self._poly = shapely.Polygon(self[key][:, 3:7].T)
        self._poly_interp = shapely.Polygon(self[key][:, 7:].T)

        proj.extend(
            west=x0,
            south=y0,
        )
        self._rot_proj = proj

    @property
    def proj(self) -> RotationProjection:
        """Project bounding box between two coordinate systems"""
        return self._rot_proj

    @property
    def poly(self) -> shapely.Polygon:
        return self._poly

    @property
    def poly_interp(self) -> shapely.Polygon:
        return self._poly_interp

    def project(self, source: str, target: str) -> None:

        proj = self._get_projection(source, target)

        if proj is None:
            self._items[target] = self._items[source]
            return

        assert isinstance(proj, LinkedProjectionsV2)

        od = self[source]
        nd = np.zeros(self._shp)

        nd[0, :], nd[1, :] = proj.to_target(od[0, :], od[1, :])

        self._items[target] = nd


class PolygonProjection(_InputObject):
    """Class for projecting a shapely Polygon objects between coordinate systems"""

    def __init__(
        self,
        src_crs: Any,
        trg_crs: Any,
        poly: Polygon,
    ) -> None:
        super().__init__(src_crs, trg_crs)

        self._items["src"] = poly

    def __getitem__(self, key: str) -> Polygon:
        return super().__getitem__(key)

    def project(self, source: str, target: str) -> None:
        """Project polygon between two coordinate systems"""
        proj = self._get_projection(source, target)

        if proj is None:
            self._items[target] = self._items[source]
            return

        assert isinstance(proj, LinkedProjectionsV2)
        proj = proj.to_target

        poly = self._items[source]._poly
        ptype = type(poly)
        # Hacking working around for lines
        if ptype in [shapely.LineString, shapely.MultiLineString]:
            return self._transform_line(poly, proj)

        polys = [poly] if ptype is shapely.Polygon else poly.geoms
        polys = [self._transform_poly(p, proj) for p in polys]

        self._items[target] = Polygon(shapely.union_all(polys))

    def _transform_line(self, line, transform):
        """Wrapper method for project and shapely.LineString"""
        lines = [line] if isinstance(line, shapely.LineString) else line.geoms
        NewLine = lambda l: shapely.LineString(list(zip(*transform(*l.xy))))
        return shapely.union_all([NewLine(l) for l in lines])

    def _transform_poly(self, poly, transform):
        """Wrapper method for project and shapely.Polygon"""

        def _parse(x, y):
            return np.array(x), np.array(y)

        holes = [list(zip(*transform(*_parse(*s.xy)))) for s in poly.interiors]
        pts = list(zip(*transform(*_parse(*poly.exterior.xy))))

        return shapely.Polygon(pts, holes=holes)
