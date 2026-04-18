from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Generic, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from . import filter
from .projection import ProjectionBase


@dataclass
class BaseType(ABC):
    @abstractmethod
    def project(self, proj: ProjectionBase) -> BaseType:
        """Return projection of spatial data"""

    @property
    def shape(self) -> tuple[int, ...]:
        assert False, "Abstract base class PointsBase"

    @property
    def ndim(self) -> int:
        assert False, "Abstract base class PointsBase"


@dataclass
class SimpleBase(BaseType):

    def __post_init__(self) -> None:
        self.setView()

    @property
    def points(self) -> tuple[NDArray, NDArray]:
        assert False, "Abstract base class PointsBase"

    @abstractmethod
    def setView(self, bounds: tuple[float | None, ...] | None = None) -> None:
        pass

    def stack(self) -> NDArray:
        """Returns all x and y grid points"""
        x, y = self.points
        return np.stack([x, y]).T

    def flatten(self) -> tuple[NDArray, NDArray]:
        """Return x and y mesh points (2D arrays) as flatten 1D arrays"""
        x, y = self.points
        return x.flatten(), y.flatten()


@dataclass
class Scatter(SimpleBase):

    x: NDArray
    y: NDArray

    def project(
        self,
        proj: ProjectionBase,
    ) -> Scatter:

        x, y = proj.toTarget(*self.points)

        kwargs = asdict(self)
        kwargs["x"] = x
        kwargs["y"] = y

        return Scatter(**kwargs)

    def setView(self, bounds: tuple[float | None, ...] | None = None) -> None:

        self.filter = filt = filter.maskScatter(self.x, self.y, bounds)
        if bounds is None:
            self.view = self
            return

        kwargs = asdict(self)
        kwargs["x"] = self.x[filt]
        kwargs["y"] = self.y[filt]

        self.view = self.__class__(**kwargs)

    @property
    def bounds(self) -> tuple[float, ...]:
        return self.x.min(), self.y.min(), self.x.max(), self.y.max()

    @property
    def points(self) -> tuple[NDArray, NDArray]:
        """Returns x and y mesh points"""
        return self.x, self.y

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.x),)

    @property
    def ndim(self) -> int:
        return 1


@dataclass
class Structured1D:

    nodes: NDArray
    bins: NDArray

    @property
    def ds(self) -> NDArray:
        return np.diff(self.nodes)

    @property
    def bounds(self) -> tuple[float, float]:
        return self.bins[0], self.bins[-1]

    @property
    def n(self) -> int:
        return self.nodes.size

    @property
    def nbins(self) -> int:
        return self.bins.size

    @property
    def bin_ranges(self) -> tuple[NDArray, NDArray]:
        return self.bins[1:], self.bins[:-1]

    def applyFilter(self, slices: slice | tuple[slice, slice]) -> Structured1D:

        def getBinSlice(ss, n):
            i = np.arange(n)[ss]
            st = ss.step

            ib = np.insert(i, -1, i[-1] + st) - st // 2

            if ib[0] < 0:
                ib[0] = 0

            if ib[-1] > n - 1:
                ib[-1] = n - 1

            return i, ib

        if isinstance(slices, tuple):
            sy, sx = slices
            ny, nx = self.nodes.shape

            i, ib = getBinSlice(sx, nx)
            j, jb = getBinSlice(sy, ny)

            ss = j, i
            sb = jb, ib
        else:
            ss = slices
            n = self.nodes.shape[0]
            ss, sb = getBinSlice(ss, n)

        return Structured1D(nodes=self.nodes[ss], bins=self.bins[sb])


@dataclass
class Structured2DMesh(SimpleBase):

    x: Structured1D
    y: Structured1D

    def project(
        self,
        proj: ProjectionBase,
    ) -> Structured2DMesh:

        x, y = proj.toTarget(*self.points)
        xb, yb = proj.toTarget(*self.bin_points)

        kwargs = asdict(self)
        kwargs["x"] = x
        kwargs["y"] = y
        kwargs["xbin"] = xb
        kwargs["ybin"] = yb

        return Structured2DMesh(**kwargs)

    def setView(
        self,
        bounds: tuple[float | None, ...] | None = None,
        stride: tuple[int, int] | int = 1,
    ) -> None:

        is_no_view = isinstance(stride, int) and stride == 1

        self.sx2d = self.sy2d = self.filter = filt = filter.maskMesh(
            self.x.nodes, self.y.nodes, bounds, stride
        )
        self.sy1d, self.sx1d = self.sx2d

        if is_no_view:
            self.view = self
        else:

            kwargs = asdict(self)

            if bounds is None:
                kwargs["x"] = self.x.applyFilter(self.sx2d)
                kwargs["y"] = self.x.applyFilter(self.sx2d)
                self.view = Structured2DMesh(**kwargs)
            else:
                kwargs["x"] = self.x.nodes[filt]
                kwargs["y"] = self.y.nodes[filt]
                del kwargs["xbin"]
                del kwargs["ybin"]
                self.view = Scatter(**kwargs)

    @property
    def bin_points(self) -> tuple[NDArray, NDArray]:
        return self.xbin, self.ybin

    @property
    def bounds(self) -> tuple[float, ...]:
        return self.xbin[0], self.ybin[0], self.xbin[-1], self.xbin[-1]

    @property
    def nodes(self) -> tuple[NDArray, NDArray]:
        """Returns x and y grid points, can either be 1D arrays (original) or 2D arrays (projected)."""
        return self.x.nodes, self.y.nodes

    @property
    def bins(self) -> tuple[NDArray, NDArray]:
        """Returns x and y grid points, can either be 1D arrays (original) or 2D arrays (projected)."""
        return self.x.bins, self.y.bins

    @property
    def shape(self) -> tuple[int, ...]:
        return self.x.nodes.shape

    @property
    def ndim(self) -> int:
        return 2


@dataclass
class Structured2D(Structured2DMesh):

    @property
    def shape(self) -> tuple[int, ...]:
        return self.y.nodes.size, self.y.nodes.size

    # Overloading to construct 2D mesh for original grid
    @property
    def points(self) -> tuple[NDArray, NDArray]:
        x, y = np.meshgrid(self.x.nodes, self.y.nodes)
        return x, y

    @property
    def bin_points(self) -> tuple[NDArray, NDArray]:
        x, y = np.meshgrid(self.x.bins, self.y.bins)
        return x, y

    def setView(
        self,
        bounds: tuple[float | None, ...] | None = None,
        stride: tuple[int, int] | int = 1,
    ) -> None:

        self.sx, self.sy = self.filter = filter.maskStructured(
            self.x.nodes, self.y.nodes, bounds, stride
        )

        if bounds is None:
            self.view = self
            return

        kwargs = asdict(self)
        kwargs["x"] = self.x[self.sx]
        kwargs["y"] = self.y[self.sy]

        self.view = Structured2D(**kwargs)


@dataclass
class Line(Scatter):

    s: Structured1D

    @property
    def dx(self) -> NDArray:
        return np.diff(self.x)

    @property
    def dy(self) -> NDArray:
        return np.diff(self.y)


@dataclass
class Lines(Scatter):

    sizes: NDArray
    items: list[Structured1D]

    def __post_init__(self) -> None:

        idxs1 = np.cumsum(self.sizes)
        idxs0 = np.insert(idxs1, 0, 0)
        self.slices = [slice(*a) for a in zip(idxs0, idxs1[:-1])]


@dataclass
class Point(BaseType):

    x: float
    y: float

    @property
    def ndim(self) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def project(self, proj: ProjectionBase, **kwargs) -> Point:
        x, y = [np.array([s]) for s in [self.x, self.y]]
        x, y = [s[0] for s in proj.toTarget(x, y)]
        return Point(x, y)


@dataclass
class Rectangle(BaseType):

    sw: Point
    se: Point
    ne: Point
    nw: Point

    @property
    def vertices(self) -> tuple[NDArray, NDArray]:
        """Return vertices starting with lower left and going counter-clockwise."""
        x = np.array([self.sw.x, self.se.x, self.ne.x, self.nw.x])
        y = np.array([self.sw.y, self.se.y, self.ne.y, self.nw.y])
        return x, y

    @property
    def line(self) -> tuple[NDArray, NDArray]:
        """Return boundary line starting with lower left and going counter-clockwise."""
        x, y = self.vertices
        x = np.insert(x, -1, x[0])
        y = np.insert(y, -1, y[0])
        return x, y

    def project(
        self,
        proj: ProjectionBase,
    ) -> Rectangle:

        x, y = proj.toTarget(*self.vertices)
        pts = [Point(x0, y0) for x0, y0 in zip(x, y)]
        return Rectangle(*pts)
