from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import shapely
from shapelysmooth import chaikin_smooth, taubin_smooth
from tqdm.notebook import tqdm

from funtools.backup import create_scatter_poly

from ..backup.buffer_fill_poly import create as buffer_fill_poly
from ..backup.create_equipartition_poly import \
    generate as create_equipartition_poly
from ..backup.create_scatter_poly import \
    generate as generate_poly_nearest_neighbor
from ..backup.polyfilter import filter_scatter_poly
# from funtools.parallel.multi import simple as eparallel
from ..parallel.simple import simple as eparallel
from . import grid
from .kdtree import KDTree


class Polygon:
    def __init__(self, poly: shapely.Polygon | shapely.MultiPolygon | tuple) -> None:

        if isinstance(poly, np.ndarray):
            poly = (poly[:,0], poly[:,1])

        if isinstance(poly, tuple):
            poly = shapely.Polygon(zip(*poly))

        if isinstance(poly, shapely.Polygon | shapely.MultiPolygon):
            if isinstance(poly, shapely.Polygon):
                poly = shapely.MultiPolygon([poly])

        self._poly = poly

    @classmethod
    def from_scatter(cls, data: np.ndarray, method: str, **opts) -> Polygon:

        match method:
            case "nearest":
                return cls.from_scatter_nearest_neighbor(data, **opts)

            case _:
                raise ValueError(f"Unknown method {method:s}")

    @classmethod
    def from_equipartition(
        cls, x: np.ndarray, y: np.ndarray, mask: np.ndarray, n_procs: int = 1
    ) -> Polygon:

        poly = create_equipartition_poly(x, y, mask, n_procs=n_procs)
        return Polygon(poly)

    @classmethod
    def from_scatter_nearest_neighbor(
        cls,
        data: np.ndarray,
        n_procs: int = 1,
    ) -> Polygon:
        poly = generate_poly_nearest_neighbor(KDTree(data), n_procs=n_procs)

        assert isinstance(poly, shapely.Polygon) or isinstance(
            poly, shapely.MultiPolygon
        )
        return Polygon(poly)

    def buffer_fill(self, **kwargs):

        self._poly = buffer_fill_poly(self._poly, **kwargs)

    @property
    def raw_polygon(self) -> shapely.MultiPolygon:
        """Returns raw shapely shapely.Polygon object"""
        return self._poly

    def to_hv_dict(self) -> list[dict]:
        """Returns a Holoviews compatible data format for shapely.Polygon plotting"""

        def _poly2dict(p: shapely.Polygon) -> dict:
            x, y = [list(s) for s in p.exterior.xy]
            data = {"x": x, "y": y}

            if len(p.interiors) > 0:
                data["holes"] = [[list(zip(*i.xy)) for i in p.interiors]]

            return data

        if isinstance(self._poly, shapely.Polygon):
            return [_poly2dict(self._poly)]
        else:
            return [_poly2dict(p) for p in self._poly.geoms]

    def to_file(self, fpath: str | Path) -> None:
        with open(fpath, "wb") as fh:
            pickle.dump(self._poly, fh, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, fpath: str | Path) -> Polygon:
        with open(fpath, "rb") as fh:
            return Polygon(pickle.load(fh))

    def intersects(self, other: Polygon) -> bool:
        return self._poly.intersects(other.raw_polygon)

    def smooth(self, tolerance: float | None = None, method: str = "chaikin", *args):
        """Returns a smooth polygon after applying a shapelysmooth algorithm and optional (recommened) simplifer, kwargs: tolerence"""
        smoothers = {
            "chaikin": chaikin_smooth,
            "taubin": taubin_smooth,
        }

        if not method in smoothers:
            raise ValueError(
                f"Invalid method '{method}'. Supported values:{smoothers.keys()}"
            )

        poly = shapely.MultiPolygon([smoothers[method](p) for p in self._poly.geoms])

        if not tolerance is None:
            poly = poly.simplify(tolerance=tolerance)

        return Polygon(poly)

    def apply_transform(self, projection):
        """Returns polygon after applying a transformation function: u, v = func(x,y)"""

        def _proj_line(l):
            x, y = [np.array(s) for s in l.xy]
            return list(zip(*[s.tolist() for s in projection(x, y)]))

        def _proj_poly(p: shapely.Polygon):
            exterior = _proj_line(p.exterior)
            interiors = [_proj_line(i) for i in p.interiors]

            return shapely.Polygon(exterior, holes=interiors)

        poly = shapely.MultiPolygon([_proj_poly(p) for p in self.raw_polygon.geoms])
        return Polygon(poly)

    def apply_crop(
        self, bounds: tuple[float, float, float, float], buffer_ratio: None | float = 0
    ):
        x0, y0, x1, y1 = bounds
        x = [x0, x1, x1, x0, x0]
        y = [y0, y0, y1, y1, y0]
        box = shapely.Polygon(zip(x, y))

        if not buffer_ratio is None:
            min_l = min([x1 - x0, y1 - y0])
            buffer = buffer_ratio * min_l
            box = box.buffer(buffer)

        return Polygon(box.intersection(self._poly))

    def equi_subdivide(
        self,
        data: np.ndarray,
        n_target_polys: int,
        filter_n_batches: int,
        padding_ratio: float = 0.05,
    ) -> list[Polygon]:
        """Subdivide polygon into target number of sub polygons"""

        def get_bounds(n, ds, s0, padding_ratio=0):
            s0 = np.arange(n) * ds + s0
            s1 = s0 + ds
            s0 -= padding_ratio * ds
            s1 += padding_ratio * ds
            return np.vstack([s0, s1]).T

        def generate_poly(bounds):
            (x0, x1), (y0, y1) = bounds
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            return shapely.Polygon(pts)

        (nx, ny), (dx, dy), (x0, y0) = grid.nearest_equal_ranges(
            self._poly.bounds, n_target_polys
        )
        # dx, dy, nx, ny, x0, y0 = grid_info
        xp_bounds = get_bounds(nx, dx, x0, padding_ratio=padding_ratio)
        yp_bounds = get_bounds(ny, dy, y0, padding_ratio=padding_ratio)

        pbounds = [(x, y) for x in xp_bounds for y in yp_bounds]
        subgrids = [generate_poly(b) for b in pbounds]

        # subpolys = [
        #    self._poly.intersection(g) for g in tqdm(subgrids, desc="Splitting Polygon")
        # ]

        args = [(self._poly, g) for g in subgrids]

        if len(args) == 0:
            return np.array([])
        subpolys = eparallel(
            shapely.intersection, args, None, desc="Splitting Polygon"
        )
        # return subpolys, subgrids
        filt = [p.area > 0 for p in subpolys]

        # NOTE: Doesn't work
        # is_full = [p.intersection(g).area == g.area for p, g in zip(subpolys, subgrids)]
        is_full = [False for p, g in zip(subpolys, subgrids)]
        x_bounds = get_bounds(nx, dx, x0)
        y_bounds = get_bounds(ny, dy, y0)
        bounds = [(x, y) for x in x_bounds for y in y_bounds]

        filt = np.array(filt).reshape(ny, nx)
        subpolys = np.array(subpolys).reshape(ny, nx)
        bounds = np.array(bounds).reshape(ny, nx, 4)
        is_full = np.array(is_full, dtype=bool).reshape(ny, nx)
        # return filt, subpolys, bounds

        # Computing slices/ranges for subdiving 2D grid
        # into targent number of squarish subdomains

        sizes = np.array(filt.shape)
        filt_n = grid.nearest_equal_subsizes(sizes, filter_n_batches)
        filt_y, filt_x = [grid.even_divide_slices(*a) for a in zip(sizes, filt_n)]

        # Separating 2D grids into prefiltering's subdomains
        filts = [(x, y) for x in filt_x for y in filt_y]
        subfilts = [filt[sy, sx] for sx, sy in filts]
        subpolys = [subpolys[sy, sx] for sx, sy in filts]
        bounds = [bounds[sy, sx] for sx, sy in filts]
        is_full = [is_full[sy, sx] for sx, sy in filts]

        # return subfilts, subpolys, bounds
        # Applying sub-filters on each subdomins
        subpolys = [x[f] for x, f in zip(subpolys, subfilts)]
        bounds = [x[f] for x, f in zip(bounds, subfilts)]
        is_full = [x[f] for x, f in zip(is_full, subfilts)]

        # Removing subdomains with no polygons/data
        filt = [len(p) > 0 for p in subpolys]
        subpolys = [x for x, f in zip(subpolys, filt) if f]
        bounds = [x for x, f in zip(bounds, filt) if f]
        is_full = [x for x, f in zip(is_full, filt) if f]

        def filter(data, indices, bounds):
            x0 = bounds[:, 0].min()
            x1 = bounds[:, 1].max()
            y0 = bounds[:, 2].min()
            y1 = bounds[:, 3].max()

            filt_x = (x0 < data[:, 0]) & (data[:, 0] <= x1)
            filt_y = (y0 < data[:, 1]) & (data[:, 1] <= y1)
            filt = filt_x & filt_y

            return data[filt, :], indices[filt]

        n, _ = data.shape
        indices = np.arange(n)

        x0, y0, x1, y1 = self._poly.bounds

        filt_x = (x0 < data[:, 0]) & (data[:, 0] <= x1)
        filt_y = (y0 < data[:, 1]) & (data[:, 1] <= y1)
        filt = filt_x & filt_y

        data = data[filt, :]
        indices = indices[filt]

        args = [(data, indices, b) for b in bounds]

        subdata = [filter(data, indices, b) for b in tqdm(bounds, desc="Prefiltering")]

        subpolys = [
            [(Polygon(p), f[i]) for i, p in enumerate(polys)]
            for polys, f in zip(subpolys, is_full)
        ]

        args = [(p, *d, f) for ps, d in zip(subpolys, subdata) for p, f in ps]

        funcs = [p.get_intersection_xyz for p, *_ in args]
        args = [a for _, *a in args]
        indices = eparallel(funcs, args, None, 16, desc="Filtering")

        indices = np.unique(np.concatenate(indices))

        return np.sort(indices)

    def get_intersection_xyz(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        is_full: bool,
        exclude: bool = False,
    ) -> np.ndarray:

        if is_full:
            return indices
        # u0, u1, v0, v1 = bounds
        x0, y0, x1, y1 = self._poly.bounds

        filt_x = (x0 <= data[:, 0]) & (data[:, 0] <= x1)
        filt_y = (y0 <= data[:, 1]) & (data[:, 1] <= y1)
        filt = filt_x & filt_y

        filt[filt] = [self._poly.contains(shapely.Point(*x)) for x in data[filt, :2]]

        return indices[filt]  # , :]
        # return filter_scatter_poly(
        #    data, self._poly, n_procs=n_procs, target_size=target_size, exclude=exclude
        # )


def equipartitioned_mask2shape(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    tolerance: float = 0.1,
    n_procs: int = 1,
):
    tolerence = 0.1
    target_length = 100

    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    n, m = mask.shape
    nbatch, mbatch = [round(s / target_length) for s in [n, m]]
    sys, sxs = [grid.even_divide_slices(*a) for a in [(n, nbatch), (m, mbatch)]]

    list_args = []
    for j, sy in enumerate(sys):
        for i, sx in enumerate(sxs):
            args = (i, j, x[sx], y[sy], mask[sy, sx], dx, dy, tolerance)
            list_args.append(args)

    rtn_val = eparallel(_wrapper, n_procs, list_args, p_desc="Merging")

    polys = np.empty([nbatch, mbatch], dtype=object)
    for i, j, data in rtn_val:
        polys[j, i] = data

    poly_rows = np.empty([nbatch], dtype=object)
    for j in range(nbatch):
        tmp = [p for p in polys[j, :] if not p is None]
        if len(tmp) == 0:
            poly_rows[j] = None
            continue

        p = tmp[0]
        for poly in tmp[1:]:
            p = p.union(poly)
        poly_rows[j] = p

    poly_rows = [p for p in poly_rows if not p is None]

    p = poly_rows[0]
    for poly in poly_rows[1:]:
        p = p.union(poly)
    return p


def _wrapper(i, j, x, y, data, dx, dy, tolerance, padding=4):
    return i, j, _mask2shape(x, y, data, dx, dy, tolerance, padding=padding)


def _mask2shape(x, y, data, dx, dy, tolerance, padding=4):
    n, m = data.shape
    min_dim = 2 * (padding + 1) + 1

    # if max(n, m) == 1:
    #    return _get_rect(x, y, dx, dy)

    if max(n, m) <= min_dim:
        if np.sum(data) == 0:
            return None
        xx, yy = np.meshgrid(x, y)
        idx = data.flatten()
        xx, yy = xx.flatten()[idx], yy.flatten()[idx]
        polys = [_get_rect(x0, y0, dx, dy) for x0, y0 in zip(xx, yy)]
        poly = polys[0]
        for p in polys[1:]:
            poly = poly.union(p)
        return poly.simplify(tolerance=tolerance)

    if n > m:
        n2 = n // 2
        x1, y1, data1 = x, y[:n2], data[:n2, :]
        x2, y2, data2 = x, y[n2:], data[n2:, :]
    else:
        m2 = m // 2
        x1, y1, data1 = x[:m2], y, data[:, :m2]
        x2, y2, data2 = x[m2:], y, data[:, m2:]

    kwargs = dict(padding=padding)

    def get_shape(x, y, data):
        p, a = data.size, np.sum(data)
        if a == 0:
            return None
        if a == p:
            return _get_rect(x, y, dx, dy)
        return _mask2shape(x, y, data, dx, dy, tolerance, **kwargs)

    poly1 = get_shape(x1, y1, data1)
    poly2 = get_shape(x2, y2, data2)

    is_poly1 = poly1 is not None
    is_poly2 = poly2 is not None

    if is_poly1 and is_poly2:
        return poly1.union(poly2).simplify(tolerance=tolerance)
    elif is_poly2:
        return poly2  # .simplify(tolerance=tolerance)
    elif is_poly1:
        return poly1  # .simplify(tolerance=tolerance)
    else:
        return None


def _get_rect(x, y, dx, dy):
    x0 = np.min(x) - dx
    x1 = np.max(x) + dx
    y0 = np.min(y) - dy
    y1 = np.max(y) + dy
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.flatten(), yy.flatten()
    poly = shapely.Polygon(((x0, y0), (x0, y1), (x1, y1), (x1, y0)))
    return poly


def _filter_box(data, indices, bounds):
    x0 = bounds[:, 0].min()
    x1 = bounds[:, 1].max()
    y0 = bounds[:, 2].min()
    y1 = bounds[:, 3].max()

    filt_x = (x0 < data[:, 0]) & (data[:, 0] <= x1)
    filt_y = (y0 < data[:, 1]) & (data[:, 1] <= y1)
    filt = filt_x & filt_y
    return data[filt, :], indices[filt]
