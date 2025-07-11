from __future__ import annotations
import rasterio
import numpy as np

from shapely.geometry import Polygon
from funtools.parallel.multi import simple as eparallel

from . import grid


import pickle
import numpy as np
import shapely

# from shapely import Polygon as shapely.Polygon
# from shapely import MultiPolygon as shapely.MultiPolygon
from shapelysmooth import chaikin_smooth, taubin_smooth
from pathlib import Path


class Polygon:
    def __init__(self, poly: shapely.Polygon | shapely.MultiPolygon) -> None:
        if isinstance(poly, shapely.Polygon | shapely.MultiPolygon):
            if isinstance(poly, shapely.Polygon):
                poly = shapely.MultiPolygon([poly])

        self._poly = poly

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

        return [_poly2dict(p) for p in self._poly.geoms]

    def to_file(self, fpath: str | Path) -> None:
        with open(fpath, "wb") as fh:
            pickle.dump(self._poly, fh, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, fpath: str | Path) -> Polygon:
        with open(fpath, "rb") as fh:
            return Polygon(pickle.load(fh))

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
