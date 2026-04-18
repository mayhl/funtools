import math

import numpy as np
import shapely
from shapely.geometry import Polygon
from tqdm.notebook import tqdm

from funtools.backup.parallel import simple as eparallel
from funtools.backup.subgrid import *  # compute_equipartition, subdivide_by_ranges, linear2coord_index

from ..math.grid import linspace


def create_box(ds, x0, y0):
    h = ds
    pts = [(x0 - h, y0 - h), (x0 - h, y0 + h), (x0 + h, y0 + h), (x0 + h, y0 - h)]
    return Polygon(pts)


def _get_subranges(s0, s1, n, pad_ratio=0.0):

    ranges = np.zeros((n,2))
    ds = (s1-s0)/n

    ranges[:,0] = (np.arange(n)-pad_ratio)*ds + s0
    ranges[:,1] = (np.arange(n)+1+pad_ratio)*ds + s0

    return ranges


def subdivde(data, n_batches, pad_ratio = 0.0):

    x0, x1 = data[:,0].min(), data[:,0].max()
    y0, y1 = data[:,1].min(), data[:,1].max()

    xl = x1 - x0
    yl = y1 - y0

    assert xl > 0
    assert yl > 0

    r = xl/yl

    ny = math.sqrt(n_batches/r)
    nx = round(r*ny)
    ny = round(ny)

    #nx = ny = 3
    x_ranges = _get_subranges(x0, x1, nx, pad_ratio=pad_ratio)
    y_ranges = _get_subranges(y0, y1, ny, pad_ratio=pad_ratio)


    def _filter(x0, x1, y0, y1):
        filt_x = (x0 <= data[:,0]) & (data[:,0] <= x1)
        filt_y = (y0 <= data[:,1]) & (data[:,1] <= y1)
        return data[filt_x & filt_y, :]


    ranges =  [(x,y) for x in x_ranges for y in y_ranges]
    data =  [_filter(*x,*y) for x, y in tqdm(ranges, desc='Seperating')]


    return [d for d in data if d.size > 0]




def generate(kdtree, n_procs=1):

    # Update to KDTree

    ds, _ = kdtree.query(kdtree.data, k=2)
    data = np.concatenate([kdtree.data, ds[:,1:2]], axis=1)

    # Creating boxes centered at each grid point with lengths twice distance to near point

    n_batches = 5 * 200 * n_procs

    data = subdivde(data,  n_batches)

    #list_args = [(d[:, :-1], d[:, -1]) for d in data]

    polys = eparallel(_create_polys, n_procs, data, p_desc="Creating")

    factor = 1 * n_procs

    while len(polys) > factor * n_procs:
        polys = np.array(polys)
        slices, n_batches = compute_equislices(polys, len(polys) // factor)
        list_args = [(polys[s], ) for s in slices[0]]
        polys = eparallel(_union, n_procs, list_args, p_desc="Merging")


    poly = _union(polys,)

    # Smoothing vertices
    min_ds = 1*np.min(ds)
    poly.simplify(min_ds)
    n_smooth = 6
    buff = -0.9 * np.mean(ds)
    r = 0.6
    for _ in tqdm(range(n_smooth), desc="Smoothing"):
        poly = poly.buffer(buff)
        poly.simplify(min_ds)
        buff = -r * buff

    return poly


def _union(polys):

    return shapely.unary_union(polys)
    #return _simplify_shape(poly, ds)


def _simplify_shape(poly, ds):

    n_smooth = 8
    ds_min = np.min(ds)
    r = 0.8
    buff = r * np.mean(ds)
    for i in range(n_smooth):
        poly = poly.buffer(buff)
        poly = poly.simplify(0.5 * ds_min)
        buff = -r * buff

    return poly


def _create_polys(data, n=8):
    #x, y = data[:, 0], data[:, 1]
    #polys = [_create_poly(*x, n) for x in zip(x, y, ds)]
    polys = [_create_poly(x[0], x[1], x[2], n) for x in data]
    poly = shapely.unary_union(polys)
    return poly
    return _simplify_shape(poly, ds)


def _create_poly(x0, y0, r, n=8):
    offset = np.pi / n - np.pi / 2
    angles = np.arange(0, n) * 2 * np.pi / n + offset
    x = r * np.cos(angles) + x0
    y = r * np.sin(angles) + y0

    return Polygon(zip(x, y))
