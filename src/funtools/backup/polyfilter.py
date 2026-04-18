import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

from .parallel import simple as sparallel


def _idx_filter_scatter(x, y, x0, y0, x1, y1):
    idx_x = (x0 <= x) & (x <= x1)
    idx_y = (y0 <= y) & (y <= y1)
    return idx_x & idx_y


def filter_scatter_bounds(data, poly):
    x = data[:, 0]
    y = data[:, 1]
    idx = _idx_filter_scatter(x, y, *poly.bounds)
    return idx
    return data[idx, :]


def get_balanced_slices(n, m, offset=0):
    sub_n = n // m
    p = n % m
    idxs = []
    i0 = offset

    for j in range(m):
        i1 = i0 + sub_n
        if j < p:
            i1 += 1
        idxs.append(slice(i0, i1))
        i0 = i1

    return idxs


from tqdm.notebook import tqdm


def _filter_scatter_poly(data, polys):

    n, _ = data.shape
    filt = np.zeros(n, dtype=bool)

    for poly in polys:

        subfilt = filter_scatter_bounds(data, poly)
        subdata = data[~filt, :2][subfilt, :]
        filt[subfilt] = [poly.contains(Point(x, y)) for x, y in tqdm(subdata)]

    return filt


def filter_scatter_poly(
    data, poly, n_procs=4, target_size=10000, exclude: bool = False
):

    if False:  # type(poly) is MultiPolygon:

        print("MultiPolygon")
        filt = filter_scatter_bounds(data, poly)
        gdata = data[filt, :]  # .copy()
        polys = list(poly.geoms)
    else:
        n, _ = data.shape
        filt = np.zeros(n, dtype=bool)
        polys = [poly]
        gdata = data  # .copy()

    n, _ = gdata.shape
    m = int(n / target_size)
    if m < n_procs:
        m = n_procs

    slices = get_balanced_slices(n, m)
    args_list = [gdata[s, :] for s in slices]
    # print(args_list[0].shape)

    rtn_vals = sparallel(
        _filter_scatter_poly, n_procs, args_list, common_args=polys, p_desc="Filtering"
    )

    rtn_vals = np.concatenate(rtn_vals).astype(bool)
    filt[~filt] = rtn_vals

    if exclude:
        return data[~filt, :]
    else:
        return data[filt, :]
