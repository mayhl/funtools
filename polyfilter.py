import numpy as np
from src.parallel import simple as sparallel
from shapely.geometry import Polygon, MultiPolygon, Point


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


def _filter_scatter_poly(data, polys):
    sdata = []
    for poly in polys:

        
        filt = filter_scatter_bounds(data, poly)
        tdata = data[filt, :]
        
        x, y = tdata[:, 0], tdata[:, 1]

        idx = [poly.contains(Point(x, y)) for x, y in zip(x, y)]
        sdata.append(tdata[idx, :])

        n, d = tdata.shape
        n2, d = tdata[idx, :].shape
        # if n > 0 and not n2 == n:
        #     print(data.shape)
        #     print(tdata.shape)
        #     print(tdata[idx, :].shape)
        #     print('-----')

    return np.concatenate(sdata)


def filter_scatter_poly(data, poly, n_procs=4):
    if type(poly) is MultiPolygon:
        idx = filter_scatter_bounds(data, poly)
        gdata = data[idx, :].copy()
        polys = list(poly.geoms)
    else:
        polys = [poly]
        gdata = data.copy()

    target_size = 10000
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

    # sdata = []
    # for p in tqdm(cpolys, desc=k):
    #     idx = filter_scatter_poly(gdata, p)

    #     if len(idx) > 0:
    #         sdata.append(gdata[idx, :])
    #         gdata = gdata[~idx,:]

    return np.concatenate(rtn_vals)
