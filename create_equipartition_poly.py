import rasterio
import numpy as np

from shapely.geometry import Polygon, Point
from funtools.parallel import simple as eparallel
from funtools.subgrid import compute_equislices as compute_equislices

def generate(kdtree, mask, n_procs=1):

    x = mask['x']
    y = mask['y']
    data = mask['data']
    
    tolerence = 0.1
    pbatch = n_procs*400

    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    (sys, sxs), (nbatch, mbatch) = compute_equislices(data, pbatch, mode='batch')
    
    list_args = []

    for j, sy in enumerate(sys):
        for i, sx in enumerate(sxs):   
            args = (i, j, x[sx], y[sy], data[sy, sx], dx, dy)
            list_args.append(args)

    rtn_val = eparallel(_wrapper, n_procs, list_args, p_desc='Merging')

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
        for poly in tmp[1:]: p = p.union(poly)
        poly_rows[j] = p
        
    poly_rows = [p for p in poly_rows if not p is None]

    p = poly_rows[0]
    for poly in poly_rows[1:]: p = p.union(poly)
    return p
    

def _wrapper(i, j, x, y, data, dx, dy): 
    return (i, j, _mask2shape(x, y, data, dx, dy))

def _mask2shape(x, y, data, dx, dy, padding = 4):

    n, m = data.shape
    min_dim = 2*(padding+1)+1
    
    tolerance = 0.1
    #if max(n, m) == 1:
    #    return _get_rect(x, y, dx, dy)

    if max(n,m) <= min_dim: 
        if np.sum(data) == 0: return None
        xx, yy = np.meshgrid(x,y)
        idx = data.flatten()
        xx, yy = xx.flatten()[idx], yy.flatten()[idx]
        polys = [_get_rect(x0, y0, dx, dy) for x0, y0 in zip(xx, yy)]
        poly = polys[0]
        for p in polys[1:]: poly = poly.union(p)
        return poly.simplify(tolerance=tolerance)
    
    if n > m:
        n2 = n//2
        x1, y1, data1 = x, y[:n2], data[:n2,:]
        x2, y2, data2 = x, y[n2:], data[n2:,:]
    else: 
        m2 = m//2
        x1, y1, data1 = x[:m2], y, data[:,:m2]
        x2, y2, data2 = x[m2:], y, data[:,m2:]

    kwargs = dict(padding   = padding  )  
    
    def get_shape(x, y, data):
        p, a = data.size, np.sum(data)
        if a == 0: return None
        if a == p: return _get_rect(x, y, dx , dy)
        return _mask2shape(x, y, data, dx, dy, **kwargs)
               
    poly1 = get_shape(x1, y1, data1)
    poly2 = get_shape(x2, y2, data2)

    is_poly1 = not poly1 is None
    is_poly2 = not poly2 is None
    
    if is_poly1 and is_poly2:
        return poly1.union(poly2).simplify(tolerance=tolerance)
    elif is_poly2:
        return poly2#.simplify(tolerance=tolerance)
    elif is_poly1:
        return poly1#.simplify(tolerance=tolerance)
    else:
        return None
    
def _get_rect(x, y, dx, dy):
    x0 = np.min(x)-dx
    x1 = np.max(x)+dx
    y0 = np.min(y)-dy
    y1 = np.max(y)+dy
    xx, yy = np.meshgrid(x,y)
    xx, yy = xx.flatten(), yy.flatten()
    poly = Polygon(((x0, y0),(x0, y1), (x1, y1), (x1, y0)))
    return poly
