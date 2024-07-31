import numpy as np

import shapely
from shapely.geometry import Polygon

from funtools.parallel import simple as eparallel
from funtools.subgrid import *# compute_equipartition, subdivide_by_ranges, linear2coord_index

def create_box(ds, x0, y0):
    h = ds
    pts = [(x0-h, y0-h),
           (x0-h, y0+h),
           (x0+h, y0+h),
           (x0+h, y0-h)]
    return Polygon(pts)    


def generate(kdtree, n_procs=1):   

    # Update to KDTree
    data = kdtree.data
    x, y = data[:,0], data[:,1]
    
    # Creating boxes centered at each grid point with lengths twice distance to near point 
    ds, _ = kdtree.query(data, k=2)
    ds = ds[:,1]

    n_batches = 100*n_procs
    ranges, subbatches, shape = compute_equipartition(data[:,0:2], n_batches, mode='batch')

    tmp_data = np.vstack([data.T, ds]).T
    subdata = subdivide_by_ranges(tmp_data, ranges)
    subdata, idxs = zip(*[(d, i) for i, d in enumerate(subdata) if d.size > 0])    

    list_args = [(d[:,:-1], d[:,-1]) for d in subdata]

    polys = eparallel(_create_polys, n_procs, list_args, p_desc='Creating')
    poly = shapely.unary_union(polys)
    return _simplify_shape(poly, ds)
    #nbx, nby = subbatches
    #idxs = [linear2coord_index(i, (nby, nbx)) for i in idxs]

    #n_batches = 4*n_procs
    
    #slices, polybatches = compute_equislices(np.zeros([nbx, nby]), n_batches, mode='batch')
   
    #print(slices)
    
    
    
    #polys = [create_box(*args) for args in zip(ds, x, y)]

    # Smoothing vertices
    n_smooth = 6
    buff = -0.9*np.mean(ds)
    r = 0.6
    for i in range(n_smooth):
        poly= poly.buffer(buff)
        buff= -r*buff

    return poly

def _simplify_shape(poly, ds):
    
    n_smooth = 8
    ds_min = np.min(ds)
    r = 0.8
    buff = r*np.mean(ds)
    for i in range(n_smooth):
        poly = poly.buffer(buff)
        poly = poly.simplify(0.5*ds_min)
        buff= -r*buff    

    return poly

def _create_polys(data, ds, n=8):
    x, y = data[:,0], data[:,1]
    polys = [_create_poly(*x, n) for x in zip(x, y, ds)]
    poly = shapely.unary_union(polys)
    return _simplify_shape(poly, ds)

def _create_poly(x0, y0, r, n=8):
    offset = np.pi/n - np.pi/2 
    angles = np.arange(0,n)*2*np.pi/n + offset
    x = r*np.cos(angles) + x0
    y = r*np.sin(angles) + y0
    
    return Polygon(zip(x,y))