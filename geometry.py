
from src.subgrid import *


from scipy.spatial import cKDTree as KDTree


import numpy as np
from src.parallel import simple as sparallel 




def distance_grid_to_curve(x, y, xc, yc, p=100, n_procs=1):
    
    
    kdtree = KDTree(np.vstack([xc, yc]).T)
    
    
    shp = (len(y), len(x))
    
    ds= np.zeros(shp)
    
    (sxs, sys), n_batches = compute_equislices(ds, p)
    slices = [(sy, sx) for sx in sxs for sy in sys]
    
    args = [(x[sx], y[sy], kdtree) for sy, sx in slices]
    rtn_vals = sparallel(_distance_grid_to_curve, n_procs, args, p_desc='Distances')
    
    idxs = np.zeros(shp).astype(int)
    for slc, (ds_loc, idxs_loc) in zip(slices, rtn_vals):
        ds[slc] = ds_loc
        idxs[slc] = idxs_loc
        #near_x[slc] = near_x_loc
        #near_y[slc] = near_y_loc
    
    
    return ds, idxs # near_x, near_y
  
def _distance_grid_to_curve(x, y, kdtree):
    shp = (len(y), len(x))
    pts = np.vstack([ss.flatten() for ss in np.meshgrid(x,y)]).T
    nearest_dist, nearest_ind = kdtree.query(pts, k=1)
    #nearest_x, nearest_y = zip(*(kdtree.data[nearest_ind]))
    return tuple(np.reshape(s, shp) for s in [nearest_dist, nearest_ind])
    
    
    
def normalize_vectors(u, v):
    
    shp = u.shape
    norms = np.sqrt(u**2 + v**2)
    u /= norms
    v /= norms
    
    vecs = np.zeros(shp, dtype=np.dtype('float,float'))
    
    for idx, x in np.ndenumerate(vecs): 
        vecs[idx] = (u[idx], v[idx])
    
    return vecs

def compute_curve_normals(x, y):
    
    def average(s):
        t = np.zeros(len(s)+1)
        t[1:-1] = (s[1:] + s[:-1])/2
        t[0]=t[1]
        t[-1] = t[-2]
        return t
    
    dx, dy = (np.diff(s) for s in [x,y])
    u, v = (average(a) for a in [-dy, dx])
    return normalize_vectors(u, v)


def _side_grid_to_curve(x, y, xc, yc, normc, idxs):

    xx, yy = np.meshgrid(x, y)
    shp = xx.shape
    
    near_x, near_y, near_norm = (np.array(a)[idxs.flatten()].reshape(shp) for a in [xc, yc, normc])
    vec2curve = normalize_vectors(xx - near_x, yy -  near_y)
    
    dotprod = lambda u0, v0, u1, v1: u0*u1 + v0*v1
    mask = np.ones(shp)
    
    for i in np.ndindex(*shp):
        cond = dotprod(*near_norm[i], *vec2curve[i]) > 0
        mask[i] = 1 if cond else -1
        
    return mask


def side_grid_to_curve(x, y, xc, yc, idxs, n_procs=1, size=100):
    
    shp = len(y), len(x)
    mask = np.zeros(shp)

    (sys, sxs), n_batches = compute_equislices(mask, size)
    slices = [(sy, sx) for sx in sxs for sy in sys]
    
    normc = compute_curve_normals(xc, yc)
    extra_arg = (xc, yc, normc)
    args = [(x[sx], y[sy], *extra_arg, idxs[sy,sx]) for sy ,sx in slices] 
    
    rtn_vals = sparallel(_side_grid_to_curve, n_procs, args, p_desc='Sign')
    
    for s, submask in zip(slices, rtn_vals): mask[s] = submask
    return mask
    
    
    #(np.array(si)[nearest_ind].reshape([n,m]) for si in [xsi, ysi])
    

def _compute_smoother(dist, mask, smooth_distance):
    smoother = dist.copy()/smooth_distance*mask
    smoother[smoother>0]=0
    smoother = -smoother
    smoother[smoother>1] = 1
    smoother = 1 - smoother 
    return smoother
    
def compute_smoother(dist, mask, smooth_distance, size=100, n_procs=1):
         
    shp = dist.shape
    (sys, sxs), n_batches = compute_equislices(dist, p=size)
    slices = [(sy, sx) for sx in sxs for sy in sys]                    

    args = [(dist[s], mask[s], smooth_distance) for s in slices]         
    rtn_vals = sparallel(_compute_smoother, n_procs, args, p_desc='Smoother')
                     
    smoother = np.zeros(shp)                 
    for s, subsmoother in zip(slices, rtn_vals): smoother[s] = subsmoother
    return smoother
                     
# #  Weights based on distance from boundary 
# smoother = ds.copy()/smooth_distance
# # Setting max weight to one
# smoother[smoother>1] = 1
# # Marking direction/side of boundary with +/-
# smoother *= mask
# # Scaling [-1,1]=[0,1]
# smoother = (smoother+1)/2



                         
# 
# nearest_dist, nearest_ind = kdtree.query(pts, k=1)

# args = (np.array(si)[nearest_ind].reshape([n,m]) for si in [xsi, ysi])
# nearest_xsi, nearest_ysi = args

# xx, yy, nearest_dist, nearest_ind = (s.reshape([n, m]) for s in [xx, yy, nearest_dist, nearest_ind])
