import shapely
from shapely import Polygon, Point
import numpy as np

from src.parallel import simple as sparallel 

def get_buff_simplify_poly(poly, buf, tol):
    return poly.buffer(buf).buffer(-buf).simplify(tolerance=tol)


def get_buff_fill_poly(poly, buffer):
    bpoly = poly.buffer(buffer).buffer(-buffer)
    diff = (bpoly.area - poly.area)/poly.area   
    return bpoly, diff  
 
def _binary_search(poly):

    x0, y0, x1, y1 = poly.bounds
    xl = x1 - x0
    yl = y1 - y0
    
    threshold = 0.005
    max_iterations = 5
    factor = 10 
    b_min = 1 
    b_max = 2.0*max(xl, yl)
    tolerence = 1.0
    n_refine = 20 

    for i in range(max_iterations):   
        _, diff = get_buff_fill_poly(poly, b_min)
        if diff < threshold: break
        b_min /= factor

    if b_min > b_max: b_max *= factor
        

    bpoly, diff = get_buff_fill_poly(poly, b_max)

    for i in range(n_refine):

        b_mid = np.exp((np.log(b_min) + np.log(b_max))/2)
        temp_poly, diff = get_buff_fill_poly(poly, b_mid)
 
        if diff > threshold:
            b_max = b_mid
            bpoly = temp_poly
        else:
            b_min = b_mid

    return bpoly.simplify(tolerence)


def _fill_holes(poly):
    
    max_iterations = 100
    p1 = _binary_search(poly)   

    for i in range(max_iterations):
        p2 = _binary_search(p1)
        diff = np.abs((p1.area-p2.area)/p1.area)
        p1 = p2
        if diff < 0.05: break

    if i + 1 == max_iterations: raise Exception()
    
    return p2


def _sub_intersections(rects, poly):
    sub_polys = [poly.intersection(rect) for rect in rects]
    return [s for s in sub_polys if s.area > 0]

def sub_divide_poly(poly, feature_size=200, pad_ratio = 0, n_procs=1):
    
    xl_loc, yl_loc,m, n = get_subdomain_info(*poly.bounds, feature_size)
    
    x0, y0, *_ = poly.bounds
    
    x = x0 + np.arange(m)*xl_loc
    y = y0 + np.arange(n)*yl_loc
    pts = [(x0, y0) for x0 in x for y0 in y]
    rects = [get_rectangle(x0, y0, xl_loc, yl_loc, pad_ratio) for x0, y0 in pts]

    if n_procs == 1:
        subshapes = sparallel(poly.intersection, 1, rects, is_p_bar=False)
        subshapes = [s for s in subshapes if s.area > 0]
    else: 
        n_batches =  n_procs
        slices = get_balanced_slices(len(rects), n_batches)
        args_list = [(rects[s], poly) for s in slices]
        rtn_vals = sparallel(_sub_intersections, n_procs, args_list, is_p_bar=False)
        subshapes = np.concatenate(rtn_vals)
        
    return subshapes

def is_point_in_poly(poly, data):
    x, y = data[0], data[1]
    return poly.contains(Point(x, y))

def filter_data_in_poly(poly, data, idxs = None):
    n, _ = data.shape
    if idxs is None: idxs = np.arange(n)
    filt = [is_point_in_poly(poly, data[i,:]) for i in range(n)]
    return idxs[filt]

def fill(poly, data, idxs=None):

    x0, y0, x1, y1 = poly.bounds

    n, _ = data.shape
    if idxs is None: idxs = np.arange(n)

    x, y = data[:,0], data[:,1]

    filt_x = (x0 <= x) & (x <= x1)
    filt_y = (y0 <= y) & (y <= y1)
    filt = filt_x & filt_y

    data = data[filt,:]
    idxs = idxs[filt]
    
    return idxs
    
    n, _ = data.shape
    filt = [is_point_in_poly(poly, data[i,:]) for i in range(n)]

    return idxs[filt]


def create(poly, feature_size=200, pad_ratio = 0.3, n_procs=1):

    subshapes = sub_divide_poly(poly, feature_size, pad_ratio, n_procs)

    npoly = shapely.union_all([s.convex_hull for s in subshapes])
    polys = [npoly] if isinstance(npoly, Polygon) else npoly.geoms
    
    polys = [Polygon(p.exterior) for p in polys]
    npoly = shapely.union_all(polys)

    buff = feature_size/2
    npoly = npoly.buffer(buff).buffer(-buff)

    return npoly


def get_subdomain_info(x0, y0, x1, y1, target_size):

    p = target_size
    
    xl_glb = x1 - x0
    yl_glb = y1 - y0

    m = int(np.ceil(xl_glb//target_size))
    n = int(np.ceil(yl_glb//target_size))

    if m == 0: m = 1
    if n == 0: n = 1
    
    xl_loc = xl_glb/m
    yl_loc = yl_glb/n

    return xl_loc, yl_loc, m, n

def get_rectangle(x0, y0, xl, yl, pad_ratio=0.0):

    def get_pad_info(s0, sl):
        spad = sl*pad_ratio
        s0 = s0 - spad
        s1 = s0 + sl + 2*spad
        
        return s0, s1

    x0, x1 = get_pad_info(x0, xl)
    y0, y1 = get_pad_info(y0, yl)

    pts = ((x0, y0), (x1, y0), (x1, y1), (x0, y1)) 
    return Polygon(pts)
    
def get_balanced_slices(n, m, offset=0):
    sub_n = n//m
    p = n % m
    idxs = [] 
    i0 = offset
    
    for j in range(m):
        i1 = i0 + sub_n
        if (j < p): i1 += 1 
        idxs.append(slice(i0,i1))
        i0 = i1
    
    return idxs
