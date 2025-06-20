import numpy as np
from functools import reduce

# Sub-divides an n-dimensional grid into a subgrid with nearly
# equal side length, i.e., as square-ish as possible 
# mode = 'size': side/linear lengths are as close to p as possible
# mode = 'batch': total number of sub-grids as close to p as possible 
def compute_equipartition(data, p, mode='size'):

    _, ndim = data.shape
    shape, offsets = zip(*[_get_linear_info(data, i) for i in range(ndim)])
    vol = np.prod(shape)

    if mode == 'size':
        
        sloc = np.sqrt(vol/p)
        n_batches = np.array([int(np.round(n/sloc)) for n in shape])

    elif mode == 'batch':
        
        shape = np.array(shape)
        ratios = shape/shape[0]
        nloc = (p/np.prod(ratios))**(1/ndim)
        n_batches = np.round(ratios*nloc).astype(int)
        
    else:
        raise Exception()
        
    n_batches[n_batches==0]=1
    n_batches = [int(x) for x in n_batches]

    args = zip(shape, n_batches, offsets)
    ranges = [_compute_equipartition(*a) for a in args]

    return ranges, n_batches, shape

def _get_linear_info(data, i):
    s = data[:,i]
    l, s0 = s.max()-s.min(), s.min()
    return float(l), float(s0)

def _compute_equipartition(length, n, offset):
    s = np.linspace(0, length, n+1) + offset
    return [(float(s[i]), float(s[i+1])) for i in range(n)]

def _filter_data(data, s, s0, s1):
    filt = (s0 <= s) & (s <= s1)
    return data[filt,:]

def _subdivide_by_ranges(data, ranges, level=0):

    ndim = len(ranges)

    s = data[:, level]
    if ndim==1:
        fdata = [_filter_data(data, s, *r) for r in ranges[0]]    
    else:
        args = (ranges[1:], level+1)
        func = lambda s0, s1: _subdivide_by_ranges(_filter_data(data, s, s0, s1), *args)

        fdata = []
        for s0, s1 in ranges[0]: fdata.extend(func(s0, s1))
            
        #fdata = [func(s0, s1) for s0, s1 in ranges[0]]
        #print([type(f) for f in fdata])

        #print(type(fdata))
        #print(type(fdata[0]))
        #fdata = np.concatenate(fdata, dtype=list)


    return fdata

def subdivide_by_ranges(data, ranges):
    
    _, ndim = data.shape
    assert ndim >= len(ranges), "Number dimensions in data must be greater than or equal to ranges."
    return _subdivide_by_ranges(data, ranges)

# Similar to compute_equipartition but for indices/slices
# for an equispaced grid 
def compute_equislices(data, p, offsets=None, mode='size'):

    ndim = data.ndim
    if offsets is None: offsets = [0 for i in range(ndim)]
    noffset = len(offsets)
    assert noffset == ndim, "Number of offsets do not match data dimensions."

    shape = data.shape
    if mode =='size':
        sloc = np.sqrt(data.size/p)
        n_batches = [int(np.round(n/sloc)) for n in shape]
    elif mode == 'batch':
        ratios = np.array(shape)/shape[0]
        nloc = (p/np.prod(ratios))**(1/ndim)
        n_batches = np.round(ratios*nloc).astype(int)
    else:
        raise Exception()

    args = zip(shape, n_batches, offsets)
    slices = [_compute_equislices(*a) for a in args]

    return slices, n_batches

def _compute_equislices(n, p, offset=0):

    # Smallest sub-grid size
    sub_n = n//p
    # Number of sub-grids with an extra point
    # i.e. n = (sub_n+1)*offset + sub_n*(p-offset)
    #        = sub_n*p + offset 
    cutoff = n % p
    
    # Computing indices/slices of each e
    i0 = offset
    slices = [] 
    for j in range(p):
        i1 = i0 + sub_n + (j < cutoff)
        slices.append(slice(i0, i1))
        i0 = i1
    
    return slices

def linear2coord_index(index, dims):
    n = len(dims)
    coords = ()
    for i in range(n-1):
        fact = np.prod(dims[i+1:])
        c = int(index//fact)
        coords += (c, )
        index -= c*fact

    return coords + (int(index),)
  
def coord2linear_index(coords, dims):
    n = len(dims)
    assert len(coords) == n, "Error"
    i = coords[-1] + np.sum([coords[i]*np.prod(dims[i+1:]) for i in range(n-1)])
    return int(i)



def compute_factors(n):    
    factors= reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))
    factors = np.concatenate( [factors, list(reversed(factors))])
    factors = [(factors[i], factors[i+1]) for i in range(0, len(factors), 2)]
    return factors


def _batch_size(n, target_size):
    if n < target_size: return 1
    return int(np.round(n/target_size))

def get_optimal_parallel(shape, threads_per_node, target_size=150, max_nodes=30):

    n_glob, m_glob = shape
    t_glob = n_glob*m_glob
    n_batch = _batch_size(n_glob, target_size)
    m_batch = _batch_size(m_glob, target_size)
    
    t_batch = n_batch*m_batch

    t_nodes = _batch_size(t_batch, threads_per_node)

    if t_nodes > max_nodes: t_nodes = max_nodes

    t_threads = t_nodes*threads_per_node

    factors = compute_factors(t_threads)

    ratios = [(n_glob/py)/(m_glob/px) for px, py in factors]
    ratios = np.array([r if r > 1 else 1/r for r in ratios])
    
    i = np.argmin(ratios-1)

    px, py = factors[i]

    return t_nodes, py, px
