import numpy as np

def get_index(s, s0, mode='closest'):

    diff = s - s0
    i = np.argmin(np.abs(diff))

    imap = {'closest': i,
            'left'   : i if diff[i] >= 0 else i + 1,
            'right'  : i if diff[i] <= 0 else i - 1}

    if not mode in imap:
        raise Exception("Invalid mode '%s'." % mode)

    return imap[mode]

def get_slice(s, s0, s1):
    if s0 is None: s0 = s.min()
    if s1 is None: s1 = s.max()
    i0 = get_index(s, s0, 'left')
    i1 = get_index(s, s1, 'right')
    return slice(i0, i1)

def crop(x, y, zz, x0=None, x1=None, y0=None, y1=None):
    sx = get_slice(x, x0, x1)
    sy = get_slice(y, y0, y1)
    return x[sx], y[sy], zz[sy, sx]

def flatten_left(x, y, zz, z_flat, xl_flat, xl_transition):

    x0 = x.min()
    x1 = x0 + xl_flat
    x2 = x1 + xl_transition
    sx_flat = get_slice(x, x0, x1)
    sx_tran = get_slice(x, x1, x2)


    zz_flat = zz.copy()
    zz_flat[:,sx_flat] = z_flat

    ramp = (x[sx_tran]-x1)/(x2-x1)

    
    n, m = zz.shape
    i1, i2 = sx_tran.start, sx_tran.stop
    for j in range(n):
        zz_flat[j, sx_tran] = z_flat*(1-ramp) + ramp*zz[j, i2]
        
    
#     idx = zz_flat[:, sx_tran] > 
#     zz_flat[:, sx_tran]

    return x, y, zz_flat

def _blend_start(s, zz, n, ds, axis, ramps=None):

    print(ramps)
    p, q = get_ramp_coeffs(n) if ramps is None else ramps

    s_ext = get_extend_domain(s, ds, n, mode='start')
    zz_ext = _get_blend_extension(zz, axis, p, q)
    
    s = np.concatenate([s_ext, s])
    zz = np.concatenate([zz_ext, zz], axis=axis) 

    return s, zz
 
def _blend_stop(s, zz, n, ds, axis, ramps=None):

    p, q = get_ramp_coeffs(n) if ramps is None else ramps

    s_ext = get_extend_domain(s, ds, n, mode='stop')
    zz_ext = _get_blend_extension(zz, axis, p, q)
    
    s = np.concatenate([s, s_ext])
    zz = np.concatenate([zz, zz_ext], axis=axis) 

    return s, zz
 
def _blend_both(s, zz, n, ds, axis):

    p, q = get_ramp_coeffs(2*n)

    # Logic issue?: Why flip in start/stop and concat order 
    s_ext0 = get_extend_domain(s, ds, n, mode='stop')
    zz_ext0 = _get_blend_extension(zz, axis, p[n:], q[n:])
    
    s_ext1 = get_extend_domain(s, ds, n, mode='start')
    zz_ext1 = _get_blend_extension(zz, axis, p[:n], q[:n])
    
    s = np.concatenate([s_ext0, s, s_ext1])
    zz = np.concatenate([zz_ext0, zz, zz_ext1], axis=axis) 
    
    return s, zz

def _get_blend_extension(zz, axis, p, q):

    s0 = slices_by_axis_index(-1, axis, zz.ndim)
    s1 = slices_by_axis_index( 0, axis, zz.ndim)
    
    n = len(p)
    ext_shape = list(zz.shape)
    ext_shape[axis] = n
    zz_ext = np.zeros(ext_shape)
    for i in range(n):
        si = slices_by_axis_index(i, axis, zz.ndim)
        zz_ext[si] = zz[s0]*p[i] + zz[s1]*q[i]  

    return zz_ext

def blend_perodic(s, zz, length, axis, mode = 'both'):

    mmap = {'both' : _blend_both ,
            'start': _blend_start,
            'stop' : _blend_stop }

    if not mode in mmap:
        raise Exception("Invalid mode '%s'." % mode)

    if mode == 'both': length /= 2

    ds = np.mean(np.diff(s))
    n = int(length//ds)

    return mmap[mode](s, zz, n, ds, axis)

def get_ramp_coeffs(n):
    dr = 1/(n+1)
    q = np.arange(dr, 1, dr)
    p = 1-q
    return p, q
    
def get_extend_domain(s, ds, n, mode):

    s_ext = np.arange(1, n+1)*ds

    if mode == 'stop': 
        return  -np.flipud(s_ext)+ s.min()
        
    if mode == 'start':  
        return s_ext + s.max()

   
    raise Exception("Invalid mode '%s'." % mode)

def slices_by_axis_index(i, axis, n):
    return tuple([i if j==axis else slice(None) for j in range(n)])

def cut_max_depth(zz, depth_max):
    idx = zz < depth_max
    zz[idx] = depth_max
    return zz
    
def create_shore_bathy(x, y, zz, xl_flat, xl_transition, yl_blend, flat_depth=None):

    args = x, y, zz
    kwargs = dict(x0=-100, y1=610, x1 = 700)
    args = crop(*args, **kwargs)

    if flat_depth is None:
        kwargs = dict(x1 = xb.min() + xl_flat)
        _, _, zz_flat = crop(*args, **kwargs)
        flat_depth = np.round(zzc_flat.mean())

    x, y, zz = flatten_left(*args, flat_depth, xl_flat, xl_transition)
    

    y, zz = blend_perodic(y, zz, yl_blend, axis=0, mode = 'both')

    zz = cut_max_depth(zz, flat_depth)
    
    
    return x, y, zz
