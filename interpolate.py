from src.parallel import simple as sparallel
from src.subgrid import compute_equislices
from scipy.interpolate import LinearNDInterpolator    
from scipy.interpolate import CloughTocher2DInterpolator


import numpy as np


def interpolate2d(xi, yi, data, target_size=200, mode='linear', n_procs=1):

    imap = {"linear": LinearNDInterpolator,
            "clough": CloughTocher2DInterpolator}

    assert mode in imap, "Unknown interpolation mode."
    interpolator = imap[mode]
    
    p = target_size

    xxi, yyi = np.meshgrid(xi, yi)
    zi = np.zeros(xxi.shape)

    xs, ys, zs = (data[:,i] for i in range(3))
    
    (sys, sxs), n_batches = compute_equislices(zi, target_size)

    print(n_batches)
    
    slices = [(x, y) for x in sxs for y in sys]
    # Flattening slices and filtering xyz data in subgrids
    args = [_filter_and_prep_args(xi, i, yi, j, data, interpolator) for i, j in slices]

    x, i, y, j, data, interpolator = args[0]
    print(data.shape)
    
    rtnvals = sparallel(_interpolate, n_procs, args, p_desc='Interpolating')
    #rtnvals = [interpolate(*x) for x in tqdm(sub_domains)]

    for i, j, z_sub in rtnvals: zi[j,i] = z_sub

    return zi

def _interpolate(x, i, y, j, data, interpolator):
    xx, yy = np.meshgrid(x,y)
    
    xs = data[:,0]
    ys = data[:,1]
    zs = data[:,2]

    interp = interpolator(list(zip(xs, ys)), zs)
    zz = interp(xx, yy)

    return i, j, zz
    
def _filter_and_prep_args(x, i, y, j, data, interpolator, pad_ratio=0.25):

    xs = data[:,0]
    ys = data[:,1]
    
    x, filt_x = _filter_linear_data(x, xs, i, pad_ratio)
    y, filt_y = _filter_linear_data(y, ys, j, pad_ratio)

    filt = filt_x & filt_y
    
    return x, i, y, j, data[filt,:], interpolator
    
def _filter_linear_data(s, ss, i, pad_ratio):

    s = s[i]
    s0, s1 = s.min(), s.max()
    sl = s1 - s0
    spad = sl*pad_ratio
    s0 -= spad
    s1 += spad

    filt = (s0 <= ss) & (ss <= s1)
    
    return s, filt
    
