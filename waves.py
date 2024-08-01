from functools import partial
import numpy as np
import scipy.signal as sig
from scipy import interpolate

def newton_iterative(f, df, x0, tolerance=10**-8, max_iterations=100):
    
    x = x0
    for i in range(max_iterations):
        
        dx = -f(x)/df(x)
        x += dx
        if np.abs(dx) < tolerance: break
    
    return x

g=9.81

def _f(k, w, h):
    return g*k*np.tanh(k*h)-w**2

def _dfdk(k, w, h):
    return g*np.tanh(k*h)+g*k*np.cosh(k*h)**-2



def solve_dispersion(w=None, h=None, k=None, tolerance=10**-8, max_iterations=100):
    
    is_w, is_h, is_k = chks = tuple((x is not None for x in [w, h, k]))
    
    n = int(np.sum(chks))
    if not n == 2:
        raise Exception("Must specify only two of w, h, and k")
        

    if not is_k:
        # Linear approximation
        k0 = np.sqrt(w**2/(g*h))
        f = partial(_f, w=w, h=h)
        df = partial(_dfdk, w=w, h=h)
        return newton_iterative(f, df, k0)
        
    elif not is_w:
        raise NotImplementedError()
    elif not is_h:
        raise NotImplementedError()

    # Sanity check
    raise Exception("Unexpected State")
        
        
        
def compute_spectra(eta, dt, t=None, n_overlap=1024, nfft=2048):
    
    fs = 1.0/dt
    
    
    if not t is None:
        t0, t1 = t.min(), t.max()
        ti = np.arange(t0, t1+dt/2, dt)  
        eta = np.interp(ti, t, eta)

    
    window = np.bartlett(nfft)
    f, spec_den = sig.welch(x=eta, fs=fs, window=window, nperseg=nfft,noverlap=n_overlap, nfft=nfft, scaling='density')

    df = f[1]-f[0]
    e_tot = np.sum(spec_den*df)
    Hrms = np.sqrt(e_tot*8)
    Hmo = np.sqrt(2.0)*Hrms

    energy=np.trapezoid(spec_den, f)
    
    return f, spec_den, Hmo, energy



