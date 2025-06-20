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
        
        
        
def compute_spectra(eta, dt, t=None, n_overlap=1024, nfft=2048*10, tlim=None):
    

    #print(dt)
    fs = 1.0/dt
    #print(fs)
    
    if not t is None:
        t = np.arange(len(eta))*dt

    if tlim is None:

        t0, t1 = t.min(), t.max()
    else: 
        t0, t1 = tlim


    n = int(np.round((t1-t0)/dt))

    nfft = n//4

    ti = np.arange(0, n+1)*dt + t0  


    eta_i = np.interp(ti, t, eta)



    #print(len(eta_i), len(ti))
    #print(eta_i.mean())
    #eta_i -= eta_i.mean()
    
    #window = np.bartlett(nfft)
    f, spec_den = sig.welch(eta_i, fs=fs, nperseg=nfft, scaling='density')

    df = f[1]-f[0]
    e_tot = np.sum(spec_den*df)
    Hrms = np.sqrt(e_tot*8)
    Hmo = np.sqrt(2.0)*Hrms

    energy=np.trapezoid(spec_den, f)
    
    return f, spec_den, Hmo, energy

# sub_ffts Number of sub ffts to performs, i.e., nfft scales with size of data 
def compute_spectra2(eta, dt, t=None, sub_ffts=1, scaling='density', tlim=None, **welch_kwargs):
    
    fs = 1.0/dt
    
    if not t is None:
        if not tlim is None:
            i0, i1 = [int(np.round(x/dt)) for x in tlim]
            eta = eta[i0:i1+1]
    else:

        # Get indices of limits
        t0, t1 = t.min(), t.max() if tlim is None else tlim

        # Interpolating onto equispaced grid 
        n = int(np.round((t1-t0)/dt))
        ti = np.arange(0, n+1)*dt + t0  
        #eta = np.interp(ti, t, eta)

    
    if len(welch_kwargs) == 0:
        kwargs = dict(
            nfft = len(eta)//sub_ffts
            scaling = scaling
        )
    else:
        # Overidding other sub_ffts and scaling if unspecfied kwargs is given,
        # assumes unspecfied kwargs is a sig.welch kwarg and is not fs
        kwargs = welch_kwargs.copy()

    f, spec_den = sig.welch(eta, fs=fs, **kwargs)
    energy = np.trapezoid(spec_den, f)

    hmo = energy2hmo(energy)

    return f, spec_den, hmo, energy

def energy2hmo(energy):
    hrms = np.sqrt(energy*8)
    hmo = np.sqrt(2.0)*hrms
    return hmo

