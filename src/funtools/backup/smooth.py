
import numpy as np

from funtools.subgrid import compute_equislices
from funtools.parallel import simple as sparallel

from scipy.ndimage import gaussian_filter
from scipy import signal

class PaddedSubgrid():
    
    
    def __init__(self, sx, sy, m , n, pad=0):
          
        n_loc = sy.stop - sy.start
        m_loc = sx.stop - sx.start
        

        self._data = np.zeros([n_loc+2*pad, m_loc+2*pad])

        self._pad = pad
    
        # Slice mapping between main parent matrix and sub-grid matrix 
        self._sx = sx
        self._sy = sy
        
        self._sx_loc = slice(pad, m_loc + pad)
        self._sy_loc = slice(pad, n_loc + pad)
        
        
        # Slice mapping between main parent matrix and sub-grid matrix 
        # with padded cells 
        i0, i1 = sx.start - pad, sx.stop + pad
        j0, j1 = sy.start - pad, sy.stop + pad
        
        i0_loc, i1_loc = 0, m_loc + 2*pad
        j0_loc, j1_loc = 0, n_loc + 2*pad
 
        # Modifying indices/slices if sub-grid is on boundary
        self._is_south_boundary = sy.start == 0
        if self._is_south_boundary:
            j0 = 0
            j0_loc = pad
            self._ss     = slice(sy.start - pad, sy.start)
            self._ss_loc = slice(0             ,  pad    )

        self._is_west_boundary = sx.start == 0
        if self._is_west_boundary:
            i0 = 0
            i0_loc = pad
            self._sw     = slice(sx.start - pad, sx.start)
            self._sw_loc = slice(0             ,  pad    )
        
        self._is_north_boundary = sy.stop  == n
        if self._is_north_boundary:
            j1 = n
            j1_loc = n_loc + pad
            self._sn     = slice(sy.stop    , sy.stop + pad)    
            self._sn_loc = slice(n_loc+pad-1, n_loc + 2*pad)
   
        self._is_east_boundary = sx.stop  == m         
        if self._is_east_boundary:
            i1 = m
            i1_loc = m_loc + pad
            self._se     = slice(sx.stop    , sx.stop + pad)
            self._se_loc = slice(m_loc+pad-1, m_loc + 2*pad)
        
        self._sx_cp = slice(i0, i1)
        self._sy_cp = slice(j0, j1)
        self._sx_cp_loc = slice(i0_loc, i1_loc)
        self._sy_cp_loc = slice(j0_loc, j1_loc)
            
    @property
    def d(self): return self._data
    
    
    def copy_from(self, data):
        
        pad = self._pad
        
        s_loc = (self._sy_cp_loc, self._sx_cp_loc)
        s_gbl = (self._sy_cp    , self._sx_cp    )
        self._data[s_loc]=data[s_gbl].copy()
        
        # If boundary, copy boundary day into padded cells
        if self._is_south_boundary:
            self._data[self._ss_loc,:] = self._data[pad,:]
            
        if self._is_north_boundary:
            idx = self._sy_loc.stop - 1
            self._data[self._sn_loc,:] = self._data[idx,:]
            
        if self._is_west_boundary:
            for i in range(self._sw_loc.start, self._sw_loc.stop):
                self._data[:, i] = self._data[:,pad]  
                
        if self._is_east_boundary:
            idx = self._sx_loc.stop - 1
            for i in range(self._se_loc.start, self._se_loc.stop):
                self._data[:, i] = self._data[:,idx]               

    def copy_to(self, data):
        
        s_loc = (self._sy_loc, self._sx_loc)
        s_gbl = (self._sy    , self._sx    )
        data[s_gbl] = self._data[s_loc].copy()
              
        
    def update(self, data):
        s_loc = (self._sy_loc, self._sx_loc)
        self._data[s_loc] = data
        
        
    def update_and_copy(self, loc_data, gbl_data):
        self.update(loc_data)
        self.copy_to(gbl_data)
        
        
def gaussian_convolution(data, pad=2, sigma=1, target_batches=100, n_procs=1):

    (sys, sxs), n_batches = compute_equislices(data, p=target_batches)
    slices = [(sx, sy) for sx in sxs for sy in sys]

    n, m = data.shape
    grids = [PaddedSubgrid(*s, m, n, pad) for s in slices]
    
    for g in grids: g.copy_from(data)

    args = [(g.d, pad, sigma) for g in grids]

    rnt_vals = sparallel(_gaussian_convolution, n_procs, args, p_desc="Convolving")
    
    new_data = np.zeros(data.shape)
    for g, d in zip (grids, rnt_vals):
        g.update_and_copy(d, new_data)
    
    return new_data 

def _gauss_kern(l=5, sig=1.):

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def _gaussian_convolution(data, pad, sigma):
    kernel = _gauss_kern(l=2*pad+1, sig=sigma)
    return signal.convolve2d(data, kernel, boundary='symm', mode='valid')
 