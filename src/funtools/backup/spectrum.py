import numpy as np
import param
import os


from funtools.waves import solve_dispersion

class _Spectrum(param.Parameterized):

    _pos_def_kwargs = dict(
        bounds = (0, None)
    )
    gamma = param.Number(3.3, **_pos_def_kwargs)
    gravity = param.Number(9.81, **_pos_def_kwargs)
    depth = param.Number(40, **_pos_def_kwargs)
    freq_min = param.Number(0.01, **_pos_def_kwargs)
    freq_peak = param.Number(0.1, **_pos_def_kwargs)
    freq_max = param.Number(0.3, **_pos_def_kwargs)
    n_freq = param.Integer(42, **_pos_def_kwargs)
    hm0 = param.Number(1,**_pos_def_kwargs)
    equal_energy = param.Boolean(default=False)
    interp_mode = param.Selector(default='simpson', objects=['left', 'trapezoidal', 'simpson'])
    n_sub = param.Integer(100, **_pos_def_kwargs)
    n_theta = param.Integer(1, **_pos_def_kwargs)
    theta_max = param.Number(45, **_pos_def_kwargs)
    sigma_theta = param.Number(10, **_pos_def_kwargs)
    theta_peak = param.Number(0)
    

    _GRAVITY_ = param.Number(9.81, readonly=True)


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._discrete_energy()

    def update(self, **kwargs):
        self.param.update(**kwargs)
    
    def sigma(self, f):
        sigma = np.zeros(f.shape)
        filt = f <= self.freq_peak
        sigma[filt] = 0.07
        sigma[~filt] = 0.09
        return sigma 

    def _compute(self, f):
        s = f/self.freq_peak 
        
        Ep = self.gravity**2*(2*np.pi)**(-4)*f**(-5)

        phi_PM = np.exp(-5/4*s**(-4))
        b = self.phi(f)
        
        y = (s-1)**2/(2*self.sigma(f)**2)
        phi_J = np.exp(np.log(self.gamma)*np.exp(-y))

        a = Ep*phi_PM
        b = phi_J*self.phi(f)

        return Ep*phi_PM*phi_J*self.phi(f)

    def _get_coeffs(self, n):

        mode = self.interp_mode
        
        if mode == 'left':
            coeff = np.ones(n+1)
            coeff[-1]=0
            return 1, coeff
            
        if mode == 'trapezoidal':
            coeff = np.ones(n+1)
            coeff[0] = coeff[-1]=0.5
            return 1, coeff

        if mode == 'simpson':
            
            coeff = 4*np.ones(n+1)

            idx = np.arange(0,n+1)
            filt = idx % 2 == 0
            coeff[filt] = 2

            coeff[0] = coeff[-1]=1
            return 1/3, coeff
                 
        raise NotImplementedError("Interpolation mode %s not implemented." % mode)
            
    def __integrate(self, df, Ei, n):
        c0, coeff = self._get_coeffs(n) 
        return np.sum(Ei*coeff)*c0*df         

    def _get_interp_data(self, f0, f1, n):
        df = (f1-f0)/n
        f = np.arange(0,n+1)*df + f0
        Ei = self._compute(f)
        return df, f, Ei

    def _integrate(self, f0, f1, n):
        df, f, Ei = self._get_interp_data(f0, f1, n)
        return self.__integrate(df, Ei, n)
        
    def _normalize(self, f_bins, e_bins, e_total):

        f = (f_bins[:-1]+f_bins[1:])/2
        #f = f_bins[:-1]
        Ei = self._compute(f)

        coeff = self.hm0**2/16
        alpha = coeff*e_bins/e_total

        return f, f_bins, alpha
    
    def _compute_equal_frequency(self):
        n = self.n_sub*self.n_freq
        f0, f1 = self.freq_min, self.freq_max

        df, f, Ei = self._get_interp_data(f0, f1, n)
        e_total = self.__integrate(df, Ei, n)
        ns = self.n_sub
        idxs = [(i*ns, (i+1)*ns+1) for i in range(self.n_freq)]
        e_bins = np.array([self.__integrate(df, Ei[i0:i1], ns) for i0, i1 in idxs])
        f_bins = f[::ns]

        return  self._normalize(f_bins, e_bins, e_total)


    def _iterative_solve(self, f0, f1, e_target):  
        for i in range(10):
            f = self._integrate(f0, f1, self.n_sub) - e_target 
            df = self._compute(np.array([f1]))
            cor = -f/df[0] 
            f1 = f1 + cor
            if np.abs(cor/f1) < 10**-5: return f1
            
        
        return f1

    def _compute_equal_energy(self):

        n = self.n_sub*self.n_freq
        f0, f1 = self.freq_min, self.freq_max

        e_total = self._integrate(f0, f1, n)
        e_bin = e_total/self.n_freq

      
        df = (f1-f0)/self.n_freq
        f1 = self.freq_peak*0.9
        f_bins = [f0]
        for i in range(self.n_freq):

            #print(f0,f1)
            f1 = self._iterative_solve(f0, f1, e_bin)

            f_bins.append(f1)

            df = f1-f0
            f0 = f1
            f1 = f0+0.5*df
            
        e_bins = np.ones(self.n_freq)*e_bin
        f_bins = np.array(f_bins)

        return self._normalize(f_bins, e_bins, e_total)

    def _compute_theta(self):

        n = self.n_theta 
        dt = 2*self.theta_max/n
        theta_bins = (np.arange(0,n+1)*dt - self.theta_max)
        theta = (np.arange(0,n)*dt + dt/2 - self.theta_max)

        
        if n == 1: 
            theta = np.array([0])
            g = np.array([1])
            return theta, theta_bins, 1

        
        scale = np.pi/180
        theta_r = scale*theta 
        theta_bins_r = scale*theta_bins
        theta_peak_r = scale*self.theta_peak

        coeff = (self.sigma_theta*scale)**2/2
                
        ig = theta_bins_r/2
        for i in range(1,41):
            ig += np.exp(-coeff*i**2)*np.sin(i*(theta_bins_r-theta_peak_r))/i
        
        g = (ig[1:]-ig[:-1])/np.pi
    
        return theta, theta_bins, g
    
    
    @param.depends('gamma', 'gravity', 'depth', 'freq_min', 'freq_peak', 'freq_max', 'n_freq', 'hm0', 
                    'equal_energy', 'interp_mode', 'n_sub', 'n_theta', 'theta_max', watch=True)
    def _discrete_energy(self):
        
        args = self._compute_equal_energy() if self.equal_energy else self._compute_equal_frequency()
        f, f_bins, spec_1d = args

        self._freq = f
        self._freq_bins = f_bins

        self._theta, self._theta_bins, self._g = self._compute_theta()

        if self.n_theta == 1:
            self._disc_energy = spec_1d
        else:
            n = self.n_freq
            m = self.n_theta
            spec_2d = np.zeros([m,n])
            
            for j in range(m):
                spec_2d[j,:] = spec_1d*self._g[j]
            
            self._disc_energy = spec_2d

    def k_crit(self, kh=np.pi):
            h = self.depth
            k = kh/h
            return k
    def length_crit(self, kh=np.pi):
        k = self.k_crit(kh)
        return 2*np.pi/k

    def freq_crit(self, kh=np.pi):
            g = self._GRAVITY_
            k = self.k_crit(kh)
            w = np.sqrt(g*k*np.tanh(kh))
            return w/(2*np.pi)

    def period_crit(self,  kh=np.pi):
        return 1/self.freq_crit(kh)


    def _solve_dispersion(self, freq):

        w = 2*np.pi*freq
        h = self.depth
        g = self._GRAVITY_
     
        k = np.sqrt(1/(g*h))*w

        for i in range(10):

            tanhkh = np.tanh(k*h)    
            f = g*k*tanhkh - w**2
            df = g*tanhkh - g*k*(1-tanhkh*tanhkh)

            cor = -f/df
            k += cor

            if np.abs(cor) < 10**-6: break

        return k

    def _get_wave_info(self, freq):
        
        k_max = self._solve_dispersion(freq)
        length_max = 2*np.pi/k_max
        period_max = 1/freq
        return k_max, length_max, period_max 

    @param.depends('freq_max', watch=True)
    def _update_max_wave_info(self):
        self._k_max, self._length_max, self._period_max = self._get_wave_info(self.freq_max)

    @param.depends('freq_min', watch=True)
    def _update_min_wave_info(self):
        self._k_min, self._length_min, self._period_min = self._get_wave_info(self.freq_min)

    @param.depends('freq_peak', watch=True)
    def _update_peak_wave_info(self):
        self._k_peak, self._length_peak, self._period_peak = self._get_wave_info(self.freq_peak)

    @param.depends('depth', watch=True)  
    def _update_all_wave_info(self):
        self._update_max_wave_info()
        self._update_min_wave_info()
        self._update_peak_wave_info()

    def k_max(self): return self._k_max
    def k_min(self): return self._k_min
    def k_peak(self): return self._k_peak
    
    def length_max(self): return self._length_max
    def length_min(self): return self._length_min
    def length_peak(self): return self._length_peak

    def period_max(self): return self._period_max
    def period_min(self): return self._period_min
    def period_peak(self): return self._period_peak


    @property
    def freq(self): return self._freq
    @property
    def theta(self): return self._theta
        
    @property
    def freq_bins(self): return self._freq_bins
    @property
    def theta_bins(self): return self._theta_bins
    
    @property
    def discrete_energy(self):
        self._discrete_energy()
        #args = (self._freq, self._freq_bins, self._disc_energy)
        return self._disc_energy

    @property
    def discrete_amplitude(self):
        energy = self.discrete_energy
        return np.sqrt(2*energy)


class TMA(_Spectrum):
    def phi(self, f):
        omega_h = 2*np.pi*f*np.sqrt(self.depth/self.gravity)
        phi = np.zeros(f.shape)

        filt = omega_h <= 1
        phi[filt] = 0.5*omega_h[filt]**2
        filt = omega_h >= 2
        phi[filt] = 1
        filt = (1 < omega_h) & (omega_h < 2)
        phi[filt] = 1-0.5*(2-omega_h[filt])**2
        return phi

class JONSWAP(_Spectrum): 
    def phi(self, f): return 1


def _gen_spec_data_file(i, dpath, fname_mask, spec, *args):

    time, f, h, t = args
    spec.update(freq_peak=f, hm0=h, theta_peak=t)

    fname = fname_mask % i 
    fpath = os.path.join(dpath, fname)

    np.savetxt(fpath, spec.discrete_amplitude, fmt='%.4f', delimiter=' ', newline='\n')


    return time, fname

def _gen_spec_file(dpath, spec_fname, spec, time_files):

    fpath = os.path.join(dpath, spec_fname)
    
    with open(fpath, 'w') as fh:
        fh.write('! file contains freq/dire bins and file names for time-dependent spectra\n')
        fh.write('! numbers of freq  and direction bins (two integers)\n')
        fh.write('%d %d\n' % (spec.n_freq, spec.n_theta) )
        fh.write('! frequencies (1/s, column array)\n')

        for f in spec.freq: fh.write("%.8f\n" % f)
        fh.write('! directions (Cartesian degree, column array)\n')
        for t in spec.theta: fh.write("%.8f\n" % (t*np.pi/180))
        fh.write('! phases (empty = 0)\n')
        fh.write('0\n')
        fh.write('! time (s) and file name (string)\n')

        for t, fname in time_files:
            fh.write("%.3f\n" % t)
            fh.write("%s\n" % fname)
    
def generate_funwave_files(dpath, spec_fname, fname_mask, wave_data, spec): 
    n, _ = wave_data.shape
    args = [_gen_spec_data_file(i+1, dpath, fname_mask, spec, *wave_data[i,:])  for i in range(n)]
    _gen_spec_file(dpath, spec_fname, spec, args)
    