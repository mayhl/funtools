#import json
import os
from glob import glob
import numpy as np 
import warnings

# dpath - Path to directory containing file
# bname - Base file name, e.g. NAME_XXXXX, where X is a number
# num - file number as integer 
# m - Mglob, number of x grid points


def load_data(dpath, fname, n, m):
    fpath = os.path.join(dpath, fname)
    data = np.fromfile(fpath, dtype='<f8')
    return data.reshape([n,m])

def load_data_step(dpath, base_fname, step_num, n, m):
    fname = "%s_%05d" % (base_fname, step_num)
    return load_data(dpath, fname, n, m)

def load_data_1d(dpath, fname, m):
    return load_data(dpath, fname, 3, m)[1,:]

def load_data_step_1d(dpath, fname, step, m):
    return load_data_step(dpath, fname, step, 3, m)[1,:]

    
    
def process_field_output(dpath):
    
    # Get output files match *_XXXXX mask, where X is a number
    mask = '*_' + ('[0-9]' * 5)
    mask_fpath = os.path.join(dpath, mask)
    fnames = [os.path.basename(f) for f in glob(mask_fpath)]
    
    # Extracting files numbers and field names from file names, i.e., FIELD_NUMBER
    cast = lambda a, b: (a, int(b))
    unique = lambda a, b: (sorted(np.unique(a)), sorted(np.unique(b)))
    fields, nums = unique(*zip(*[cast(*n.split('_')) for n in fnames]))
    nf, nn = len(fields), len(nums)
    
    # Removing error number
    error_num = 99999
    if error_num in nums: 
        warnings.warn("Error file detected, simulation was UNSTABLE. Removing file numbers from list.") 
        nums.remove(error_num)
        
    if not nn*nf == len(fnames):
        raise Exception("Number of files found does not match number of field names multipled by file numbers.")
    
    if np.any(np.diff(nums) != 1):
        warnings.warn("Missing file numbers detetected.")
        
    return fields, len(fields), nums, len(nums)


def parse_str(val):

    def cast_type(val, cast):
        try:
            return cast(val)
        except ValueError:
            return None

    ival = cast_type(val, int)
    fval = cast_type(val, float)
    
    if ival is None and fval is None:
        if type(val) is str and len(val) == 1:
            if val[0] == 'T': return True
            if val[0] == 'F': return False 
           
        return str(val)
   
    elif ival is not None and fval is not None:
        return ival if ival == fval else fval
    elif fval is not None: # and ival is None
        return fval
    else: # fval is None, ival is not None 
        # Case should not be possible 
        raise Exception('Unexpected State')
        
def read_input_file(fpath):
    """
    Convert FUNWAVE input/driver file to dictionary

    :param fpath: Path to FUNWAVE input/driver file
    :type fpath: str
    """

    def _split_first(line, char):

        first, *second = line.split(char)
        second = char.join(second)
        return first, second 

    def _filter_comment(line):

        if "!" not in line: return False, line, None 
        first, second = _split_first(line, "!")
        return True, first, second 


    with open(fpath, 'r') as fh: lines = fh.readlines()

    params = {}
    for line in lines: 

        if not '=' in line: continue
        first, second = _split_first(line, "=")        

        is_comment, name, _  = _filter_comment(first.strip())
        if is_comment: continue

        is_comment, val_str, _ = _filter_comment(second.strip())
        
        params[name] = parse_str(val_str)

    return params                 


def _get_banner(title, indent = 10, max_length = 80):

    banner = '! ' + ''.join(['-']*indent)
    banner += ' ' + title + ' '
    n = max_length - len(banner)
    if n < 1: n = 1
    banner += ''.join(['-']*n)
    return "\n" + banner + "\n"

def _get_parameter_line(name, value):

    if isinstance(value, bool):
        value_str = 'T' if value else 'F'
    elif isinstance(value, int):
        value_str = "%d" % value
    elif isinstance(value, float):
        value_str = "%f" % value
    elif isinstance(value, str):
        value_str = value
    else:
        value_str = "%s" % value

    if '.' in value_str and value_str[-1] == '0':
        value_str = value_str.strip('0')
        if value_str[0] == '.': value_str = '0' + value_str
        if value_str[-1] == '.': value_str += '0'
    
    return "%s = %s\n" % (name, value_str)

def create_input_file(fpath, params):
    
    categories_maps = {
        'General': ['TITLE'],
        'Parallel': ['PX', 'PY'],
        'Grid': ['DX', 'DY', 'Mglob', 'Nglob', 'StretchGrid'],
        'Bathy': ['DEPTH_TYPE', 'DEPTH_FILE', 'WaterLevel'],
        'Time': ['TOTAL_TIME', 'PLOT_INTV', 'SCREEN_INTV'],
        'Hot Start': ['HOT_START', 'INI_UVZ'],
        'Wave Maker': ['WAVEMAKER', 'WAVE_DATA_TYPE', 'DEP_WK', 'Xc_WK', 'Yc_WK',
                       'FreqPeak', 'FreqMin', 'FreqMax', 'Hmo',
                       'GammaTMA', 'Sigma_Theta','Delta_WK',
                        'EqualEnergy', 'Nfreq', 'Ntheta','alpha_c', 'Tperiod', 'AMP_WK'],
        'Boundary Conditions': ['PERIODIC', 'DIFFUSION_SPONGE', 'FRICTION_SPONGE',
                                'DIRECT_SPONGE', 'Csp', 'CDsponge', 'Sponge_west_width',
                                'Sponge_east_width', 'Sponge_south_width',
                                'Sponge_north_width'],
        'Tidal Boundary Forcing': ['TIDAL_BC_GEN_ABS' ,'TideBcType', 'TideWest_ETA'],
        'Numerics': ['Gamma1', 'Gamma2', 'Gamma3', 'Beta_ref',
                     'HIGH_ORDER', 'CONSTRUCTION', 'CFL', 'FroudeCap', 
                     'MinDepth', 'MinDepthFrc'],
        'Breaking': ['DISPERSION', 'SWE_ETA_DEP', 'SHOW_BREAKING',
                     'VISCOSITY_BREAKING', 'Cbrk1', 'Cbrk2', 'WAVEMAKER_Cbrk'],
        'Friction' : ['Friction_Matrix', 'Cd'],
        'Mixing': ['STEADY_TIME', 'T_INTV_mean', 'C_smg'],
        'Stations': ['NumberStations', 'STATIONS_FILE', 'PLOT_INTV_STATION'],
        'Output': ['FIELD_IO_TYPE', 'DEPTH_OUT', 'U', 'V', 'ETA', 'Hmax',
                   'Hmin', 'MFmax', 'Umax', 'VORmax', 'Umean', 'Vmean',
                   'ETAmean', 'MASK', 'MASK9', 'SXL', 'SXR', 'SYL', 'SYR',
                   'SourceX', 'SourceY', 'P', 'Q', 'Fx', 'Fy', 'Gx', 'Gy',
                   'AGE', 'TMP', 'WaveHeight', 'OUT_NU']
    }
    
    
    for key in params:
        
        is_found = False
        for subparams in categories_maps.values():
            if key in subparams:
                is_found = True
                break
                
        if not is_found:
            raise Exception("Parameter '%s' has no category." % key)
        
        
    
    with open(fpath, 'w') as fh: 
        for category, subparams in categories_maps.items():
            
            is_first = True 

            for subparam in subparams:
                if subparam in params:
                    
                    # Only create banner if at least one parameter is found
                    if is_first:
                        fh.write(_get_banner(category))
                        is_first = False 

                    fh.write(_get_parameter_line(subparam, params[subparam]))
                    
                          
def is_string_in_lines(string, lines):  
    
    for line in lines:
        if string in line: return True
    
    return False

def any_strings_in_lines(strings, lines):
    
    for s in strings:
        if is_string_in_lines(s, lines):
            return True
        
    return False

def check_simulation(dpath):

    cfls = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    cfls = list(reversed(sorted(cfls)))

    pbs_fpath = os.path.join(dpath, 'run_script.pbs')

    name = None
    with open(pbs_fpath, 'r') as fh:

        line = fh.readline()
        while line:
            if '#PBS -N' in line:
                name = line.replace('#PBS -N', '').strip()
                break

            line = fh.readline()

    if name is None: raise Exception("Could not parse name from pbs file")

    # Sorted o files by highest job id (newest)
    o_fpaths = list(reversed(sorted(glob(os.path.join(sim_dpath, "%s*" % name)))))

    if len(o_fpaths) < 1: return 0


    with open(o_fpaths[0], 'r') as fh:
        lines = fh.readlines()

    success_msgs = ['Normal Termination!']

    if any_strings_in_lines(success_msgs, lines): 
        return 1

    error_msgs = ['PRINTING FILE NO. 99999']
    if any_strings_in_lines(error_msgs, lines):

    
        input_fpath = os.path.join(sim_dpath, 'input.txt')

        params = read_input_file(input_fpath)

        cfl = params['CFL']

        is_valid = False
        for c in cfls:
            if c < cfl:
                is_valid = True
                break

        if not is_valid: raise Exception("Min CFL reached")

        params['CFL'] = c    
        
        create_input_file(input_fpath, params)
        
   
        return 2

    # Assume simulations is still running
    return 3