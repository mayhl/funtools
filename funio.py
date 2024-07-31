#import json
import os
from glob import glob
import numpy as np 
import warnings

# dpath - Path to directory containing file
# bname - Base file name, e.g. NAME_XXXXX, where X is a number
# num - file number as integer 
# m - Mglob, number of x grid points

    
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


