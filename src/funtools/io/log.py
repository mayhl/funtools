# import json
import os
from glob import glob
import numpy as np
import warnings


from type import list

# dpath - Path to directory containing file
# bname - Base file name, e.g. NAME_XXXXX, where X is a number
# num - file number as integer
# m - Mglob, number of x grid points


def is_string_in_lines(string: str, lines: list[str]) -> bool:

    for line in lines:
        if string in line:
            return True

    return False


def any_strings_in_lines(strings: list[str], lines: list[str]) -> bool:

    for s in strings:
        if is_string_in_lines(s, lines):
            return True

    return False


class Log:

    def __init__(self, path: Path):

        path = path / "LOG.txt"


def check_simulation(dpath):

    cfls = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    cfls = list(reversed(sorted(cfls)))

    pbs_fpath = os.path.join(dpath, "run_script.pbs")

    name = None
    with open(pbs_fpath, "r") as fh:

        line = fh.readline()
        while line:
            if "#PBS -N" in line:
                name = line.replace("#PBS -N", "").strip()
                break

            line = fh.readline()

    if name is None:
        raise Exception("Could not parse name from pbs file")

    # Sorted o files by highest job id (newest)
    o_fpaths = list(reversed(sorted(glob(os.path.join(sim_dpath, "%s*" % name)))))

    if len(o_fpaths) < 1:
        return 0

    with open(o_fpaths[0], "r") as fh:
        lines = fh.readlines()

    success_msgs = ["Normal Termination!"]

    if any_strings_in_lines(success_msgs, lines):
        return 1

    error_msgs = ["PRINTING FILE NO. 99999"]
    if any_strings_in_lines(error_msgs, lines):

        input_fpath = os.path.join(sim_dpath, "input.txt")

        params = read_input_file(input_fpath)

        cfl = params["CFL"]

        is_valid = False
        for c in cfls:
            if c < cfl:
                is_valid = True
                break

        if not is_valid:
            raise Exception("Min CFL reached")

        params["CFL"] = c

        create_input_file(input_fpath, params)

        return 2

    # Assume simulations is still running
    return 3
