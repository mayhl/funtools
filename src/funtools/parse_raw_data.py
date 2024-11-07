import numpy as np
import pandas as pd
import warnings
import rasterio

from pyproj import CRS


def read_csv(fpath, crs_info, xyz_idxs=None, **kwargs):

    if 'delimiter' in kwargs:
        data = np.loadtxt(fpath, **kwargs)
    else:
        
        delimiters = [' ', ',']
        is_data_load = False
        
        for d in delimiters:
            try:
                data = np.loadtxt(fpath, delimiter = d, **kwargs)             
            except ValueError as e:
                continue

            is_data_load = True
            break
        
        # Throw default error 
        if not is_data_load: np.loadtxt(fpath, **kwargs)
        
                
    if not xyz_idxs is None:
        return data[:, xyz_idxs]

    _, d = data.shape
    assert d >= 3, "Less than three (3) columns of data read. File Path: %s." % fpath

    if d > 3:
        msg = (
            "Reading the first three columns of data as xyz, more than three columns detected. File Path: %s."
            % fpath
        )
        warnings.warn(msg)

    return data[:, 0:3], crs_info, {}


def read_pandas_csv(fpath, xyz_cols, crs_info, **kwargs):

    df = pd.read_csv(fpath, **kwargs)
    return df[xyz_cols].values, crs_info, {}


def read_geotiff(fpath, factor=1):
    # Reading data
    img = rasterio.open(fpath)
    m, n = img.height, img.width
    z = img.read(1)

    # Transforming pixel points to coordinates
    n, m = z.shape
    cols, rows = np.meshgrid(np.arange(m), np.arange(n))
    x, y = rasterio.transform.xy(img.transform, rows, cols)

    #  Orienting matrix for increase x, y points
    data = [np.array(s) for s in [x, y, z]]
    x, y = data[0][0, :], data[1][:, 0]
    if y[1] - y[0] < 0:
        data = [np.flipud(s) for s in data]
    if x[1] - x[0] < 0:
        data = [np.fliplr(s) for s in data]

    idxs = data[2] != img.nodata
    x, y = data[0][0, :], data[1][:, 0]
    mask = dict(x=x, y=y, data=idxs)

    # Flattening 2D grid to scatter grid and remove null points
    data = np.vstack([s.flatten() for s in data]).T[idxs.flatten(), :]

    # Converting rasterio CRS class to pyproj CRS class
    crs = img.read_crs().to_epsg()
    print(crs, fpath)
    crs = CRS.from_epsg(crs) #img.read_crs().to_epsg())
    return data, crs, {"mask": mask}


def read_ww3_mesh(fpath, crs_info):
    skiprows = 4
    with open(fpath, 'r') as fh:
        for i in range(skiprows): fh.readline()
        n = int(fh.readline())


    data = np.loadtxt(fpath, skiprows=skiprows+1, max_rows=n)[:,1:]

    x = data[:,0]
    y = data[:,1]

    # Hacky patch for now
    import pyproj

    myproj = pyproj.Proj(crs_info)

    x, y = myproj(x ,y)

    data[:,0]=x
    data[:,1]=y

    return data, crs_info, {}