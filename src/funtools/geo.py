import xyzservices
import warnings
from pathlib import Path
import requests
from tqdm import tqdm
from PIL import Image
import numpy as np
import pyproj
from scipy.interpolate import RegularGridInterpolator

import funtools.grid as fgrid
# Wrapper base class for downloading and combining image tiles from an XYZ tile provider
# based on Slippy Map index range for a given resolution (https://wiki.openstreetmap.org/wiki/Slippy_map)
# NOTES: 
#  1) Due to images conventions with positive y/vertical component denoting downwards 
#     instead of upwards, 0 and 1 suffixes are used to denote start and end 
#           ——— x ——►
#      ┌─────────────────── 
#   |  │ (x0, y0)  (x1, y0)          
#   y  │ 
#   |  │     
#   ▼  │ (x0, y1)  (x1, y1)
#
#  2) Requested tiles are saved locally
# DEV NOTES:
#  1) Resolution calculations assume tile sizes are 256x256 pixels
#  2) Combining tiles assumes all tiles are the same size from a given provider
class XYZImageService(xyzservices.lib.TileProvider):

    @classmethod
    def create(cls, obj, access_token=None, tiles_dpath=None):

        if not isinstance(obj, xyzservices.lib.TileProvider):
            raise Exception('Object is not an instance of xyzservices.lib.TileProvider.')

 
        obj.__class__ = XYZImageService
        
        if obj.requires_token():
            
            if access_token is None:
                raise Exception("Tile provider '%s' requires an access token, none provided." % obj.name)
            obj["accessToken"] = access_token   

        provider_dname = obj.name.replace('.', '_')
        if tiles_dpath is None: tiles_dpath = 'temp_xyz_tiles'

        obj._tiles_dpath = Path(tiles_dpath) / provider_dname
        obj._tiles_dpath.mkdir(parents=True, exist_ok=True)
        obj._zoom = None
    

    def _set_and_validate_zoom(self, zoom):
        
        getdefattr = lambda a, v: v if not hasattr(self, a) else getattr(self, a)
        max_zoom = getdefattr('max_zoom', 22)
        min_zoom = getdefattr('min_zoom', 0)

        if min_zoom < 0:
            raise Exception("Unexpected state")

        def wrapper(val, zoom):
            self._set_zoom(zoom)
            return val
        
        if zoom > max_zoom: return wrapper(1, max_zoom)
        if zoom < min_zoom: return wrapper(-1, min_zoom) 
        return wrapper(0, zoom)

    def _set_zoom(self, zoom):
        self._zoom = zoom
        self._n = n = 2**zoom
        self._tile_url = self.build_url().replace("{z}", "%d" % self._zoom)
        # NOTE: +1 in n+1 to correct the formula for n exactly a power of 10
        index_fmt = "%d" if n < 10  else "%%0%dd" % np.ceil(np.log10(int(n)+1))    
        self._fname_fmtter = "%02d_%s_%s.png" % (self._zoom, index_fmt, index_fmt)

    @property
    def zoom(self):
        return self._zoom
            
    def _get_tile_url(self, x, y):
        return self._tile_url.replace('{x}', "%d" % x).replace('{y}', "%d" % y) + ".png"

    def _get_local_tile_path(self, x, y):
        return self._tiles_dpath / (self._fname_fmtter % (x ,y))

    def _get_url_path_pair(self, x, y):
        url = self._get_tile_url(x, y)
        fpath = self._get_local_tile_path(x, y)
        return url, fpath

    def _get_path_info(self, x, y):
        url = self._get_tile_url(x, y)
        fpath = self._get_local_tile_path(x, y)
        return url, fpath, fpath.is_file()

    def _get_paths_info(self, x0, y0, x1, y1):
        xs, ys = (range(s0, s1+1) for s0, s1 in [(x0, x1), (y0, y1)])
        return [self._get_path_info(x,y) for y in ys for x in xs]

    def new_files_count(self, x0, y0, x1, y1):
        _ , n_new = self._get_file_info(*self._get_paths_info(self, x0, y0, x1, y1))
        return n_new 

    def file_count(self, x0, y0, x1, y1):
        return self._get_file_info(*self._get_paths_info(self, x0, y0, x1, y1))



    
    def get_image(self, x0, y0, x1, y1, zoom, mode='img'):
        check = self._set_and_validate_zoom(zoom)

        if not check == 0:
            raise NotImplementedError()

        img = self._get_image(x0, y0, x1, y1)
        ny, nx = img.shape
        bounds = (0, 0, nx, ny)
        
        return self._convert_type(img, bounds, mode)

    # Method to convert data from an image standard to a
    # mathematical grid standard 
    def _convert_type(self, img, bounds, mode):

        
        if mode == 'img': 
            return img, bounds

        x0, y0, x1, y1 = bounds
        img = np.flipud(img)
        ny, nx = img.shape
        x, y = fgrid.linspace2d(x0, x1, nx, y0, y1, ny, mode)

        # NOTE: Returning x, y as a tuple to somewhat maintain return signature
        return img, (x, y)

    def _get_image(self, x0, y0, x1, y1):

        info_files = self._get_paths_info(x0, y0, x1, y1)

        new_files = [x for *x, is_file in info_files if not is_file]
    
        if len(new_files) > 0: 
            for url, fpath in tqdm(new_files, desc="Downloading"):
                with open(fpath, 'wb') as handler:
                    handler.write(requests.get(url).content)        
            
        _size = lambda a, b: b - a + 1
        shp = (_size(*a) for a in [(y0, y1), (x0, x1)])

        fpaths = [fpath for _, fpath, _ in info_files]
        
        def load_image(fpath) :
            img = Image.open(fpath)
            img.load()
            return np.asarray(img, dtype="int32")

        sub_img = load_image(fpaths[0])

        n_loc, m_loc, n_channels = sub_img.shape
        n_sub, m_sub = shp
        shp_gbl = n_loc*n_sub, m_loc*m_sub, n_channels
        dtype = sub_img.dtype

        img = np.zeros(shp_gbl, dtype)

        _slices = lambda n, m: [slice(i*m, (i+1)*m) for i in range(n)]
        slices = [(si, sj) for sj in _slices(n_sub, n_loc) for si in _slices(m_sub, m_loc)]

        args = list(zip(fpaths, slices))

        _, (si, sj) = args[0]
        img[sj, si, :] = sub_img

        for fpath, (si, sj) in args[1:]:
            img[sj, si, :] = load_image(fpath)
            
        return img
        
# Derived class mapping Geographic coordinates to Slippy map indices ranges 
#  1) A lower suffix denotes the lower left corner
#  2) An upper suffix denotes the upper right corner
#
#   ▲  │           (xu, yu)      ▲  │ (xl, yu)  (xu, yu)
#   |  │                         |  │
#   y  │                         y  │
#   |  │ (xl, yl)                |  │ (xl, yl)  (xu, yl)  
#      └───────────────────         └───────────────────
#             ——— x ——►                   ——— x ——►
#  3) Indexing code derived from:
#       https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Derivation_of_tile_names
class GeoImageService(XYZImageService):

    @classmethod
    def create(cls, obj, pixels=256, **kwargs):
        XYZImageService.create(obj, **kwargs)
        obj.__class__ = GeoImageService
        obj._bounds = None
        obj._pixels = pixels 
        

    def _validate_resolution(self, resolution):
        
        zoom = int(np.ceil(np.log2(360/(self._pixels*resolution))))
        check = self._set_and_validate_zoom(zoom)
        if check == 0: return  

        new_res = 360/(self._pixels*2**self.zoom) 
        
        if check < 0: 
            msg = "Specified resolution, %f m, is smaller than provider '%s' has available, %e deg." % (resolution, self.name, new_res) 
            warnings.warn(msg)

        if check > 0: 
            msg = "Specified resolution, %f m, is greater than provider '%s' has available, %e deg." % (resolution, self.name, new_res) 
            warnings.warn(msg)

    # Wrapper method for separating input validation and postprocessing
    # from image generation 
    def get_image(self, lon_lower, lat_lower, lon_upper, lat_upper, resolution, mode='img'):
        self._validate_resolution(resolution)
        img, bounds = self._get_image(lon_lower, lat_lower, lon_upper, lat_upper)
        return self._convert_type(img, bounds, mode)
        
    def _get_image(self, lon_lower, lat_lower, lon_upper, lat_upper):
        
        # NOTE: Latitude swap is due to flipped orientation
        #       of y/vertical-axis of images vs. math convention 
        x0, y0 = self._geo2slippy(lon_lower, lat_upper)
        x1, y1 = self._geo2slippy(lon_upper, lat_lower)

        # print(x0, y0)
        # print(x1, y1)
        # x0, y0 = self._geo2slippy(lon_lower, lat_lower)
        # x1, y1 = self._geo2slippy(lon_upper, lat_upper)

        # print(x0, y0)
        # print(x1, y1)

        # NOTE: Same y swap
        lon_lower, lat_upper = self._slippy2geocorner(x0, y0, is_lower=False)
        lon_upper, lat_lower = self._slippy2geocorner(x1, y1, is_lower=True)

        geo_bounds = lon_lower, lat_lower, lon_upper, lat_upper

        img = super()._get_image(x0, y0, x1, y1)
       
        def pprint(x, y): print(y, x)



        return img, geo_bounds      

    def _geo2slippy(self, lon, lat):

        n = 2**self._zoom

        x = int(np.floor(n*(lon+180)/360))
        rlat = np.deg2rad(lat)
        y = np.log(np.tan(rlat) + 1/np.cos(rlat))/np.pi
        y = int(np.floor(n*(1-y)/2))

        # lon, lat = (np.deg2rad(s) for s in [lon, lat])
        # x = lon 
        # y = np.arcsinh(np.tan(lat))
    
        # x = (1+x/np.pi)/2
        # y = (1-y/np.pi)/2
    
        # scale = lambda x: int(np.floor(2**self.zoom*x))
        # x, y = (scale(s) for s in (x,y))
        return x, y
    
    # Funtion to get Geographic coordinates of the lower left corner 
    # or upper right corner of a Slippy Tile 



    def _slippy2geocorner(self, x, y, is_lower=False):

        if is_lower:
            x += 1
            y += 1
        
        n = 2**self.zoom
        lon = x / n * 360.0 - 180.0
        lat = np.rad2deg(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))

        return lon, lat 

   



    
        # x = np.pi*(2*x/scale - 1)
        # y = np.pi*(1 - 2*y/scale)
    
        # lon = x
        # lat = np.arctan(np.sinh(y))
    
        # ds = 360/scale
        # lon, lat = (np.rad2deg(s) for s in (lon ,lat))
    
        return lon, lat
        

# Derived wrapper class mapping an image with Geographic coordinates 
# to another coordinate system represented by a European Petroleum 
# Survey Group (EPSG) code, https://epsg.io/.
# NOTES:
#  Uses interpolation 
class PyProjGeoService(GeoImageService):

    @classmethod
    def create(cls, obj, epsg=None, **kwargs):
        GeoImageService.create(obj, **kwargs)
        obj.__class__ = PyProjGeoService
        obj._bounds = None
        crs = pyproj.CRS.from_epsg(epsg)
        obj._proj = pyproj.Proj(crs)
        


    # Method transforming/projecting the coordinates of a rectilinear rectangle 
    # into another coordinate system and extracting a bounding box that is rectilinear
    # in the transformed coordinates. 
    # The bounding box can either be:
    #  the inner box contained by the transformed coordinate space/region
    #  the outer box containing the transformed coordinate space/region
    #
    # NOTE: Method assumes the transformation is a homomorphic projection
    #
    # NOTATION:
    #   ▲  │ (x10, y10)  (x11, y11)   ▲  │ (xl, yu)  (xu, yu)
    #   |  │                          |  │
    #   y  │                          y  │   
    #   |  │ (x00, y00)  (x01, y01)   |  │ (xl, yl)  (xu, yl)
    #      └───────────────────          └───────────────────
    #           ——— x ——►                       ——— x ——►
    @classmethod
    def proj_bounds(cls, bounds, proj, is_inner=False):
        # Transforming corners of bounding box to new coordinate system
        x_lower, y_lower, x_upper, y_upper = bounds
        u_00, v_00 = proj(x_lower, y_lower)
        u_01, v_01 = proj(x_lower, y_upper)
        u_10, v_10 = proj(x_upper, y_lower)
        u_11, v_11 = proj(x_upper, y_upper)

        # Swapping min/max functions depending on whether the inner or outer box is specified 
        fs = (np.max, np.min)
        f_l, f_u = fs if is_inner else reversed(fs)

        # Extracting specified box coordinates from rhombus coordinates
        lower = [u_00, u_01], [v_00, v_10]
        upper = [u_10, u_11], [v_11, v_01]
        u_lower, v_lower = (f_l(s) for s in lower)
        u_upper, v_upper = (f_u(s) for s in upper)
        return u_lower, v_lower, u_upper, v_upper


    _EARTH_RADIUS_ = 40075016.686
    
    def _validate_resolution(self, resolution, lat_lower, lat_upper):

        
        scales = [self._EARTH_RADIUS_*np.abs(np.cos(np.deg2rad(a)))/self._pixels for a in [lat_lower, lat_upper]]

        res2zoom = lambda x: int(np.ceil(np.log2(x/resolution)))
        zoom = np.max([res2zoom(a) for a in scales])
        check = self._set_and_validate_zoom(zoom)

        zoom2res = lambda x: x/2**self.zoom
        r_min, r_max = sorted([zoom2res(a) for a in scales])
        
        if check == 0: return  
            
        if check < 0: 
            msg = "Specified resolution, %f m, is smaller than provider '%s' has available, %f ~ %f m." % (resolution, self.name, r_min, r_max) 
            warnings.warn(msg)

        if check > 0: 
            msg = "Specified resolution, %f m, is greater than provider '%s' has available, %f ~ %f m." % (resolution, self.name, r_min, r_max) 
            warnings.warn(msg)

    # Wrapper method for separating input validation and postprocessing
    # from image generation 
    def get_image(self, x_lower, y_lower, x_upper, y_upper, resolution, mode='img'):

        # Hacky repeated code for validation check 
        proj = lambda x, y: self._proj(x, y, inverse=True)
        bounds = x_lower, y_lower, x_upper, y_upper
        _, lat_lower, _, lat_upper = self.__class__.proj_bounds(bounds, proj)
        self._validate_resolution(resolution, lat_lower, lat_upper)
        
        img, bounds = self._get_image(x_lower, y_lower, x_upper, y_upper, resolution)
        return self._convert_type(img, bounds, mode)

    def _get_image(self, x_lower, y_lower, x_upper, y_upper, ds):

        bounds = x_lower, y_lower, x_upper, y_upper
        proj = lambda x, y: self._proj(x, y, inverse=True)
        bounds_geo = self.__class__.proj_bounds(bounds, proj)

        img, new_bounds_geo = super()._get_image(*bounds_geo)
        

        # Getting rectilinear bounding box in transformed coordinates contained in Geographic domain
        proj = lambda x, y: self._proj(x, y)
        new_bounds = self.__class__.proj_bounds(new_bounds_geo, proj, is_inner=True)
        new_bounds = bounds
        # Generating rectilinear grid in transformed coordinates than projecting to
        # Geographic coordinates for interpolation 

   
        x0, y0, x1, y1 = new_bounds

        #x0, y0, x1, y1 = x_lower, y_lower, x_upper, y_upper

        x, y = fgrid.nearest_linspace2d(x0, x1, ds, y0, y1, ds)
        (xx, yy), shp = fgrid.flat_meshgrid(x, y)
        lon_i, lat_i = self._proj(xx, yy, inverse=True)

        # Setting up Geographic interpolation coordinates of the image
        lon_0, lat_0, lon_1, lat_1 = new_bounds_geo
        ny, nx, n_channels = img.shape

    
        lons, lats = fgrid.linspace2d(lon_0, lon_1, nx, lat_0, lat_1, ny)

        lats = np.flipud(lats)
        img_new = np.zeros(shp + (n_channels,), dtype=img.dtype)
    
        for i in range(n_channels):
            interp = RegularGridInterpolator((lats, lons), img[:,:,i])
            img_new[:, :, i] = np.flipud(interp((lat_i, lon_i)).reshape(shp))

        return img_new, new_bounds