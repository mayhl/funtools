import numpy as np
from abc import ABC, abstractmethod

def _get_range_slice(s, s0, s1):
    
    i0 = np.argmin(np.abs(s-s0))
    if s[i0] > s0: i0-=1

    i1 = np.argmin(np.abs(s-s1))
    if s[i1] > s1: i1-=1
        
        
    if i0 < 0: i0 = 0
    nm = len(s)-1
    if i1 > nm: i1 = nm
        
    return slice(int(i0), int(i1)+1)


class _Structure(ABC):
    
    def __init__(self, x0=None, y0=None, height_sealevel=None, target_volume=None):
        
        if x0 is None: raise ValueError('x0 not specified')
        if y0 is None: raise ValueError('y0 not specified')
                
        is_height_sealevel = not height_sealevel is None
        is_target_volume = not target_volume is None

        if is_height_sealevel == is_target_volume:
            if is_height_sealevel:
                raise ValueError("Can not specify both height_sealevel and target_volume.")
            else:
                raise ValueError("Must specify either height_sealevel or target_volume.")
        

        if is_height_sealevel:
            self._height_sl = height_sealevel
            self._is_height_mode = True
        else:
            self._trg_vol = target_volume
            self._is_height_mode = False
            
        self._x0 = x0
        self._y0 = y0
        

    def construct_structure_subgrid(self, *args, **kwargs):
        
        if self._is_height_mode:
            return self._construct_structure_subgrid_height(*args, **kwargs)
        else:
            return self._construct_structure_subgrid_volume(*args, **kwargs)
        
    
    def _construct_structure_subgrid_volume(self, x, y, z, tolerance = 10**-3, max_iterations = 50):
        
         # Initial estimate of depth
        i0 = np.argmin(np.abs(x-self._x0))
        j0 = np.argmin(np.abs(y-self._y0))
        z_min = z[j0, i0]
    
    
        dx = np.mean(np.diff(x))
        dy = np.mean(np.diff(y))

        # Using iterative method and bounding box search to refine
        # structure's total height and footprint
        
        self._height_sl = z_min + 1 
        dz_numeric = 0.2

        for i in range(max_iterations):
        
            sx, sy = self._bounding_box(x, y, z, z_min)
            z_new = z[sy, sx].min()
            
            self._height_sl += z_new-z_min
            v = self._estimate_volume(z_new)
            self._height_sl += dz_numeric
            vp = self._estimate_volume(z_new)
            self._height_sl -= 2*dz_numeric
            vm = self._estimate_volume(z_new)
            self._height_sl += dz_numeric
   
            dfdz = (vp-vm)/(2*dz_numeric) 
            f = v - self._trg_vol
            dz = -f/dfdz
            self._height_sl += dz
            z_min = z_new
      
            if np.abs(dz) < tolerance: break
            
    
        
         # Using interative and exact check to refine 
        # structure's total height and footprint 
        
        def compute_volume_diff(z_min):
            sx, sy = self._bounding_box(x, y, z, z_min)
            subx, suby = np.meshgrid(x[sx], y[sy])
            subz = z[sy, sx]
            sub_shape = subz.shape
            subx, suby, subz = (s.flatten() for s in[subx, suby, subz])
            subu, subv = self._transform_2_local_coords(subx, suby)
        
        
            new_subz = self._update_subgrid(subu, subv, subz, z_min, sub_shape)
            
            n, m = sub_shape
            return (np.sum(new_subz-z[sy, sx])*dx*dy-self._trg_vol)/(n*m)
        
        

    
        dv0 = compute_volume_diff(z_min)
        

        if dv0 < 0:
            
            dv1 = dv0
            dh = 0.05
            while dv1 < 0:
                dv0 = dv1
                self._height_sl += dh
                dv1 = compute_volume_diff(z_min)
   
            h1 = self._height_sl
            h0 = h1 - dh
    
            while np.abs(h1-h0)>tolerance:
            
                hc = (h1 + h0)/2
                self._height_sl = hc
                dvc = compute_volume_diff(z_min)           
                
                if dvc < 0:
                    h0 = hc
                else:
                    h1 = hc
                
                
            sx, sy = self._bounding_box(x, y, z, z_min)
            subx, suby = np.meshgrid(x[sx], y[sy])
            subz = z[sy, sx]
            sub_shape = subz.shape
            subx, suby, subz = (s.flatten() for s in[subx, suby, subz])
            subu, subv = self._transform_2_local_coords(subx, suby)
  
            new_subz = self._update_subgrid(subu, subv, subz, z_min, sub_shape)
            
        else:
            raise NotImplementedError()
        
        
        return sx, sy, new_subz             
        
        
        
    def _construct_structure_subgrid_height(self, x, y, z, tolerance = 10**-3, max_iterations = 10):
        
         # Initial estimate of depth
        i0 = np.argmin(np.abs(x-self._x0))
        j0 = np.argmin(np.abs(y-self._y0))
        z_min = z[j0, i0]
        
        # Using iterative method and bounding box search to refine
        # structure's total height and footprint
        for i in range(max_iterations):
        
            sx, sy = self._bounding_box(x, y, z, z_min)
            z_new = z[sy, sx].min()
            dz = np.abs(z_new-z_min)
            z_min = z_new
            if dz < tolerance: break
            
            
        # Extracting sub-grid and flattening
        sx, sy = self._bounding_box(x, y, z, z_min)
        subx, suby = np.meshgrid(x[sx], y[sy])
        subz = z[sy, sx]
        sub_shape = subz.shape
        subx, suby, subz = (s.flatten() for s in[subx, suby, subz])
        subu, subv = self._transform_2_local_coords(subx, suby)
        
        # Using interative and exact check to refine 
        # structure's total height and footprint 
        for i in range(max_iterations):
        
            filt = self._local_filter_footprint(subu, subv, subz, z_min)
            z_new = subz[filt].min()
            dz = np.abs(z_new-z_min)
            z_min = z_new
        
            if dz < tolerance: break     
                
        new_subz = self._update_subgrid(subu, subv, subz, z_min, z_shape)
        
        return sx, sy, new_subz       
    
    def _update_subgrid(self, subu, subv, subz, z_min, z_shape):
        
        # Creating structure
        filt = self._local_filter_footprint(subu, subv, subz, z_min)
        structure = self._construct_structure_local(subu[filt], subv[filt], z_min)  
        
        # Keeping original grid point values if higher/larger than structure
        new_subz = subz.copy()   
        filt2 = new_subz[filt] > structure   
        structure[filt2] = new_subz[filt][filt2]  
        
        # Creating new sub-grid up structure
        new_subz[filt] = structure
        new_subz = new_subz.reshape(z_shape)   
        return new_subz
        
   
    def update_grid(self, x, y, z, tolerance = 10**-3, max_iterations = 60):
        
        sx, sy, new_subz = self.construct_structure_subgrid(x, y, z, tolerance, max_iterations)
        new_z = z.copy()
        new_z[sy, sx] = new_subz
        return new_z
        
    def _bounding_box(self, x, y, z, z_min):
      
        u0, u1, v0, v1 = self._local_bounding_box(x, y, z, z_min)
        
        u = np.array([u0, u0, u1, u1])
        v = np.array([v0, v1, v1, v0])
        
        xb, yb = self._transform_2_global_coords(u, v)
            
        x0, x1 = xb.min(), xb.max()
        y0, y1 = yb.min(), yb.max()

        sx = _get_range_slice(x, x0, x1)
        sy = _get_range_slice(y, y0, y1)
        
        return sx, sy      
    
    # Transformation between global and local grid
    def _transform_2_local_coords(self, x, y):
        u = x - self._x0
        v = y - self._y0
        return u, v
        
    def _transform_2_global_coords(self, u, v):
        x = u + self._x0
        y = v + self._y0
        return x, y
    
    # Compute bounding box of structure in local grid based on estimate min z value
    @abstractmethod
    def _local_bounding_box(self, x, y ,z, z_min):
        pass
    
    # Exact check for local grid points in structure footprint based on estimate z value
    @abstractmethod
    def _local_filter_footprint(self, u, v ,z, z_min):
        pass
    
    # Function to define structure profile
    @abstractmethod
    def _construct_structure_local(self, u, v, z_min):
        pass
    
    
        # Function to define structure profile
    @abstractmethod
    def _estimate_volume(self, z_min):
        pass
    
    
# Stucture with a defined direction/orientation/rotation
# Overloads transformation methods of parent class

class _DirectionalStructure(_Structure):
    
    def __init__(self, rotation=None, **kwargs):
        if rotation is None: raise ValueError('rotation not specified')
        self._rotation = rotation
        super().__init__(**kwargs)
        
    def _transform_2_local_coords(self, x, y):
        
        s = x - self._x0
        t = y - self._y0
        
        cos = np.cos(-self._rotation)
        sin = np.sin(-self._rotation)
        
        u =  s*cos + t*sin
        v = -s*sin + t*cos
        
        return u, v
    
    def _transform_2_global_coords(self, u, v):
             
        cos = np.cos(self._rotation)
        sin = np.sin(self._rotation)
        
        x =  u*cos + v*sin + self._x0
        y = -u*sin + v*cos + self._y0
        return x, y

class _Prism(_DirectionalStructure):
    
    def __init__(self, length=None, **kwargs):
        if length is None: raise ValueError('length not specified')
        self._length = length 
        super().__init__(**kwargs)
     
    # Partially defining function
    def _local_bounding_box(self, x, y ,z, z_min):
        
        v0 = -self._length/2
        v1 =  self._length/2        
        u0, u1 = self._local_crosssection_box(x, y, z, z_min)
        
        return u0, u1, v0, v1
    
    
    def _local_filter_footprint(self, u, v ,z, z_min):
        filt_u = self._local_filter_crosssection(u, v, z, z_min)
        
        v0 = -self._length/2
        v1 =  self._length/2  
        filt_v = (v0 <= v) & (v <= v1)
        return filt_u & filt_v
    
    
    def _construct_structure_local(self, u, v, z_min):    
        return self._construct_crossection_local(u, z_min)
       
    def _estimate_volume(self, z_min):
        return self._length*self._estimate_crosssection_area(z_min)
    
    
    # Range of cross-section defined is derived class
    @abstractmethod
    def _local_crosssection_box(self, x, y, z, z_min):
        pass
        # Range of cross-section defined is derived class
        
    @abstractmethod
    def _local_filter_crosssection(self, u, v, z, z_min):
        pass
        # Range of cross-section defined is derived class
        
    @abstractmethod
    def _construct_crossection_local(self, u, z_min):
        pass
    
    @abstractmethod
    def _estimate_crosssection_area(self, z_min):
        pass
    
    
class RightAnglePrism(_Prism):

    def __init__(self, slope = None, **kwargs):
        
        if slope is None: raise ValueError('slope not specified')
        self._slope = slope
        super().__init__(**kwargs)
        
    def _get_width(self, z_min):
        dz = self._height_sl - z_min
        return dz/np.tan(self._slope)  
        
    def _local_crosssection_box(self, x, y, z, z_min):
        du = self._get_width(z_min)         
        return -du/2, du/2  
    
    def _local_filter_crosssection(self, u, v, z, z_min):
        u0, u1 = self._local_crosssection_box(u, v, z, z_min)
        return (u0 <= u) & (u <= u1)

    def _construct_crossection_local(self, u, z_min):
        u0 = -self._get_width(z_min)/2
        return np.tan(self._slope)*(u-u0) + z_min

    def _estimate_crosssection_area(self, z_min):
        b = self._get_width(z_min)
        h = self._height_sl - z_min
        return 0.5*b*h
    
        
class RectangularPrism(_Prism):
    
    def __init__(self, width = None, **kwargs):
        if width is None: raise ValueError('width not specified')
        self._width = width
        super().__init__(**kwargs)      
         
    def _local_crosssection_box(self, x, y, z, z_min):
        return -self._width/2, self._width/2   
    
    def _local_filter_crosssection(self, u, v, z, z_min):
        u0, u1 = self._local_crosssection_box(u, v, z, z_min)
        return (u0 <= u) & (u <= u1)
    
    def _construct_crossection_local(self, u, z_min): 
        return np.ones(np.shape(u))*self._height_sl
    
        
class _Corner(_DirectionalStructure):
    
    # Partially defining function
    def _local_bounding_box(self, x, y ,z, z_min):
        u0, u1 = self._local_crosssection_u(x, y, z, z_min)
        v0, v1 = self._local_crosssection_v(x, y, z, z_min)
        return u0, u1, v0, v1
    
    def _local_filter_footprint(self, u, v ,z, z_min):
        filt_u = self._local_filter_footprint_u(u, v, z, z_min)
        filt_v = self._local_filter_footprint_v(u, v, z, z_min)
        return filt_u & filt_v
      
    # Range of cross-section in local u direction
    @abstractmethod
    def _local_crosssection_u(self, x, y, z, z_min):
        pass
    
    # Range of cross-section in local v direction
    @abstractmethod
    def _local_crosssection_v(self, x, y, z, z_min):
        pass
    
    @abstractmethod
    def _local_filter_footprint_u(self, u, v, z, z_min):
        pass

    @abstractmethod
    def _local_filter_footprint_v(self, u, v, z, z_min):
        pass
    
class SlopedCorner(_Corner):
    
    def __init__(self, slope_x=None, slope_y=None, **kwargs):
        
        if slope_x is None: raise ValueError('slope_x not specified')
        if slope_y is None: raise ValueError('slope_y not specified')
        self.slope_y = slope_y
        self.slope_x = slope_x
        super().__init__(**kwargs)         
        
    def _local_filter_footprint_u(self, u, v, z, z_min):
        dz = self._height_sl - z_min
        du = dz/np.tan(self._slope_x)  
        u0, u1 = -du/2, du/2
        return (u0 <= u) & (u <= u1)   
    
    def _local_filter_footprint_u(self, u, v, z, z_min):
        dz = self._height_sl - z_min
        dv = dz/np.tan(self._slope_y)  
        v0, v1 = -dv/2, dv/2
        return (v0 <= v) & (v <= v1)