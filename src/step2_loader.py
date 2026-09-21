import sys
import numpy as np
import math
import scipy.ndimage as ndi
from osgeo import gdal, osr

def resample_to_square(data, phi_center):
    cos_phi = math.cos(math.radians(phi_center))
    # Scale X axis by cos(phi)
    resampled_data = ndi.zoom(data, (1.0, cos_phi), order=1)
    return resampled_data

def load_dem_window(filepath, col_off=0, row_off=0, width=100, height=100):
    ds = gdal.Open(filepath)
    if not ds:
        print(f"Error: Cannot open {filepath}")
        return None, None
        
    band = ds.GetRasterBand(1)
    scale = band.GetScale() or 1.0
    offset = band.GetOffset() or 0.0
    
    data = band.ReadAsArray(col_off, row_off, width, height)
    elevation = data.astype(np.float64) * scale + offset
    
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=proj)
    
    print(f"\nProcessing: {filepath}")
    print(f"Geotransform: {gt}")
    
    center_y = gt[3] + (row_off + height/2) * gt[5]
    
    if srs.IsGeographic():
        R = 1737400.0
        lat_rad = math.radians(center_y)
        dy_m = abs(gt[5]) * (math.pi / 180.0) * R
        dx_m = abs(gt[1]) * (math.pi / 180.0) * R * math.cos(lat_rad)
        px_x, px_y = dx_m, dy_m
        
        expected_y = 30.32
        expected_x = expected_y * math.cos(lat_rad)
        print(f"Expected Y: ~{expected_y:.2f} m vs Actual Y: {dy_m:.2f} m")
        print(f"Expected X: ~{expected_x:.2f} m vs Actual X: {dx_m:.2f} m")
        
        assert math.isclose(dy_m, 30.3233, rel_tol=1e-3), f"Y size {dy_m} != 30.32"
        assert math.isclose(dx_m, expected_x, rel_tol=1e-3), f"X size {dx_m} != {expected_x}"
        assert px_x <= px_y, f"X size {px_x} > Y size {px_y}, which is impossible for geographic CRS on sphere."
        
        resampled_data = resample_to_square(elevation, center_y)
        print(f"Original array shape: {elevation.shape}, Resampled shape (square metric pixels): {resampled_data.shape}")
        
    else:
        px_x = abs(gt[1])
        px_y = abs(gt[5])
        
    print(f"Pixel Size: {px_x:.2f} m (X), {px_y:.2f} m (Y)")
    return elevation, px_x, px_y

if __name__ == '__main__':
    if len(sys.argv) > 1:
        load_dem_window(sys.argv[1])
    else:
        print("Usage: python step2_loader.py <filepath>")
