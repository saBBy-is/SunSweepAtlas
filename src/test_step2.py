from osgeo import gdal, osr
import numpy as np
import os

def create_synth(filename, lat):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(filename, 100, 100, 1, gdal.GDT_UInt16)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    
    y_start = lat + (100 / 2) * 0.001
    ds.SetGeoTransform((0, 0.001, 0, y_start, 0, -0.001))
    
    band = ds.GetRasterBand(1)
    band.SetScale(0.5)
    band.SetOffset(0.0)
    data = np.random.randint(10000, 12000, (100, 100), dtype=np.uint16)
    band.WriteArray(data)
    band.FlushCache()
    ds = None
    
if __name__ == '__main__':
    create_synth('data/scratch/synth_lat0.tif', 0.0)
    create_synth('data/scratch/synth_lat60.tif', 60.0)
