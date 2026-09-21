import numpy as np
try:
    from osgeo import gdal
except ImportError:
    gdal = None
import logging

logger = logging.getLogger(__name__)

class DEMLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = None
        self.elevation_data = None
        
    def load(self):
        """Loads the DEM GeoTIFF using GDAL."""
        if gdal is None:
            logger.warning("GDAL is not installed. Returning dummy DEM data.")
            self.elevation_data = np.zeros((1024, 1024), dtype=np.float32)
            return self.elevation_data
            
        self.dataset = gdal.Open(self.filepath)
        if not self.dataset:
            raise FileNotFoundError(f"Could not open DEM at {self.filepath}")
            
        band = self.dataset.GetRasterBand(1)
        self.elevation_data = band.ReadAsArray()
        logger.info(f"Loaded DEM of shape {self.elevation_data.shape}")
        return self.elevation_data

    def get_metadata(self):
        """Extracts geotransform and projection."""
        if not self.dataset:
            return {}
        return {
            "geotransform": self.dataset.GetGeoTransform(),
            "projection": self.dataset.GetProjection()
        }
