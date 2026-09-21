import numpy as np
import cv2
import logging
import json

logger = logging.getLogger(__name__)

class ImageLoader:
    def __init__(self, image_path: str, metadata_path: str = None):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.image = None
        self.metadata = {}
        
    def load(self):
        """Loads an ISRO image (OHRC/TMC-2)."""
        # Note: In a real scenario with PDS4, we would use pds4_tools.
        # Here we mock it with cv2 for simplicity if it's exported as TIF/PNG.
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            logger.warning(f"Could not read {self.image_path}. Returning dummy image.")
            self.image = np.zeros((1024, 1024), dtype=np.uint8)
        
        self.load_metadata()
        return self.image
        
    def load_metadata(self):
        """Extracts metadata like sun elevation and azimuth angles."""
        # Mocking metadata extraction. In reality, parse PDS4 XML label.
        if self.metadata_path:
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Metadata file {self.metadata_path} not found.")
        
        # Default mock values if not found
        if "sun_elevation" not in self.metadata:
            self.metadata["sun_elevation"] = 45.0
        if "sun_azimuth" not in self.metadata:
            self.metadata["sun_azimuth"] = 90.0
            
    def get_sun_angles(self):
        return self.metadata.get("sun_elevation", 45.0), self.metadata.get("sun_azimuth", 90.0)
