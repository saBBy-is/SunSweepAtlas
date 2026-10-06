import numpy as np
import cv2
import logging
import json
import os
import sys

logger = logging.getLogger(__name__)

# Try to import sensor profiles (for GSD defaults)
try:
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from sensors import SENSORS, SensorProfile, get_sensor
    _SENSORS_AVAILABLE = True
except ImportError:
    _SENSORS_AVAILABLE = False


class ImageLoader:
    def __init__(self, image_path: str, metadata_path: str = None,
                 sensor_type: str = None):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.sensor_type = sensor_type   # e.g. "OHRC", "TMC-2", "IIRS"
        self.image = None
        self.metadata = {}
        self.sensor_profile = None

        # Look up sensor profile if available
        if sensor_type and _SENSORS_AVAILABLE:
            try:
                self.sensor_profile = get_sensor(sensor_type)
            except KeyError:
                logger.warning(f"Unknown sensor type: {sensor_type}")
        
    def load(self):
        """Loads an ISRO image (OHRC/TMC-2/IIRS)."""
        # Note: In a real scenario with PDS4, we would use pds4_tools.
        # Here we mock it with cv2 for simplicity if it's exported as TIF/PNG.
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            logger.warning(f"Could not read {self.image_path}. Returning dummy image.")
            self.image = np.zeros((1024, 1024), dtype=np.uint8)
        
        self.load_metadata()
        return self.image
        
    def load_metadata(self):
        """Extracts metadata like sun elevation, azimuth, sensor, and GSD."""
        if self.metadata_path:
            try:
                if self.metadata_path.lower().endswith(".xml"):
                    # Process real ISRO PDS4 XML Labels
                    try:
                        from pds4_parser import ISROC2Parser
                    except ImportError:
                        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from pds4_parser import ISROC2Parser
                        
                    parser = ISROC2Parser(self.metadata_path)
                    tr_inputs = parser.get_trust_router_inputs()
                    
                    self.metadata["sun_elevation"] = tr_inputs['sun_el']
                    self.metadata["sun_azimuth"] = tr_inputs['sun_az']
                    self.metadata["sensor"] = tr_inputs['sensor']
                    self.metadata["gsd_m"] = tr_inputs['gsd_m']
                    
                    logger.info(f"Loaded real ISRO PDS4 metadata: {self.metadata['sensor']} (Elev: {self.metadata['sun_elevation']})")
                else:
                    # Fallback for old JSON mock metadata
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse metadata file {self.metadata_path}: {e}")
        
        # Default mock values if not found or parsing failed
        if self.metadata.get("sun_elevation") is None:
            self.metadata["sun_elevation"] = 45.0
        if self.metadata.get("sun_azimuth") is None:
            self.metadata["sun_azimuth"] = 90.0

        # Set sensor info from metadata or constructor arg
        if "sensor" not in self.metadata and self.sensor_type:
            self.metadata["sensor"] = self.sensor_type
        if "gsd_m" not in self.metadata and self.sensor_profile:
            self.metadata["gsd_m"] = self.sensor_profile.gsd_m
            
    def get_sun_angles(self):
        """Return (sun_elevation, sun_azimuth) in degrees."""
        return (self.metadata.get("sun_elevation", 45.0),
                self.metadata.get("sun_azimuth", 90.0))

    def get_sensor_info(self):
        """Return (sensor_name, gsd_m) for multi-modal routing."""
        sensor = self.metadata.get("sensor", self.sensor_type)
        gsd = self.metadata.get("gsd_m", None)
        if gsd is None and self.sensor_profile:
            gsd = self.sensor_profile.gsd_m
        return sensor, gsd

    def get_gsd(self) -> float:
        """Return ground sample distance in metres/pixel."""
        _, gsd = self.get_sensor_info()
        return gsd if gsd is not None else 20.0  # default synthetic

