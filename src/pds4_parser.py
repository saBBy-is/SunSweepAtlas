import xml.etree.ElementTree as ET
import os
import glob
from datetime import datetime
import numpy as np

class ISROC2Parser:
    """
    Parser for ISRO ISSDC Chandrayaan-2 PDS4 Data Products.
    Extracts critical metadata (Sensor, Sun Angles, Resolution) from XML labels 
    to feed directly into the LunarAlign Trust Router.
    """
    
    # Common PDS4 namespaces used in ISRO data
    NAMESPACES = {
        'pds': 'http://pds.nasa.gov/pds4/pds/v1',
        'cart': 'http://pds.nasa.gov/pds4/cart/v1',
        'geom': 'http://pds.nasa.gov/pds4/geom/v1',
        'isro': 'http://pds.nasa.gov/pds4/isro/v1' # Hypothetical ISRO specific if applicable
    }

    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.metadata = {}
        self._parse()

    def _parse(self):
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML label not found: {self.xml_path}")
            
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        # Helper to find elements regardless of namespace prefix verbosity
        def find_text(xpath, default="Unknown"):
            try:
                el = root.find(xpath, self.NAMESPACES)
                return el.text if el is not None else default
            except Exception:
                return default

        # 1. Mission & Instrument Identification
        self.metadata['mission'] = find_text('.//pds:Investigation_Area/pds:name')
        self.metadata['instrument'] = find_text('.//pds:Observing_System_Component[pds:type="Instrument"]/pds:name')
        self.metadata['target'] = find_text('.//pds:Target_Identification/pds:name')
        self.metadata['timestamp'] = find_text('.//pds:Time_Coordinates/pds:start_date_time')
        
        # ISRO specific normalizations
        instr = self.metadata['instrument'].upper()
        if 'OHRC' in instr or 'ORBITER HIGH RESOLUTION' in instr:
            self.metadata['sensor_id'] = 'OHRC'
        elif 'TMC' in instr or 'TERRAIN MAPPING' in instr:
            self.metadata['sensor_id'] = 'TMC-2'
        elif 'IIRS' in instr or 'INFRARED' in instr:
            self.metadata['sensor_id'] = 'IIRS'
        else:
            self.metadata['sensor_id'] = self.metadata['instrument']

        # 2. Solar Geometry (Crucial for Trust Router)
        # Incidence Angle (Zenith angle of sun) -> Elevation = 90 - Incidence
        inc_str = find_text('.//geom:Geometry//geom:incidence_angle', None)
        if not inc_str:
            # Fallback for some ISRO early releases which put it in custom tags
            inc_str = find_text('.//pds:Solar_Longitude', None)
            
        if inc_str and inc_str != "Unknown":
            inc_angle = float(inc_str)
            self.metadata['sun_elevation'] = 90.0 - inc_angle
        else:
            self.metadata['sun_elevation'] = None # Needs manual estimation

        # Azimuth
        az_str = find_text('.//geom:Geometry//geom:azimuth_angle', None)
        if az_str and az_str != "Unknown":
            self.metadata['sun_azimuth'] = float(az_str)
        else:
            self.metadata['sun_azimuth'] = None

        # 3. Image Map Properties
        self.metadata['lines'] = find_text('.//pds:File_Area_Observational//pds:lines', "0")
        self.metadata['samples'] = find_text('.//pds:File_Area_Observational//pds:samples', "0")
        self.metadata['resolution_m'] = find_text('.//cart:Cartography//cart:pixel_resolution_x', None)
        
        # Fallback to standard specs if XML lacks resolution
        if not self.metadata['resolution_m']:
            if self.metadata['sensor_id'] == 'OHRC':
                self.metadata['resolution_m'] = 0.3
            elif self.metadata['sensor_id'] == 'TMC-2':
                self.metadata['resolution_m'] = 5.0
            elif self.metadata['sensor_id'] == 'IIRS':
                self.metadata['resolution_m'] = 80.0
                
    def get_trust_router_inputs(self):
        """
        Returns exactly what the Trust Router needs to make a decision.
        """
        return {
            'sensor': self.metadata.get('sensor_id'),
            'sun_el': self.metadata.get('sun_elevation'),
            'sun_az': self.metadata.get('sun_azimuth'),
            'gsd_m': float(self.metadata.get('resolution_m', 1.0))
        }

    def print_summary(self):
        print("="*50)
        print(" ISRO PDS4 DATA EXTRACT (CHANDRAYAAN-2)")
        print("="*50)
        print(f"Mission    : {self.metadata.get('mission')}")
        print(f"Instrument : {self.metadata.get('instrument')} ({self.metadata.get('sensor_id')})")
        print(f"Target     : {self.metadata.get('target')}")
        print(f"Date       : {self.metadata.get('timestamp')}")
        print("-"*50)
        print(f"Sun Elev   : {self.metadata.get('sun_elevation')} deg")
        print(f"Sun Azim   : {self.metadata.get('sun_azimuth')} deg")
        print(f"Resolution : {self.metadata.get('resolution_m')} m/px")
        print(f"Image Size : {self.metadata.get('samples')} x {self.metadata.get('lines')} px")
        print("="*50)

if __name__ == "__main__":
    # Test block
    print("ISRO PDS4 Parser Module Loaded.")
