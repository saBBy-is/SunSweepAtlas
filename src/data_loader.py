import xml.etree.ElementTree as ET
import sys
import re

def parse_sun_angles(xml_path: str):
    """
    Parses a PDS4 XML metadata file (like those from ISRO Chandrayaan-2)
    and extracts the Sun Azimuth and Sun Elevation angles.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        sun_azimuth = None
        sun_elevation = None
        
        # In PDS4, elements usually have namespaces (e.g., {http://pds.nasa.gov/pds4/geom/v1}sun_azimuth_angle)
        # We can iterate through all elements and match by local tag name to avoid strict namespace dependency.
        for elem in root.iter():
            tag_name = re.sub(r'^{.*}', '', elem.tag).lower()
            if tag_name == 'sun_azimuth_angle':
                sun_azimuth = float(elem.text)
            elif tag_name == 'sun_elevation_angle':
                sun_elevation = float(elem.text)
                
        return sun_azimuth, sun_elevation
        
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None, None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_xml>")
        sys.exit(1)
        
    xml_path = sys.argv[1]
    azimuth, elevation = parse_sun_angles(xml_path)
    
    print(f"File: {xml_path}")
    print(f"Extracted Sun Azimuth: {azimuth} degrees")
    print(f"Extracted Sun Elevation: {elevation} degrees")
