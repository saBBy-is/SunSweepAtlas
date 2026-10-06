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


def parse_sensor_info(xml_path: str):
    """
    Parses a PDS4 XML metadata file and extracts sensor/instrument
    identification for multi-modal routing.

    Returns (sensor_name, gsd_m) or (None, None) on failure.
    Sensor name is normalised to one of: 'OHRC', 'TMC-2', 'IIRS'.
    """
    # Known ISRO instrument ID → canonical sensor name mapping
    _INSTRUMENT_MAP = {
        'ch2_ohrc': 'OHRC',
        'ohrc': 'OHRC',
        'orbiter high resolution camera': 'OHRC',
        'ch2_tmc2': 'TMC-2',
        'tmc2': 'TMC-2',
        'tmc-2': 'TMC-2',
        'terrain mapping camera': 'TMC-2',
        'ch2_iirs': 'IIRS',
        'iirs': 'IIRS',
        'imaging infrared spectrometer': 'IIRS',
    }
    _GSD_DEFAULTS = {
        'OHRC': 0.3,
        'TMC-2': 5.0,
        'IIRS': 80.0,
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        sensor_name = None
        gsd_m = None

        for elem in root.iter():
            tag = re.sub(r'^{.*}', '', elem.tag).lower()
            text = (elem.text or '').strip().lower()

            # Look for instrument/investigation identifiers
            if tag in ('name', 'instrument_name', 'instrument_id',
                       'instrument_host_name'):
                if text in _INSTRUMENT_MAP:
                    sensor_name = _INSTRUMENT_MAP[text]

            # Look for GSD / spatial resolution
            if tag in ('pixel_resolution_spatial', 'map_scale',
                       'pixel_scale', 'ground_sample_distance'):
                try:
                    gsd_m = float(text)
                except ValueError:
                    pass

        # Fall back to default GSD if sensor was identified but GSD not found
        if sensor_name and gsd_m is None:
            gsd_m = _GSD_DEFAULTS.get(sensor_name)

        return sensor_name, gsd_m

    except Exception as e:
        print(f"Error parsing sensor info from {xml_path}: {e}")
        return None, None


def parse_full_metadata(xml_path: str):
    """
    Parse all relevant metadata from a PDS4 XML label.
    Returns a dict with keys: sun_azimuth, sun_elevation, sensor, gsd_m.
    """
    az, el = parse_sun_angles(xml_path)
    sensor, gsd = parse_sensor_info(xml_path)
    return {
        'sun_azimuth': az,
        'sun_elevation': el,
        'sensor': sensor,
        'gsd_m': gsd,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_xml>")
        sys.exit(1)
        
    xml_path = sys.argv[1]

    # Sun angles
    azimuth, elevation = parse_sun_angles(xml_path)
    print(f"File: {xml_path}")
    print(f"Extracted Sun Azimuth: {azimuth} degrees")
    print(f"Extracted Sun Elevation: {elevation} degrees")

    # Sensor info
    sensor, gsd = parse_sensor_info(xml_path)
    print(f"Extracted Sensor: {sensor}")
    print(f"Extracted GSD: {gsd} m/px")

