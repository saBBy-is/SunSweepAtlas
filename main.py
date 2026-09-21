import logging
import numpy as np
from src.data.dem_loader import DEMLoader
from src.data.image_loader import ImageLoader
from src.simulation.shading import Simulator
from src.pipeline.router import TrustRouter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing LunarAlign Pipeline...")
    
    # 1. Initialize Loaders
    # (In a real scenario, paths would point to actual GeoTIFF/PDS4 data)
    dem_loader = DEMLoader("data/raw/lola_dem.tif")
    img_loader = ImageLoader("data/raw/ohrc_image.tif")
    
    # Generate dummy data for demonstration since files don't exist
    dem_data = dem_loader.load() 
    ohrc_image = img_loader.load()
    sun_elevation, sun_azimuth = img_loader.get_sun_angles()
    
    logger.info(f"Loaded image with Sun Elevation: {sun_elevation}°, Azimuth: {sun_azimuth}°")
    
    # 2. Shading Simulation (Optional step used mostly for generating the Atlas)
    simulator = Simulator(dem_loader)
    synthetic_image = simulator.generate_synthetic_image(sun_azimuth, sun_elevation)
    
    # 3. Trust Router & Inference
    # The router decides the best matcher based on the sun elevation
    router = TrustRouter()
    
    # We try to match the real OHRC image against the synthetic shaded DEM
    # (Or in practice, OHRC vs TMC-2 under different lighting)
    logger.info("Routing image correspondence request...")
    match_result, trust_level = router.route_and_match(ohrc_image, synthetic_image, sun_elevation)
    
    logger.info(f"--- Pipeline Result ---")
    logger.info(f"Trust Light: {trust_level}")
    
    if match_result:
        logger.info(f"Selected Matcher: {match_result['matcher']}")
        logger.info(f"Inliers found: {match_result['inliers']}")
    else:
        logger.warning("Matching aborted due to extreme lighting conditions (RED trust light).")

if __name__ == "__main__":
    main()
