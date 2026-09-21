import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from PIL import Image
import os

def generate_hillshade(dem_array: np.ndarray, sun_azimuth: float, sun_elevation: float, vert_exag: float = 1.0) -> np.ndarray:
    """
    Generates a synthetic hillshade image from a Digital Elevation Model (DEM)
    given specific sun angles.

    Args:
        dem_array: A 2D numpy array representing the elevation map.
        sun_azimuth: Sun azimuth angle in degrees (0-360, clockwise from North).
        sun_elevation: Sun elevation (altitude) angle in degrees (0-90).
        vert_exag: Vertical exaggeration factor to enhance terrain features.

    Returns:
        A 2D numpy array representing the shaded image with values 0-255.
    """
    # Matplotlib's LightSource computes the hillshade
    ls = LightSource(azdeg=sun_azimuth, altdeg=sun_elevation)
    
    # Generate hillshade (values between 0 and 1)
    # The hillshade algorithm calculates the illumination based on slope and aspect.
    shaded = ls.hillshade(dem_array, vert_exag=vert_exag)
    
    # Scale to 0-255 for standard image format (uint8)
    synthetic_image = (shaded * 255).astype(np.uint8)
    
    return synthetic_image

if __name__ == "__main__":
    # Test the simulator with a dummy DEM (a simple gaussian bump)
    print("Testing sun_sim.py...")
    x, y = np.mgrid[-50:50:500j, -50:50:500j]
    # Create a crater-like or bump-like shape for the dummy DEM
    dem = 100 * np.exp(-(x**2 + y**2) / 400.0)
    
    azimuth = 45.0
    elevation = 30.0
    
    synthetic_lunar_photo = generate_hillshade(dem, sun_azimuth=azimuth, sun_elevation=elevation)
    
    # Save the output to verify
    output_dir = "data/scratch"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_test.png")
    
    Image.fromarray(synthetic_lunar_photo).save(output_path)
    
    print(f"Generated synthetic image with Azimuth={azimuth}, Elevation={elevation}")
    print(f"Saved test output to {output_path}")
