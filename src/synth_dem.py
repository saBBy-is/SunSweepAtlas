import numpy as np
import scipy.ndimage as ndi

def make_dem(size=512, resolution=20):
    np.random.seed(42)
    dem = np.random.normal(0, 50, (size, size))
    dem = ndi.gaussian_filter(dem, sigma=15) * 5
    dem += ndi.gaussian_filter(np.random.normal(0, 30, (size, size)), sigma=4) * 3
    
    y, x = np.mgrid[0:size, 0:size]
    for _ in range(15):
        cx, cy = np.random.randint(0, size, 2)
        r = np.random.randint(15, 60)
        depth = np.random.uniform(80, 200)
        
        dist = np.hypot(x - cx, y - cy)
        in_crater = dist < r
        dem[in_crater] -= depth * (1 - (dist[in_crater]/r)**2)

        rim = (dist >= r) & (dist < r * 1.2)
        dem[rim] += depth * 0.5 * (1 - (dist[rim] - r)/(r * 0.2))
        
    return dem
