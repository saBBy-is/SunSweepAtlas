import numpy as np
import time
from sun_sim_v2 import horizon_map, render, to_uint8
from synth_dem import make_dem

if __name__ == '__main__':
    dem = make_dem(512, 20)
    
    h270 = horizon_map(dem, 20.0, 270.0)
    h90 = horizon_map(dem, 20.0, 90.0)
    
    img1_f = render(dem, 20.0, 270.0, 40.0, horizon=h270)
    img2_f = render(dem, 20.0, 90.0, 10.0, horizon=h90)
    
    frac1 = float((h270 >= np.radians(40.0)).mean())
    frac2 = float((h90 >= np.radians(10.0)).mean())
    
    print("\n--- STEP 1 OUTPUT ---")
    print(f"Shadow fraction view 1 (az=270, el=40): {frac1:.3f}")
    print(f"Shadow fraction view 2 (az=90, el=10): {frac2:.3f}")
