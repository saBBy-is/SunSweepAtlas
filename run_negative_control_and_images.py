"""
run_negative_control_and_images.py — Executes:
1. Explicit wrong-location negative control for MINIMA and SIFT.
2. Verified PPT image generation from current committed code.
"""

import sys, os, json, hashlib
import numpy as np
import scipy.ndimage as ndi
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Hash check
EXPECTED_HASHES = {
    os.path.join(PROJECT_ROOT, "src", "sun_sim_v2.py"): "eb05f657168a53c99b8be71ee9c634d67eadcb82fb2a9da824afd36530b9a79b",
    os.path.join(PROJECT_ROOT, "src", "synth_dem.py"):   "841ba178ac8b1d1833ac9cb121083cd1446f895e5f0b6b96280e291b97dc1a9a",
}
for path, exp in EXPECTED_HASHES.items():
    with open(path, "rb") as fh:
        act = hashlib.sha256(fh.read()).hexdigest()
    assert act == exp, f"Integrity check failed for {os.path.basename(path)}!"

from sun_sim_v2 import horizon_map, render, to_uint8
from synth_dem import make_dem
import vismatch
from step3_sweep import run_matcher, evaluate
from router import negative_control_check

def make_dem_seed(seed=9999, size=512):
    """Generates synthetic crater DEM with a different random seed to simulate a different terrain location."""
    np.random.seed(seed)
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

def main():
    scratch = os.path.join(PROJECT_ROOT, 'data', 'scratch')
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)

    # 1. Reference terrain (seed=42)
    dem_ref = make_dem(512, 20)
    h_ref = horizon_map(dem_ref, 20.0, 270.0)
    img_ref = to_uint8(render(dem_ref, 20.0, 270.0, 40.0, horizon=h_ref))

    # 2. Negative control: wrong location (seed=9999), same illumination (az=270, el=40)
    dem_neg = make_dem_seed(seed=9999, size=512)
    h_neg = horizon_map(dem_neg, 20.0, 270.0)
    img_neg = to_uint8(render(dem_neg, 20.0, 270.0, 40.0, horizon=h_neg))

    M_identity = np.eye(2, 3)
    MAX_ERR = 3.0
    MIN_COR = 20

    print("=================================================================")
    print("NEGATIVE CONTROL: Reference (seed=42) vs Different Terrain (seed=9999)")
    print("Illumination: az=270°, el=40° for both")
    print("=================================================================")

    neg_results = {}
    for m_name in ['minima', 'sift']:
        pts1, pts2, rtime = run_matcher(m_name, img_ref, img_neg, scratch)
        n_m = len(pts1) if pts1 is not None else 0

        # Evaluate against identity transform (which is completely false)
        n_m_eval, n_c_eval, gerr_eval, succ_eval = evaluate(pts1, pts2, M_identity, MAX_ERR, MIN_COR)

        # RANSAC check via router.negative_control_check
        is_valid_ransac, M_est, reason_ransac = negative_control_check(pts1, pts2, reproj_thresh=MAX_ERR, min_inliers=15)

        # Inlier count if M_est exists
        inlier_count = 0
        if M_est is not None and pts1 is not None and len(pts1) > 0:
            pts1_t = (M_est @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T
            errs = np.hypot(pts1_t[:, 0] - pts2[:, 0], pts1_t[:, 1] - pts2[:, 1])
            inlier_count = int(np.sum(errs <= MAX_ERR))

        # Check router verdict
        # Under the deck's success criterion: success = False
        # Under RANSAC check: is_valid = False or True?
        neg_results[m_name] = {
            "matcher": m_name,
            "raw_matches": n_m,
            "correct_matches_vs_identity": n_c_eval,
            "grid_error_vs_identity": str(gerr_eval),
            "success_criterion_met": succ_eval,
            "ransac_valid": is_valid_ransac,
            "ransac_inliers": inlier_count,
            "ransac_reason": reason_ransac,
            "produced_red_refusal": (not succ_eval) or (not is_valid_ransac),
            "runtime_s": round(rtime, 3)
        }

        print(f"\nMatcher: {m_name}")
        print(f"  Raw matches found         : {n_m}")
        print(f"  Correct matches (vs Ident): {n_c_eval}")
        print(f"  Grid error (vs Identity)  : {gerr_eval}")
        print(f"  Success Criterion Met     : {succ_eval} (Expected False)")
        print(f"  RANSAC Valid Transform    : {is_valid_ransac}")
        print(f"  RANSAC Inliers            : {inlier_count}")
        print(f"  RANSAC Reason / Diagnostic: {reason_ransac}")
        print(f"  Verdict Produced Refusal  : {neg_results[m_name]['produced_red_refusal']}")

    with open(os.path.join(results_dir, "negative_control_report.json"), "w") as f:
        json.dump(neg_results, f, indent=2)

    # 3. Export verified PPT images from committed code
    print("\n--- Exporting Verified PPT Images from Current Committed Code ---")
    h_cache = {270: h_ref}
    for a in [90]:
        h_cache[a] = horizon_map(dem_ref, 20.0, a)

    img_ref_fresh = img_ref
    img_flip_fresh = to_uint8(render(dem_ref, 20.0, 90.0, 40.0, horizon=h_cache[90]))
    img_grz10_fresh = to_uint8(render(dem_ref, 20.0, 270.0, 10.0, horizon=h_ref))
    img_grf10_fresh = to_uint8(render(dem_ref, 20.0, 90.0, 10.0, horizon=h_cache[90]))

    # Grazing at 5° (from demo.py)
    h_grz5 = horizon_map(dem_ref, 20.0, 270.0)
    h_grf5 = horizon_map(dem_ref, 20.0, 90.0)
    img_grz05_fresh = to_uint8(render(dem_ref, 20.0, 270.0, 5.0, horizon=h_grz5))
    img_grf05_fresh = to_uint8(render(dem_ref, 20.0, 90.0, 5.0, horizon=h_grf5))

    images_to_save = {
        "ppt_render_reference_az270_el40.png": img_ref_fresh,
        "ppt_render_180flip_az090_el40.png": img_flip_fresh,
        "ppt_render_grazing_az270_el10.png": img_grz10_fresh,
        "ppt_render_grazing_flip_az090_el10.png": img_grf10_fresh,
        "ppt_render_grazing_az270_el05.png": img_grz05_fresh,
        "ppt_render_grazing_flip_az090_el05.png": img_grf05_fresh,
    }
    for name, img in images_to_save.items():
        out_p = os.path.join(results_dir, name)
        cv2.imwrite(out_p, img)
        print(f"  Wrote {name} ({img.shape[1]}x{img.shape[0]} px, 20 m/px)")

    print("\nCompleted negative control and image export.")

if __name__ == '__main__':
    main()
