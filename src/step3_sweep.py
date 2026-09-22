import yaml
import numpy as np
import time
import cv2
import hashlib
from PIL import Image
import os
import sys
import vismatch
import warnings
from datetime import datetime, timezone

# ── Simulator integrity check ─────────────────────────────────────
_sweep_dir = os.path.dirname(os.path.abspath(__file__))
_EXPECTED_HASHES = {
    os.path.join(_sweep_dir, "sun_sim_v2.py"): "eb05f657168a53c99b8be71ee9c634d67eadcb82fb2a9da824afd36530b9a79b",
    os.path.join(_sweep_dir, "synth_dem.py"):   "841ba178ac8b1d1833ac9cb121083cd1446f895e5f0b6b96280e291b97dc1a9a",
}
for _path, _expected in _EXPECTED_HASHES.items():
    with open(_path, "rb") as _fh:
        _actual = hashlib.sha256(_fh.read()).hexdigest()
    if _actual != _expected:
        print(f"INTEGRITY ERROR: {os.path.basename(_path)} has been modified!")
        print(f"  Expected SHA-256: {_expected}")
        print(f"  Actual   SHA-256: {_actual}")
        print("Refusing to run. Restore the original file or update SIMULATOR_HASHES.txt.")
        sys.exit(1)

from sun_sim_v2 import horizon_map, render, to_uint8
from synth_dem import make_dem

warnings.filterwarnings("ignore")

def apply_affine_img(img, angle_deg, scale, tx, ty):
    h, w = img.shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, M

def transform_pts(pts, M):
    if len(pts) == 0: return pts
    pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ pts_homo.T).T

def evaluate_matches(pts1, pts2, M_gt):
    if len(pts1) == 0:
        return 0, 0, False, float('inf')
    pts1_trans = transform_pts(pts1, M_gt)
    errs = np.hypot(pts1_trans[:, 0] - pts2[:, 0], pts1_trans[:, 1] - pts2[:, 1])
    correct = np.sum(errs <= 3.0)
    
    mean_err = float('inf')
    success = False
    if len(pts1) >= 3:
        M_est, _ = cv2.estimateAffine2D(pts1, pts2, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0)
        if M_est is not None:
            grid_y, grid_x = np.mgrid[0:512:51.2, 0:512:51.2]
            grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            grid_trans_gt = transform_pts(grid_pts, M_gt)
            grid_trans_est = transform_pts(grid_pts, M_est)
            errs_grid = np.hypot(grid_trans_gt[:, 0] - grid_trans_est[:, 0], grid_trans_gt[:, 1] - grid_trans_est[:, 1])
            mean_err = np.mean(errs_grid)
            success = (mean_err < 3.0) and (correct >= 20)
    return len(pts1), correct, success, mean_err

def run_matcher(m_name, img1, img2):
    _proj = os.path.dirname(_sweep_dir)
    scratch = os.path.join(_proj, 'data', 'scratch')
    os.makedirs(scratch, exist_ok=True)
    p1, p2 = os.path.join(scratch, 'm1.png'), os.path.join(scratch, 'm2.png')
    Image.fromarray(img1).save(p1)
    Image.fromarray(img2).save(p2)
    start = time.time()
    try:
        if m_name in ['sift', 'orb']:
            if m_name == 'sift':
                detector = cv2.SIFT_create()
            elif m_name == 'orb':
                detector = cv2.ORB_create(nfeatures=10000)
                
            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)
            pts1, pts2 = [], []
            if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
                if m_name == 'orb':
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    good = matches
                else:
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)
                    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                
                if good:
                    pts1 = np.float32([kp1[m.queryIdx].pt if hasattr(m, 'queryIdx') else kp1[m[0].queryIdx].pt for m in good])
                    pts2 = np.float32([kp2[m.trainIdx].pt if hasattr(m, 'trainIdx') else kp2[m[0].trainIdx].pt for m in good])
            pts1, pts2 = np.array(pts1), np.array(pts2)
        else:
            matcher = vismatch.get_matcher(m_name, device='cpu')
            res = matcher(p1, p2)
            if isinstance(res, tuple):
                pts1, pts2 = res[:2]
            else:
                pts1, pts2 = res['matched_kpts0'], res['matched_kpts1']
        return pts1, pts2, time.time() - start
    except Exception as e:
        return None, str(e), 0.0

def apply_protocol(img, proto):
    if proto == 'stretch_2_98':
        p2, p98 = np.percentile(img, (2, 98))
        if p98 > p2:
            img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    return img

if __name__ == '__main__':
    _project_root = os.path.dirname(_sweep_dir)  # parent of src/
    config_path = os.path.join(_project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dem = make_dem(512, 20)
    
    # Precompute horizons for all azimuths in the grid
    azs = config['grid']['azimuth']
    els = config['grid']['elevation']
    
    # Generate reference image
    h_ref = horizon_map(dem, 20.0, 270.0)
    img_ref_f = render(dem, 20.0, 270.0, 40.0, horizon=h_ref)
    img_ref = to_uint8(img_ref_f)
    
    angle, scale, tx, ty = 2.0, 1.1, 5.0, -3.0
    
    passed_matchers = ['sift', 'orb', 'loftr', 'superpoint-lightglue', 'aliked-lightglue', 'minima']
    protos = config['protocols']
    factors = ['A_identity', 'B_affine']
    
    results_dir = os.path.join(_project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, 'atlas.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)
    with open(csv_file, 'w') as f:
        f.write("matcher,protocol,factor,az,el,matches,correct,grid_err,success,runtime,terrain_id,timestamp\n")
            
    print("\nRunning full synthetic sweep (all matchers)...")
    total_cells = len(azs) * len(els) * len(passed_matchers) * len(protos) * len(factors)
    cell_num = 0
    for az in azs:
        h_az = horizon_map(dem, 20.0, az)
        for el in els:
            img_target_f = render(dem, 20.0, az, el, horizon=h_az)
            img_target = to_uint8(img_target_f)
            
            for m in passed_matchers:
                for proto in protos:
                    img_ref_p = apply_protocol(img_ref, proto)
                    img_target_p = apply_protocol(img_target, proto)
                    
                    for factor in factors:
                        cell_num += 1
                        if factor == 'A_identity':
                            i1, i2, M_gt = img_ref_p, img_target_p, np.eye(2, 3)
                        else:
                            img_t_warp, M_w = apply_affine_img(img_target_p, angle, scale, tx, ty)
                            i1, i2, M_gt = img_ref_p, img_t_warp, M_w
                            
                        # single repeat for speed in this synthetic sweep
                        pts1, pts2, rtime = run_matcher(m, i1, i2)
                        if pts1 is None:
                            continue
                            
                        tot, corr, succ, merr = evaluate_matches(pts1, pts2, M_gt)
                        ts = datetime.now(timezone.utc).isoformat()
                        
                        with open(csv_file, 'a') as f:
                            f.write(f"{m},{proto},{factor},{az},{el},{tot},{corr},{merr:.3f},{succ},{rtime:.3f},synth01,{ts}\n")
                        
                        if cell_num % 10 == 0 or cell_num == total_cells:
                            print(f"  [{cell_num}/{total_cells}] {m} | az={az} el={el} | {proto} | {factor} | corr={corr}/{tot} | {succ}")
                            
    print(f"\nFull synthetic sweep complete! {cell_num} cells written to {csv_file}")
