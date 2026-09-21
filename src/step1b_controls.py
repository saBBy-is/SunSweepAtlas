import numpy as np
import time
import cv2
from PIL import Image
import os
import vismatch
import warnings
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
    if len(pts1) == 0: return 0, 0, 0.0, False, float('inf')
    pts1_trans = transform_pts(pts1, M_gt)
    errs = np.hypot(pts1_trans[:, 0] - pts2[:, 0], pts1_trans[:, 1] - pts2[:, 1])
    correct = np.sum(errs <= 3.0)
    precision = correct / len(pts1) if len(pts1) > 0 else 0.0
    
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
    return len(pts1), correct, precision, success, mean_err

def run_matcher(matcher_name, img1, img2, M_gt):
    os.makedirs('data/scratch', exist_ok=True)
    img1_path = 'data/scratch/tmp1.png'
    img2_path = 'data/scratch/tmp2.png'
    Image.fromarray(img1).save(img1_path)
    Image.fromarray(img2).save(img2_path)
    
    start = time.time()
    pts1, pts2 = np.array([]), np.array([])
    
    try:
        if matcher_name.lower() == 'sift':
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            
            if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                if good:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        else:
            matcher = vismatch.get_matcher(matcher_name, device='cpu')
            res = matcher(img1_path, img2_path)
            if isinstance(res, tuple):
                pts1, pts2 = res[:2]
            else:
                pts1, pts2 = res['matched_kpts0'], res['matched_kpts1']
    except Exception as e:
        print(f"Error {matcher_name}: {e}")
        
    runtime = time.time() - start
    tot, corr, prec, succ, merr = evaluate_matches(pts1, pts2, M_gt)
    return tot, corr, prec, runtime, succ, merr

if __name__ == '__main__':
    dem = make_dem(512, 20)
    
    h270 = horizon_map(dem, 20.0, 270.0)
    h90 = horizon_map(dem, 20.0, 90.0)
    
    img_ref = to_uint8(render(dem, 20.0, 270.0, 40.0, horizon=h270))
    img_hard = to_uint8(render(dem, 20.0, 90.0, 10.0, horizon=h90))
    
    angle, scale, tx, ty = 2.0, 1.1, 5.0, -3.0
    img_ref_warp, M_warp = apply_affine_img(img_ref, angle, scale, tx, ty)
    img_hard_warp, _ = apply_affine_img(img_hard, angle, scale, tx, ty)
    
    M_identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cases = [
        ("Same Illum, Id", img_ref, img_ref, M_identity),
        ("Same Illum, Affine", img_ref, img_ref_warp, M_warp),
        ("Hard Pair, Id", img_ref, img_hard, M_identity),
        ("Hard Pair, Affine", img_ref, img_hard_warp, M_warp)
    ]
    matchers = ['sift', 'loftr']
    
    print("\n--- STEP 1b CONTROLS ---")
    print(f"{'Case':<20} | {'Matcher':<8} | {'Tot':<5} | {'Corr':<5} | {'Prec':<5} | {'Time(s)':<7} | {'Succ(Err)':<12}")
    print("-" * 80)
    for case_name, i1, i2, M_gt in cases:
        for m_name in matchers:
            tot, corr, prec, rtime, succ, merr = run_matcher(m_name, i1, i2, M_gt)
            succ_str = f"YES({merr:.1f})" if succ else f"NO({merr:.1f})"
            print(f"{case_name:<20} | {m_name:<8} | {tot:<5} | {corr:<5} | {prec:.3f} | {rtime:<7.2f} | {succ_str:<12}")
