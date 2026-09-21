import yaml
import numpy as np
import cv2
import time
from PIL import Image
import vismatch
import scipy.ndimage as ndi
import warnings
import os

warnings.filterwarnings("ignore")

from sun_sim_v2 import horizon_map, render, to_uint8
from synth_dem import make_dem

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

def evaluate_matches_robust(pts1, pts2, M_gt, maxIters=2000):
    if len(pts1) < 20:
        return len(pts1), 0, float('nan'), False, "degenerate (<20 matches)", None, (0,0), 0, 0, 0, False

    pts1_trans = transform_pts(pts1, M_gt)
    errs_gt = np.hypot(pts1_trans[:, 0] - pts2[:, 0], pts1_trans[:, 1] - pts2[:, 1])
    correct = np.sum(errs_gt <= 3.0)
    
    frac_1 = np.sum(errs_gt <= 1.0) / len(pts1)
    frac_3 = np.sum(errs_gt <= 3.0) / len(pts1)
    frac_10 = np.sum(errs_gt <= 10.0) / len(pts1)
    median_err = np.median(errs_gt)
    
    dx = np.median(pts2[:, 0] - pts1_trans[:, 0])
    dy = np.median(pts2[:, 1] - pts1_trans[:, 1])
    
    # RANSAC fit
    M_est, inliers = cv2.estimateAffine2D(pts1, pts2, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0, maxIters=maxIters)
    
    grid_y, grid_x = np.mgrid[0:512:51.2, 0:512:51.2]
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_err = float('nan')
    success = False
    reason = ""
    
    if M_est is None:
        reason = "degenerate (RANSAC failed)"
    else:
        # Check degeneracy of M_est
        det = M_est[0,0]*M_est[1,1] - M_est[0,1]*M_est[1,0]
        if abs(det) < 0.1 or abs(det) > 10.0:
            reason = f"degenerate (det={det:.3f})"
        else:
            grid_trans_gt = transform_pts(grid_pts, M_gt)
            grid_trans_est = transform_pts(grid_pts, M_est)
            errs_grid = np.hypot(grid_trans_gt[:, 0] - grid_trans_est[:, 0], grid_trans_gt[:, 1] - grid_trans_est[:, 1])
            grid_err = np.mean(errs_grid)
            success = (grid_err < 3.0) and (correct >= 20)
            if not success:
                reason = f"grid_err={grid_err:.1f}"

    # Oracle fit on GT-correct matches
    oracle_success = False
    correct_idx = np.where(errs_gt <= 3.0)[0]
    if len(correct_idx) >= 3:
        pts1_corr = pts1[correct_idx]
        pts2_corr = pts2[correct_idx]
        M_oracle, _ = cv2.estimateAffine2D(pts1_corr, pts2_corr, method=cv2.LMEDS)
        if M_oracle is not None:
            det_oracle = M_oracle[0,0]*M_oracle[1,1] - M_oracle[0,1]*M_oracle[1,0]
            if 0.1 <= abs(det_oracle) <= 10.0:
                grid_trans_gt = transform_pts(grid_pts, M_gt)
                grid_trans_est = transform_pts(grid_pts, M_oracle)
                errs_grid_oracle = np.hypot(grid_trans_gt[:, 0] - grid_trans_est[:, 0], grid_trans_gt[:, 1] - grid_trans_est[:, 1])
                oracle_err = np.mean(errs_grid_oracle)
                oracle_success = oracle_err < 3.0

    return len(pts1), correct, grid_err, success, reason, M_est, (dx, dy), frac_1, frac_3, frac_10, oracle_success

def get_pts(m_name, img1, img2):
    os.makedirs('data/scratch', exist_ok=True)
    p1, p2 = 'data/scratch/m1.png', 'data/scratch/m2.png'
    Image.fromarray(img1).save(p1)
    Image.fromarray(img2).save(p2)
    
    if m_name == 'sift':
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        pts1, pts2 = [], []
        if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if good:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return np.array(pts1), np.array(pts2)
    else:
        matcher = vismatch.get_matcher(m_name, device='cpu')
        res = matcher(p1, p2)
        if isinstance(res, tuple):
            return res[0], res[1]
        else:
            return res['matched_kpts0'], res['matched_kpts1']

if __name__ == '__main__':
    print("\n--- TASK 2: Heavy Matchers on Same-Illum Affine (448px) ---")
    dem = make_dem(512, 20)
    h270 = horizon_map(dem, 20.0, 270.0)
    img_ref_f = render(dem, 20.0, 270.0, 40.0, horizon=h270)
    img_ref = to_uint8(img_ref_f)
    angle, scale, tx, ty = 2.0, 1.1, 5.0, -3.0
    img_ref_warp, M_warp = apply_affine_img(img_ref, angle, scale, tx, ty)

    img_ref_448 = cv2.resize(img_ref, (448, 448))
    img_warp_448 = cv2.resize(img_ref_warp, (448, 448))
    M_scale = np.array([[448/512, 0, 0], [0, 448/512, 0]])
    # Adjust ground truth M for 448px
    # pts_448 = pts_512 * 448/512
    # M_warp_448 * pts_448 = M_warp * pts_512 * 448/512
    M_warp_448 = M_warp.copy()
    M_warp_448[0,2] *= (448/512)
    M_warp_448[1,2] *= (448/512)

    heavy = ['minima', 'roma', 'xoftr']
    for m in heavy:
        print(f"\nVariant for '{m}': {m} (default in vismatch)")
        try:
            t0 = time.time()
            pts1, pts2 = get_pts(m, img_ref_448, img_warp_448)
            t1 = time.time()
            pts1_c, pts2_c = get_pts(m, img_ref_448, img_warp_448)
            t2 = time.time()
            
            tot, corr, gerr, succ, reason, _, _, _, _, _, _ = evaluate_matches_robust(pts1, pts2, M_warp_448)
            print(f"{m:<10} | Corr: {corr}/{tot} | Err: {gerr:.2f} | Succ: {succ} | Time1: {t1-t0:.2f}s | Time2: {t2-t1:.2f}s")
        except Exception as e:
            print(f"{m:<10} | ERROR: {e}")

    print("\n--- TASK 3: Reference Cell (270, 40) rows ---")
    if os.path.exists('results/atlas.csv'):
        with open('results/atlas.csv') as f:
            lines = f.readlines()
            print(lines[0].strip())
            for l in lines[1:]:
                parts = l.strip().split(',')
                if len(parts) > 4 and parts[3] == '270' and parts[4] == '40':
                    print(l.strip())

    print("\n--- TASK 4: SIFT Hard Cell Matrix Diagnostics ---")
    h90 = horizon_map(dem, 20.0, 90.0)
    img_hard_f = render(dem, 20.0, 90.0, 20.0, horizon=h90)
    img_hard = to_uint8(img_hard_f)
    M_identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Raw
    pts1, pts2 = get_pts('sift', img_ref, img_hard)
    tot, corr, gerr, succ, reason, M_est, _, _, _, _, _ = evaluate_matches_robust(pts1, pts2, M_identity)
    print(f"SIFT RAW (90,20) | Matches: {tot} | Corr: {corr} | Err: {gerr} | Reason: {reason}")
    print(f"M_est:\n{M_est}")

    # Stretch
    p2, p98 = np.percentile(img_ref, (2, 98))
    img_ref_str = np.clip((img_ref - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    p2, p98 = np.percentile(img_hard, (2, 98))
    img_hard_str = np.clip((img_hard - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

    pts1_s, pts2_s = get_pts('sift', img_ref_str, img_hard_str)
    tot, corr, gerr, succ, reason, M_est_s, _, _, _, _, _ = evaluate_matches_robust(pts1_s, pts2_s, M_identity)
    print(f"SIFT STR (90,20) | Matches: {tot} | Corr: {corr} | Err: {gerr} | Reason: {reason}")
    print(f"M_est:\n{M_est_s}")


    print("\n--- TASK 5 & 6: LoFTR Hard Cells (90, 20), (90, 40), (90, 80) ---")
    print("matcher,az,el,dx,dy,frac_1,frac_3,frac_10,oracle_success,grid_err,reason")
    els = [20, 40, 80]
    for el in els:
        img_h_f = render(dem, 20.0, 90.0, el, horizon=h90)
        img_h = to_uint8(img_h_f)
        pts1, pts2 = get_pts('loftr', img_ref, img_h)
        tot, corr, gerr, succ, reason, M_est, (dx, dy), f1, f3, f10, orac = evaluate_matches_robust(pts1, pts2, M_identity, maxIters=20000)
        print(f"loftr,90,{el},{dx:.2f},{dy:.2f},{f1:.2f},{f3:.2f},{f10:.2f},{orac},{gerr:.2f},{reason}")

    print("\n--- TASK 7: Protocol shadow_masked ---")
    def apply_shadow_masked(img_uint8, el_deg, horizon):
        img = img_uint8.copy()
        shadow = horizon >= np.radians(el_deg)
        dilated = ndi.binary_dilation(shadow, iterations=3)
        lit_vals = img[~dilated]
        med = np.median(lit_vals) if len(lit_vals) > 0 else 127
        img[dilated] = int(med)
        return img

    print(f"{'Matcher':<8} | {'el':<3} | {'Matches':<8} | {'Correct':<8} | {'Err':<6} | {'Reason'}")
    for el in els:
        img_h_f = render(dem, 20.0, 90.0, el, horizon=h90)
        img_h = to_uint8(img_h_f)
        
        img_ref_sm = apply_shadow_masked(img_ref, 40.0, h270)
        img_h_sm = apply_shadow_masked(img_h, el, h90)
        
        for m in ['sift', 'loftr']:
            pts1, pts2 = get_pts(m, img_ref_sm, img_h_sm)
            tot, corr, gerr, succ, reason, M_est, _, _, _, _, _ = evaluate_matches_robust(pts1, pts2, M_identity)
            print(f"{m:<8} | {el:<3} | {tot:<8} | {corr:<8} | {gerr:<6.2f} | {reason}")

