import pandas as pd
import cv2
import numpy as np
import time
import os
import vismatch
from PIL import Image

def load_atlas(atlas_path="results/atlas.csv"):
    if not os.path.exists(atlas_path):
        return None
    return pd.read_csv(atlas_path)

def get_best_matcher(az1, el1, az2, el2, atlas_df):
    """
    Decide the best matcher based on illumination difference.
    If the difference is large (hard pair), use loftr. 
    Otherwise use superpoint-lightglue or sift.
    """
    diff_el = abs(el1 - el2)
    diff_az = min(abs(az1 - az2), 360 - abs(az1 - az2))
    
    # Simple heuristic rules based on atlas data:
    if diff_el > 20 or diff_az > 45:
        # Hard illumination changes: LoFTR handles large changes better
        return "loftr", "raw"
    else:
        # Similar illumination: superpoint-lightglue or sift are very accurate and fast
        return "superpoint-lightglue", "raw"

def negative_control_check(pts1, pts2):
    """
    Robust check to reject false matches (Negative Control).
    Requires at least 20 matches, and the RANSAC fit must not be completely degenerate.
    """
    if len(pts1) < 20:
        return False, None, "Insufficient matches (<20)"
    
    M_est, inliers = cv2.estimateAffine2D(pts1, pts2, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0, maxIters=2000)
    
    if M_est is None:
        return False, None, "RANSAC failed to find a valid transform"
        
    inlier_count = np.sum(inliers)
    if inlier_count < 15:
        return False, M_est, f"Too few inliers ({inlier_count} < 15)"
        
    det = M_est[0,0]*M_est[1,1] - M_est[0,1]*M_est[1,0]
    if abs(det) < 0.1 or abs(det) > 10.0:
        return False, M_est, f"Degenerate transform (det={det:.3f})"
        
    return True, M_est, f"Success ({inlier_count} inliers)"

def run_matcher_pipeline(img1, img2, matcher_name):
    # Save temporary files for vismatch
    os.makedirs('data/scratch', exist_ok=True)
    p1, p2 = 'data/scratch/route_1.png', 'data/scratch/route_2.png'
    Image.fromarray(img1).save(p1)
    Image.fromarray(img2).save(p2)
    
    start = time.time()
    
    if matcher_name == 'sift':
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        pts1, pts2 = [], []
        good = []
        if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if good:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        pts1, pts2 = np.array(pts1), np.array(pts2)
    else:
        matcher = vismatch.get_matcher(matcher_name, device='cpu')
        res = matcher(p1, p2)
        if isinstance(res, tuple):
            pts1, pts2 = res[:2]
        else:
            pts1, pts2 = res['matched_kpts0'], res['matched_kpts1']
            
    runtime = time.time() - start
    
    is_valid, M_est, reason = negative_control_check(pts1, pts2)
    
    return {
        "matcher": matcher_name,
        "runtime": runtime,
        "raw_matches": len(pts1),
        "is_valid": is_valid,
        "transform": M_est,
        "reason": reason
    }
