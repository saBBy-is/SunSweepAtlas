"""
step3_sweep.py — Complete, honest Sun-Sweep Atlas for SIH 2026 PPT.
════════════════════════════════════════════════════════════════════
Protocol is locked by config.yaml which was committed BEFORE this
script runs.  The simulator files (sun_sim_v2.py, synth_dem.py) are
verified against the SHA-256 hashes in SIMULATOR_HASHES.txt at
startup; any change causes an immediate exit.

Grid (from config.yaml, PPT slide 6):
  delta_azimuth : [0, 45, 90, 135, 180]  — relative to reference az=270
  absolute_az   : [270, 315, 0, 45, 90]
  elevation     : [10, 40]
  protocols     : raw, stretch_2_98
  factors       : A_identity, B_affine

For every cell this script also records:
  ncc    — normalised cross-correlation to the reference image (no matcher)
  This makes the 180°-flip NCC claim auditable from atlas.csv directly.

Output: results/atlas.csv — one row per (matcher, protocol, factor, az, el).
"""

import yaml
import numpy as np
import time
import cv2
import hashlib
import os
import sys
import warnings
from datetime import datetime, timezone
from PIL import Image

# ── Simulator integrity check ──────────────────────────────────────────────────
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

try:
    import vismatch
    _VISMATCH_OK = True
except ImportError:
    _VISMATCH_OK = False
    print("WARNING: vismatch not importable — only SIFT and ORB will run.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def ncc(a, b):
    """Normalised cross-correlation (Pearson) between two uint8 images."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def apply_affine(img, angle_deg, scale, tx, ty):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, scale)
    M[0, 2] += tx; M[1, 2] += ty
    warped = cv2.warpAffine(img, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, M


def transform_pts(pts, M):
    if len(pts) == 0:
        return pts
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    return (M @ pts_h.T).T


def evaluate(pts1, pts2, M_gt, max_err_px=3.0, min_correct=20):
    """Return (n_matches, n_correct, grid_err, success).

    success = (grid_err < max_err_px) AND (n_correct >= min_correct).
    grid_err is mean reprojection error over a 10x10 grid.
    """
    n = len(pts1) if pts1 is not None else 0
    if n < min_correct:
        return n, 0, float('inf'), False

    pts1_t = transform_pts(pts1, M_gt)
    errs = np.hypot(pts1_t[:, 0] - pts2[:, 0], pts1_t[:, 1] - pts2[:, 1])
    n_correct = int(np.sum(errs <= max_err_px))

    # 10x10 grid reprojection
    gy, gx = np.mgrid[0:512:51.2, 0:512:51.2]
    gpts = np.column_stack([gx.ravel(), gy.ravel()])
    M_est, _ = cv2.estimateAffine2D(
        pts1, pts2,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=max_err_px,
        maxIters=2000,
    )
    if M_est is None:
        return n, n_correct, float('inf'), False
    det = M_est[0, 0] * M_est[1, 1] - M_est[0, 1] * M_est[1, 0]
    if abs(det) < 0.1 or abs(det) > 10.0:
        return n, n_correct, float('inf'), False

    grid_gt  = transform_pts(gpts, M_gt)
    grid_est = transform_pts(gpts, M_est)
    grid_err = float(np.mean(np.hypot(
        grid_gt[:, 0] - grid_est[:, 0],
        grid_gt[:, 1] - grid_est[:, 1],
    )))
    success = (grid_err < max_err_px) and (n_correct >= min_correct)
    return n, n_correct, grid_err, success


def apply_protocol(img, proto):
    if proto == 'stretch_2_98':
        p2, p98 = np.percentile(img, (2, 98))
        if p98 > p2:
            img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    return img


def run_matcher(m_name, img1, img2, scratch):
    """Run a single matcher; return (pts1, pts2, runtime) or (None, error_str, 0)."""
    os.makedirs(scratch, exist_ok=True)
    p1 = os.path.join(scratch, 'm1.png')
    p2 = os.path.join(scratch, 'm2.png')
    Image.fromarray(img1).save(p1)
    Image.fromarray(img2).save(p2)
    t0 = time.time()
    try:
        if m_name == 'sift':
            det = cv2.SIFT_create()
            kp1, d1 = det.detectAndCompute(img1, None)
            kp2, d2 = det.detectAndCompute(img2, None)
            if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
                return np.array([]), np.array([]), time.time() - t0
            bf = cv2.BFMatcher()
            raw = bf.knnMatch(d1, d2, k=2)
            good = [m for m, n in raw if m.distance < 0.75 * n.distance]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]) if good else np.array([])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]) if good else np.array([])
            return pts1, pts2, time.time() - t0
        elif m_name == 'orb':
            det = cv2.ORB_create(nfeatures=10000)
            kp1, d1 = det.detectAndCompute(img1, None)
            kp2, d2 = det.detectAndCompute(img2, None)
            if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
                return np.array([]), np.array([]), time.time() - t0
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(d1, d2)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]) if matches else np.array([])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]) if matches else np.array([])
            return pts1, pts2, time.time() - t0
        else:
            if not _VISMATCH_OK:
                return None, "vismatch not available", 0.0
            matcher = vismatch.get_matcher(m_name, device='cpu')
            res = matcher(p1, p2)
            if isinstance(res, tuple):
                pts1, pts2 = res[:2]
            else:
                pts1, pts2 = res['matched_kpts0'], res['matched_kpts1']
            return pts1, pts2, time.time() - t0
    except Exception as e:
        return None, str(e), 0.0


# ── Main sweep ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _project = os.path.dirname(_sweep_dir)
    config_path = os.path.join(_project, 'config.yaml')

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    REF_AZ  = float(cfg['reference']['azimuth'])   # 270
    REF_EL  = float(cfg['reference']['elevation'])  # 40
    MAX_ERR = float(cfg['success']['max_grid_error_px'])    # 3.0
    MIN_COR = int(cfg['success']['min_correct_matches'])    # 20

    delta_azs = cfg['grid']['delta_azimuth']        # [0,45,90,135,180]
    elevations = cfg['grid']['elevation']           # [10, 40]
    protocols  = cfg['protocols']                   # ['raw','stretch_2_98']
    factors    = cfg['factors']                     # ['A_identity','B_affine']

    AFF_ROT   = float(cfg['affine_params']['rotation_deg'])
    AFF_SCALE = float(cfg['affine_params']['scale'])
    AFF_TX    = float(cfg['affine_params']['tx_px'])
    AFF_TY    = float(cfg['affine_params']['ty_px'])

    # Absolute azimuths: (REF_AZ + delta) mod 360
    abs_azs = [int((REF_AZ + d) % 360) for d in delta_azs]

    # Matchers to test — in deterministic order
    matchers = ['sift', 'orb', 'loftr', 'superpoint-lightglue', 'aliked-lightglue', 'minima']

    scratch = os.path.join(_project, 'data', 'scratch')
    results_dir = os.path.join(_project, 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'atlas.csv')

    # ── Delete any previous atlas — start fresh ──────────────────────────────
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Deleted previous {csv_path}")

    header = ("matcher,protocol,factor,delta_az,abs_az,el,"
              "matches,correct,grid_err,success,"
              "ncc_vs_ref,runtime,terrain_id,timestamp\n")
    with open(csv_path, 'w') as f:
        f.write(header)

    # ── Generate DEM and reference image (seed=42, 512px, 20m/px) ───────────
    print("Generating DEM (seed=42, 512x512, 20 m/px)...")
    dem = make_dem(512, 20)

    print(f"Rendering reference image (az={REF_AZ}°, el={REF_EL}°)...")
    h_ref = horizon_map(dem, 20.0, REF_AZ)
    ref_f = render(dem, 20.0, REF_AZ, REF_EL, horizon=h_ref)
    img_ref = to_uint8(ref_f)

    # Pre-compute affine warp matrix for B_affine (same for all cells)
    _, M_aff = apply_affine(img_ref, AFF_ROT, AFF_SCALE, AFF_TX, AFF_TY)
    M_identity = np.eye(2, 3)

    # ── Pre-cache horizons for all absolute azimuths ─────────────────────────
    print("Pre-computing horizon maps...")
    h_cache = {int(REF_AZ): h_ref}
    for abs_az in abs_azs:
        if abs_az not in h_cache:
            h_cache[abs_az] = horizon_map(dem, 20.0, abs_az)

    total = len(delta_azs) * len(elevations) * len(matchers) * len(protocols) * len(factors)
    cell_n = 0

    print(f"\nRunning PPT-spec sweep: {len(delta_azs)} Δaz × {len(elevations)} el × "
          f"{len(matchers)} matchers × {len(protocols)} protocols × {len(factors)} factors "
          f"= {total} cells\n")

    for di, (daz, abs_az) in enumerate(zip(delta_azs, abs_azs)):
        h = h_cache[abs_az]
        for el in elevations:
            # Render target image at this (az, el)
            tgt_f = render(dem, 20.0, abs_az, el, horizon=h)
            img_tgt = to_uint8(tgt_f)
            ncc_val = ncc(img_ref, img_tgt)

            for m in matchers:
                for proto in protocols:
                    ref_p = apply_protocol(img_ref.copy(), proto)
                    tgt_p = apply_protocol(img_tgt.copy(), proto)

                    for factor in factors:
                        cell_n += 1
                        if factor == 'A_identity':
                            i1, i2, M_gt = ref_p, tgt_p, M_identity.copy()
                        else:
                            i2_warp, _ = apply_affine(tgt_p, AFF_ROT, AFF_SCALE, AFF_TX, AFF_TY)
                            i1, i2, M_gt = ref_p, i2_warp, M_aff.copy()

                        pts1, pts2, rtime = run_matcher(m, i1, i2, scratch)
                        if pts1 is None:
                            # Error — record as 0 matches, failure
                            n_m, n_c, gerr, succ = 0, 0, float('inf'), False
                        else:
                            n_m, n_c, gerr, succ = evaluate(pts1, pts2, M_gt, MAX_ERR, MIN_COR)

                        gerr_str = f"{gerr:.4f}" if gerr != float('inf') else "inf"
                        ts = datetime.now(timezone.utc).isoformat()

                        row = (f"{m},{proto},{factor},{daz},{abs_az},{el},"
                               f"{n_m},{n_c},{gerr_str},{succ},"
                               f"{ncc_val:.6f},{rtime:.3f},synth01,{ts}\n")
                        with open(csv_path, 'a') as f:
                            f.write(row)

                        pct = 100 * cell_n / total
                        print(f"  [{cell_n:>3}/{total}] {pct:5.1f}%  "
                              f"Δaz={daz:>3}° abs={abs_az:>3}° el={el:>2}°  "
                              f"{m:<25} {proto:<15} {factor:<12}  "
                              f"m={n_m:>5} c={n_c:>5} err={gerr_str:>9} "
                              f"{'OK' if succ else 'FAIL'}  {rtime:.1f}s")

    print(f"\nSweep complete. {cell_n} rows written to {csv_path}")
    print(f"(Expected {total})")
