"""
router.py — Atlas-driven Trust Router for LunarAlign (SIH 2026)
═══════════════════════════════════════════════════════════════
Reads results/atlas.csv and selects the best matcher + trust level
for a given (az, el) illumination pair using data-driven rules.

Trust Levels
────────────
  GREEN  — atlas shows ≥85% success in this lighting regime → match
  AMBER  — atlas shows 40–85% success → match with caveat
  RED    — atlas shows <40% success   → refuse to match

Router cascade (atlas-trained):
  1. minima        (overall 93% success, best across all elevations)
  2. superpoint-lightglue  (fallback when minima not available)
  3. sift          (fast control, always last resort)
"""

import os
import csv
import math
import time
import cv2
import numpy as np
from collections import defaultdict

try:
    from PIL import Image
    import vismatch
    _VISMATCH_AVAILABLE = True
except ImportError:
    _VISMATCH_AVAILABLE = False


# ── Atlas loader ──────────────────────────────────────────────────────────────

def load_atlas(atlas_path="results/atlas.csv"):
    """Load the sweep atlas CSV into a nested success-rate dict.

    Returns
    -------
    dict  { matcher -> { (az_bin, el_bin) -> success_rate } }
          where az_bin / el_bin are the nearest grid points.
    """
    if not os.path.exists(atlas_path):
        return None

    # Accumulate successes
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [successes, total]
    with open(atlas_path, newline="") as f:
        for row in csv.DictReader(f):
            m = row["matcher"]
            az = int(row["az"])
            el = int(row["el"])
            acc[m][(az, el)][1] += 1
            if row["success"] == "True":
                acc[m][(az, el)][0] += 1

    # Convert to rates
    atlas = {}
    for m, cells in acc.items():
        atlas[m] = {cell: s / t for cell, (s, t) in cells.items() if t > 0}
    return atlas


# ── Grid snapping ─────────────────────────────────────────────────────────────

_AZ_GRID = [0, 45, 90, 135, 180, 225, 270, 315]
_EL_GRID = [3, 5, 10, 20, 30, 45, 60, 80]


def _snap(value, grid):
    """Snap a continuous value to the nearest grid point."""
    return min(grid, key=lambda g: abs(g - value))


# ── Matcher cascade ───────────────────────────────────────────────────────────

# Ordered by preference (best first).  minima is #1 based on atlas evidence.
_MATCHER_CASCADE = ["minima", "superpoint-lightglue", "sift"]

# Atlas-derived success rate thresholds for trust lights
_GREEN_THRESH = 0.85
_AMBER_THRESH = 0.40


def get_best_matcher(az, el, atlas=None):
    """Return (matcher_name, protocol, trust_level, success_rate).

    Parameters
    ----------
    az, el  : float  — sun azimuth / elevation in degrees
    atlas   : dict   — output of load_atlas(); if None, uses heuristics

    Returns
    -------
    matcher : str    — e.g. "minima"
    protocol: str    — "raw" or "stretch_2_98"
    trust   : str    — "GREEN" | "AMBER" | "RED"
    rate    : float  — estimated success probability [0, 1]
    """
    az_bin = _snap(az, _AZ_GRID)
    el_bin = _snap(el, _EL_GRID)

    if atlas is None:
        # Fallback heuristic (no atlas available)
        rate = 0.93 if el >= 5 else 0.75
        trust = "GREEN" if rate >= _GREEN_THRESH else "AMBER"
        return "minima", "raw", trust, rate

    # Walk the cascade and pick first matcher with known atlas data
    for matcher in _MATCHER_CASCADE:
        if matcher not in atlas:
            continue
        cell_data = atlas[matcher]
        rate = cell_data.get((az_bin, el_bin), None)
        if rate is None:
            # Estimate from elevation-only (collapse azimuth)
            el_rates = [v for (a, e), v in cell_data.items() if e == el_bin]
            rate = sum(el_rates) / len(el_rates) if el_rates else 0.0

        trust = (
            "GREEN" if rate >= _GREEN_THRESH
            else "AMBER" if rate >= _AMBER_THRESH
            else "RED"
        )
        # Use raw protocol; stretch only if grazing (<= 5°)
        protocol = "stretch_2_98" if el_bin <= 5 else "raw"
        return matcher, protocol, trust, rate

    # Nothing in atlas — RED light, refuse
    return "sift", "raw", "RED", 0.0


# ── Negative control ──────────────────────────────────────────────────────────

def negative_control_check(pts1, pts2, reproj_thresh=3.0, min_inliers=15):
    """Validate a match set via RANSAC.

    Returns (is_valid, M_est, reason_str)
    """
    if pts1 is None or len(pts1) < 20:
        return False, None, f"Insufficient matches ({len(pts1) if pts1 is not None else 0} < 20)"

    M_est, inliers = cv2.estimateAffine2D(
        pts1, pts2,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=reproj_thresh,
        maxIters=2000,
    )
    if M_est is None:
        return False, None, "RANSAC failed to find a valid transform"

    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
    if inlier_count < min_inliers:
        return False, M_est, f"Too few inliers ({inlier_count} < {min_inliers})"

    det = M_est[0, 0] * M_est[1, 1] - M_est[0, 1] * M_est[1, 0]
    if abs(det) < 0.1 or abs(det) > 10.0:
        return False, M_est, f"Degenerate transform (det={det:.3f})"

    return True, M_est, f"Success ({inlier_count} inliers, det={det:.3f})"


# ── Low-level matcher runner ──────────────────────────────────────────────────

def run_matcher_pipeline(img1, img2, matcher_name, protocol="raw", scratch_dir="data/scratch"):
    """Run a single matcher and return a result dict.

    Parameters
    ----------
    img1, img2     : np.ndarray uint8 grayscale
    matcher_name   : str  e.g. "minima"
    protocol       : "raw" | "stretch_2_98"
    scratch_dir    : path for temp PNG files

    Returns
    -------
    dict with keys: matcher, runtime, raw_matches, is_valid, transform, reason
    """
    def _apply_protocol(img, proto):
        if proto == "stretch_2_98":
            p2, p98 = np.percentile(img, (2, 98))
            if p98 > p2:
                img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        return img

    img1 = _apply_protocol(img1.copy(), protocol)
    img2 = _apply_protocol(img2.copy(), protocol)

    os.makedirs(scratch_dir, exist_ok=True)
    p1 = os.path.join(scratch_dir, "route_1.png")
    p2 = os.path.join(scratch_dir, "route_2.png")

    start = time.time()
    pts1, pts2 = np.array([]), np.array([])

    try:
        if matcher_name == "sift":
            if _VISMATCH_AVAILABLE:
                from PIL import Image as _PIL
                _PIL.fromarray(img1).save(p1)
                _PIL.fromarray(img2).save(p2)
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in (p for p in matches if len(p) == 2)
                        if m.distance < 0.75 * n.distance]
                if good:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        else:
            if not _VISMATCH_AVAILABLE:
                raise ImportError("vismatch not available")
            from PIL import Image as _PIL
            _PIL.fromarray(img1).save(p1)
            _PIL.fromarray(img2).save(p2)
            matcher = vismatch.get_matcher(matcher_name, device="cpu")
            res = matcher(p1, p2)
            if isinstance(res, tuple):
                pts1, pts2 = res[:2]
            else:
                pts1, pts2 = res["matched_kpts0"], res["matched_kpts1"]
    except Exception as e:
        return {
            "matcher": matcher_name,
            "runtime": time.time() - start,
            "raw_matches": 0,
            "is_valid": False,
            "transform": None,
            "reason": f"Error: {e}",
        }

    runtime = time.time() - start
    is_valid, M_est, reason = negative_control_check(pts1, pts2)

    return {
        "matcher": matcher_name,
        "runtime": runtime,
        "raw_matches": len(pts1),
        "is_valid": is_valid,
        "transform": M_est,
        "reason": reason,
    }


# ── High-level entry point ────────────────────────────────────────────────────

def route_and_match(img1, img2, az1, el1, az2=None, el2=None,
                    atlas_path="results/atlas.csv", scratch_dir="data/scratch"):
    """Full routing + matching pipeline.

    Parameters
    ----------
    img1, img2 : np.ndarray uint8 grayscale — images to match
    az1, el1   : float — sun angle for image 1 (reference)
    az2, el2   : float — sun angle for image 2; defaults to az1, el1 if None

    Returns
    -------
    result : dict — matcher result + routing metadata
    trust  : str  — "GREEN" | "AMBER" | "RED"
    """
    if az2 is None:
        az2 = az1
    if el2 is None:
        el2 = el1

    # Use the larger illumination change for routing (conservative)
    az_use = az1
    el_use = min(el1, el2) if el2 is not None else el1

    atlas = load_atlas(atlas_path)
    matcher, protocol, trust, rate = get_best_matcher(az_use, el_use, atlas)

    if trust == "RED":
        return {
            "matcher": matcher,
            "runtime": 0.0,
            "raw_matches": 0,
            "is_valid": False,
            "transform": None,
            "reason": f"RED trust light — estimated success rate {rate:.0%} at az={az_use}° el={el_use}°",
            "trust": trust,
            "estimated_success_rate": rate,
        }, trust

    result = run_matcher_pipeline(img1, img2, matcher, protocol, scratch_dir)
    result["trust"] = trust
    result["estimated_success_rate"] = rate
    result["protocol"] = protocol
    return result, trust
