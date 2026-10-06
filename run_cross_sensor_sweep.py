#!/usr/bin/env python
"""
run_cross_sensor_sweep.py — Cross-Sensor Multi-Modal Sweep for SIH 2026
═══════════════════════════════════════════════════════════════════════════

Extends the single-sensor Sun-Sweep Atlas with CROSS-SENSOR matching:
  OHRC  (0.3 m/px) ↔ TMC-2 (5 m/px)   — 17× scale difference
  TMC-2 (5 m/px)   ↔ IIRS  (80 m/px)  — 16× scale difference
  OHRC  (0.3 m/px) ↔ IIRS  (80 m/px)  — 267× scale difference

For each (sensor_pair, az, el) cell:
  1. Generates terrain at fine resolution (1 m/px base)
  2. Renders as each sensor would see it (different GSDs)
  3. Resamples to common resolution for matching
  4. Runs all available matchers
  5. Evaluates success (identity geometry GT)

Output: results/cross_sensor_atlas.csv

This script does NOT modify sun_sim_v2.py or synth_dem.py.
Uses config.yaml for the sun-angle grid.
"""

import os
import sys
import time
import yaml
import csv
import cv2
import numpy as np
from datetime import datetime, timezone

# Path setup
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))

from sensors import OHRC, TMC2, IIRS, SENSOR_PAIRS, SensorPair
from multimodal import (
    MultiScaleTerrain, render_sensor_view,
    prepare_cross_sensor_match, cross_sensor_ncc,
    generate_composition_map,
)

try:
    import vismatch
    _VISMATCH_OK = True
except ImportError:
    _VISMATCH_OK = False
    print("WARNING: vismatch not importable — only SIFT and ORB will run.")

try:
    from PIL import Image
except ImportError:
    Image = None


# ── Matcher runner ──────────────────────────────────────────────────────────

def run_matcher(m_name, img1, img2, scratch):
    """Run a single matcher, return (pts1, pts2, runtime) or (None, error, 0)."""
    os.makedirs(scratch, exist_ok=True)
    p1 = os.path.join(scratch, "xm1.png")
    p2 = os.path.join(scratch, "xm2.png")
    if Image is not None:
        Image.fromarray(img1).save(p1)
        Image.fromarray(img2).save(p2)
    else:
        cv2.imwrite(p1, img1)
        cv2.imwrite(p2, img2)

    t0 = time.time()
    try:
        if m_name == "sift":
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
        elif m_name == "orb":
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
            matcher = vismatch.get_matcher(m_name, device="cpu")
            res = matcher(p1, p2)
            if isinstance(res, tuple):
                pts1, pts2 = res[:2]
            else:
                pts1, pts2 = res["matched_kpts0"], res["matched_kpts1"]
            return pts1, pts2, time.time() - t0
    except Exception as e:
        return None, str(e), 0.0


# ── Evaluation (identity geometry GT since same terrain, same alignment) ────

def evaluate_cross_sensor(pts1, pts2, max_err_px=5.0, min_correct=10):
    """Evaluate cross-sensor match quality.

    Since both images are registered (same terrain, same centre), the GT
    transform is identity. But images may be different sizes, so we
    evaluate based on the absolute point displacement.

    Uses a looser threshold (5px) than single-sensor (3px) because
    resolution mismatch introduces inherent positional uncertainty.
    """
    n = len(pts1) if pts1 is not None else 0
    if n < min_correct:
        return n, 0, float("inf"), False

    # Points should match near-identity (after resampling to common res)
    errs = np.hypot(pts1[:, 0] - pts2[:, 0], pts1[:, 1] - pts2[:, 1])
    n_correct = int(np.sum(errs <= max_err_px))

    # Estimate affine (should be near-identity)
    M_est, inliers = cv2.estimateAffine2D(
        pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=max_err_px,
        maxIters=2000,
    )
    if M_est is None:
        return n, n_correct, float("inf"), False

    det = M_est[0, 0] * M_est[1, 1] - M_est[0, 1] * M_est[1, 0]
    if abs(det) < 0.1 or abs(det) > 10.0:
        return n, n_correct, float("inf"), False

    # Grid reprojection error (identity GT)
    sz = min(pts1[:, 0].max(), pts2[:, 0].max(), 512)
    gy, gx = np.mgrid[0:sz:sz / 10, 0:sz:sz / 10]
    gpts = np.column_stack([gx.ravel(), gy.ravel()])
    gpts_h = np.hstack([gpts, np.ones((len(gpts), 1))])
    grid_est = (M_est @ gpts_h.T).T
    grid_err = float(np.mean(np.hypot(
        gpts[:, 0] - grid_est[:, 0], gpts[:, 1] - grid_est[:, 1]
    )))

    success = (grid_err < max_err_px) and (n_correct >= min_correct)
    return n, n_correct, grid_err, success


# ── Main sweep ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_path = os.path.join(_here, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    REF_AZ = float(cfg["reference"]["azimuth"])
    REF_EL = float(cfg["reference"]["elevation"])
    delta_azs = cfg["grid"]["delta_azimuth"]
    elevations = cfg["grid"]["elevation"]

    abs_azs = [int((REF_AZ + d) % 360) for d in delta_azs]

    # Matchers — use whatever is available
    matchers = ["sift", "orb"]
    if _VISMATCH_OK:
        matchers += ["loftr", "superpoint-lightglue", "aliked-lightglue", "minima"]

    # Sensor pairs
    pairs = SENSOR_PAIRS

    results_dir = os.path.join(_here, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "cross_sensor_atlas.csv")

    # Write header
    header = [
        "sensor_pair", "sensor_a", "sensor_b", "scale_ratio",
        "matcher", "delta_az", "abs_az", "el",
        "matches", "correct", "grid_err", "success",
        "ncc_cross", "resample_strategy", "common_gsd",
        "size_a", "size_b", "runtime", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    scratch = os.path.join(_here, "data", "scratch")

    # Generate multi-scale terrain
    print("=" * 70)
    print("  CROSS-SENSOR MULTI-MODAL SWEEP")
    print("=" * 70)
    print("\nGenerating multi-scale terrain (seed=42, 2560m, 2m/px base)...")
    terrain = MultiScaleTerrain(
        physical_extent_m=2560.0, base_gsd_m=2.0, seed=42
    )
    print(f"  Base DEM: {terrain.base_size}x{terrain.base_size} at {terrain.base_gsd}m/px")

    # Generate spectral composition map (same for all pairs)
    print("Generating spectral composition map (mineral abundance)...")
    composition = generate_composition_map(terrain.dem, seed=42)
    print(f"  Composition: {composition.shape}, mean sum={composition.sum(axis=2).mean():.3f}")

    total = len(pairs) * len(delta_azs) * len(elevations) * len(matchers)
    cell_n = 0

    print(f"\nSweep: {len(pairs)} pairs x {len(delta_azs)} daz x "
          f"{len(elevations)} el x {len(matchers)} matchers = {total} cells\n")

    for pair in pairs:
        print(f"\n{'=' * 60}")
        print(f"  SENSOR PAIR: {pair.key}  (scale ratio: {pair.scale_ratio:.0f}x)")
        print(f"  {pair.sensor_a.name} ({pair.sensor_a.gsd_m}m/px) <-> "
              f"{pair.sensor_b.name} ({pair.sensor_b.gsd_m}m/px)")
        print(f"{'=' * 60}")

        max_a = terrain.get_max_output_size(pair.sensor_a)
        max_b = terrain.get_max_output_size(pair.sensor_b)
        use_a = min(512, max_a)
        use_b = min(512, max_b)

        if use_b < 8:
            print(f"  WARNING: {pair.sensor_b.name} output too small ({use_b}px) -- "
                  f"marking all RED")

        center = (terrain.extent_m / 2, terrain.extent_m / 2)

        for di, (daz, abs_az) in enumerate(zip(delta_azs, abs_azs)):
            for el in elevations:
                # Render sensor A (reference sun angle) and sensor B (target sun angle)
                try:
                    img_a = render_sensor_view(
                        terrain, pair.sensor_a,
                        REF_AZ, REF_EL, use_a, center, composition
                    )
                    img_b = render_sensor_view(
                        terrain, pair.sensor_b,
                        abs_az, el, use_b, center, composition
                    )
                except ValueError as e:
                    # Sensor can't view this terrain — skip
                    for m in matchers:
                        cell_n += 1
                        row = [
                            pair.key, pair.sensor_a.name, pair.sensor_b.name,
                            f"{pair.scale_ratio:.1f}",
                            m, daz, abs_az, el,
                            0, 0, "inf", False,
                            0.0, "coarse", 0.0,
                            0, 0, 0.0,
                            datetime.now(timezone.utc).isoformat(),
                        ]
                        with open(csv_path, "a", newline="") as f:
                            csv.writer(f).writerow(row)
                    continue

                # Cross-sensor NCC
                ncc_val = cross_sensor_ncc(img_a, img_b)

                # Prepare for matching (resample to common GSD)
                try:
                    proc_a, proc_b, common_gsd, sf = prepare_cross_sensor_match(
                        img_a, img_b,
                        pair.sensor_a, pair.sensor_b,
                        strategy="coarse",
                    )
                except Exception:
                    proc_a = proc_b = None
                    common_gsd = 0.0

                for m in matchers:
                    cell_n += 1

                    if proc_a is None or proc_a.shape[0] < 8 or proc_b.shape[0] < 8:
                        n_m, n_c, gerr, succ = 0, 0, float("inf"), False
                        rtime = 0.0
                    else:
                        pts1, pts2_or_err, rtime = run_matcher(
                            m, proc_a, proc_b, scratch
                        )
                        if pts1 is None:
                            n_m, n_c, gerr, succ = 0, 0, float("inf"), False
                        else:
                            n_m, n_c, gerr, succ = evaluate_cross_sensor(
                                pts1, pts2_or_err
                            )

                    gerr_str = f"{gerr:.4f}" if gerr != float("inf") else "inf"
                    ts = datetime.now(timezone.utc).isoformat()

                    row = [
                        pair.key, pair.sensor_a.name, pair.sensor_b.name,
                        f"{pair.scale_ratio:.1f}",
                        m, daz, abs_az, el,
                        n_m, n_c, gerr_str, succ,
                        f"{ncc_val:.6f}", "coarse", f"{common_gsd:.1f}",
                        img_a.shape[0], img_b.shape[0],
                        f"{rtime:.3f}", ts,
                    ]
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow(row)

                    pct = 100 * cell_n / total
                    status = "OK" if succ else "FAIL"
                    print(
                        f"  [{cell_n:>3}/{total}] {pct:5.1f}%  "
                        f"{pair.key:15s}  Δaz={daz:>3}° el={el:>2}°  "
                        f"{m:<25s}  m={n_m:>5} c={n_c:>5} err={gerr_str:>9}  "
                        f"{status}  {rtime:.1f}s"
                    )

    print(f"\n{'=' * 70}")
    print(f"  Cross-sensor sweep complete. {cell_n} rows → {csv_path}")
    print(f"{'=' * 70}")
