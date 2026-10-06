#!/usr/bin/env python
"""
run_multimodal_demo.py -- Full Multi-Modal Pipeline Demo for SIH 2026
=====================================================================

Generates:
  1. Multi-sensor comparison figure (OHRC, TMC-2, IIRS side-by-side)
  2. Spectral contrast analysis between sensors
  3. Cross-sensor matching results with SIFT and ORB
  4. Trust routing decisions for each sensor pair + sun angle
  5. All outputs saved to results/ for dashboard and presentation

This is the script you run to prove the project is genuinely multi-modal.
"""

import os
import sys
import time
import csv
import cv2
import numpy as np
from datetime import datetime, timezone

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))

from sensors import OHRC, TMC2, IIRS, SENSOR_PAIRS, SensorPair, get_sensor
from multimodal import (
    MultiScaleTerrain, render_sensor_view,
    prepare_cross_sensor_match, cross_sensor_ncc,
    generate_composition_map, spectral_albedo_map,
    _SENSOR_BAND_MAP,
)

try:
    import yaml
    with open(os.path.join(_here, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
except Exception:
    cfg = {
        "reference": {"azimuth": 270, "elevation": 40},
        "grid": {"delta_azimuth": [0, 45, 90, 135, 180], "elevation": [10, 40]},
    }

REF_AZ = float(cfg["reference"]["azimuth"])
REF_EL = float(cfg["reference"]["elevation"])

results_dir = os.path.join(_here, "results")
os.makedirs(results_dir, exist_ok=True)


# ========================================================================
# STEP 1: Generate Multi-Sensor Comparison Figure
# ========================================================================

def generate_comparison_figure():
    """Generate side-by-side sensor views showing spectral + scale differences."""
    print("\n" + "=" * 70)
    print("  STEP 1: Multi-Sensor Comparison Figure")
    print("=" * 70)

    terrain = MultiScaleTerrain(physical_extent_m=2560.0, base_gsd_m=2.0, seed=42)
    comp = generate_composition_map(terrain.dem, seed=42)

    # Render each sensor at reference illumination
    sensors = [OHRC, TMC2, IIRS]
    sensor_imgs = {}
    center = (terrain.extent_m / 2, terrain.extent_m / 2)

    for sensor in sensors:
        max_sz = terrain.get_max_output_size(sensor)
        use_sz = min(256, max_sz)
        img = render_sensor_view(terrain, sensor, REF_AZ, REF_EL,
                                 use_sz, center, comp)
        # Resize to common display size for comparison
        display = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        sensor_imgs[sensor.name] = display
        shadow_pct = float(np.sum(img == 0)) / img.size * 100
        mean_lit = float(img[img > 0].mean()) if img.max() > 0 else 0
        band = _SENSOR_BAND_MAP.get(sensor.name, "?")
        print(f"  {sensor.name:6s}: {img.shape[0]}x{img.shape[1]}px "
              f"(GSD={sensor.gsd_m}m), band={band}, "
              f"shadow={shadow_pct:.1f}%, mean_lit={mean_lit:.1f}")

    # Create 3-panel comparison
    # Add labels
    panels = []
    for name in ["OHRC", "TMC-2", "IIRS"]:
        img = sensor_imgs[name].copy()
        # Add text label
        cv2.putText(img, name, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,), 2, cv2.LINE_AA)
        gsd = get_sensor(name).gsd_m
        cv2.putText(img, f"GSD={gsd}m/px", (8, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200,), 1, cv2.LINE_AA)
        band = _SENSOR_BAND_MAP.get(name, "?")
        cv2.putText(img, band.replace("_", " "), (8, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,), 1, cv2.LINE_AA)
        panels.append(img)

    # Separator lines
    sep = np.full((256, 3), 128, dtype=np.uint8)
    comparison = np.hstack([panels[0], sep, panels[1], sep, panels[2]])

    out_path = os.path.join(results_dir, "multimodal_3sensor_comparison.png")
    cv2.imwrite(out_path, comparison)
    print(f"\n  -> Saved: {out_path}")

    # Also generate per-sun-angle views (grazing vs normal)
    for el, label in [(10, "grazing"), (40, "normal")]:
        imgs = []
        for sensor in sensors:
            max_sz = terrain.get_max_output_size(sensor)
            use_sz = min(256, max_sz)
            img = render_sensor_view(terrain, sensor, REF_AZ, el,
                                     use_sz, center, comp)
            display = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.putText(display, f"{sensor.name} el={el}deg", (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1, cv2.LINE_AA)
            imgs.append(display)
        row = np.hstack([imgs[0], sep, imgs[1], sep, imgs[2]])
        path = os.path.join(results_dir, f"multimodal_3sensor_{label}.png")
        cv2.imwrite(path, row)
        print(f"  -> Saved: {path}")

    return terrain, comp


# ========================================================================
# STEP 2: Spectral Contrast Analysis
# ========================================================================

def spectral_contrast_analysis(terrain, comp):
    """Measure spectral differences between sensors."""
    print("\n" + "=" * 70)
    print("  STEP 2: Spectral Contrast Analysis")
    print("=" * 70)

    center = (terrain.extent_m / 2, terrain.extent_m / 2)
    sensor_data = {}

    for sensor in [OHRC, TMC2, IIRS]:
        max_sz = terrain.get_max_output_size(sensor)
        use_sz = min(128, max_sz)
        img = render_sensor_view(terrain, sensor, REF_AZ, REF_EL,
                                 use_sz, center, comp)
        sensor_data[sensor.name] = img

    # Cross-sensor NCC matrix
    names = ["OHRC", "TMC-2", "IIRS"]
    print("\n  Cross-Sensor NCC Matrix (spectral + scale):")
    print("  " + "-" * 50)
    print(f"  {'':8s}", end="")
    for n in names:
        print(f"{n:>10s}", end="")
    print()

    ncc_matrix = {}
    for a in names:
        print(f"  {a:8s}", end="")
        for b in names:
            ncc = cross_sensor_ncc(sensor_data[a], sensor_data[b])
            ncc_matrix[(a, b)] = ncc
            print(f"{ncc:+10.3f}", end="")
        print()

    # Save NCC matrix
    csv_path = os.path.join(results_dir, "spectral_ncc_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sensor_a", "sensor_b", "ncc", "scale_ratio"])
        for (a, b), ncc in ncc_matrix.items():
            sa, sb = get_sensor(a), get_sensor(b)
            ratio = max(sa.gsd_m, sb.gsd_m) / min(sa.gsd_m, sb.gsd_m)
            w.writerow([a, b, f"{ncc:.6f}", f"{ratio:.1f}"])
    print(f"\n  -> Saved: {csv_path}")

    return ncc_matrix


# ========================================================================
# STEP 3: Cross-Sensor Matching Sweep (SIFT + ORB)
# ========================================================================

def run_cross_sensor_sweep(terrain, comp):
    """Run cross-sensor matching with available matchers."""
    print("\n" + "=" * 70)
    print("  STEP 3: Cross-Sensor Matching Sweep")
    print("=" * 70)

    delta_azs = cfg["grid"]["delta_azimuth"]
    elevations = cfg["grid"]["elevation"]
    abs_azs = [int((REF_AZ + d) % 360) for d in delta_azs]

    # Available matchers
    matchers = ["sift", "orb"]
    try:
        import vismatch
        matchers += ["loftr", "superpoint-lightglue", "aliked-lightglue", "minima"]
        print("  vismatch available -- using all 6 matchers")
    except ImportError:
        print("  vismatch not available -- using SIFT + ORB only")

    pairs = SENSOR_PAIRS
    scratch = os.path.join(_here, "data", "scratch")
    os.makedirs(scratch, exist_ok=True)

    csv_path = os.path.join(results_dir, "cross_sensor_atlas.csv")
    header = [
        "sensor_pair", "sensor_a", "sensor_b", "scale_ratio",
        "matcher", "delta_az", "abs_az", "el",
        "matches", "correct", "grid_err", "success",
        "ncc_cross", "spectral_band_a", "spectral_band_b",
        "resample_strategy", "common_gsd",
        "size_a", "size_b", "runtime", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    total = len(pairs) * len(delta_azs) * len(elevations) * len(matchers)
    cell_n = 0
    successes = 0

    print(f"\n  Sweep: {len(pairs)} pairs x {len(delta_azs)} daz x "
          f"{len(elevations)} el x {len(matchers)} matchers = {total} cells\n")

    center = (terrain.extent_m / 2, terrain.extent_m / 2)

    for pair in pairs:
        print(f"\n  --- {pair.key} (scale: {pair.scale_ratio:.0f}x) ---")

        max_a = terrain.get_max_output_size(pair.sensor_a)
        max_b = terrain.get_max_output_size(pair.sensor_b)
        use_a = min(512, max_a)
        use_b = min(512, max_b)

        band_a = _SENSOR_BAND_MAP.get(pair.sensor_a.name, "?")
        band_b = _SENSOR_BAND_MAP.get(pair.sensor_b.name, "?")

        for daz, abs_az in zip(delta_azs, abs_azs):
            for el in elevations:
                # Render with spectral composition
                try:
                    img_a = render_sensor_view(
                        terrain, pair.sensor_a, REF_AZ, REF_EL,
                        use_a, center, comp
                    )
                    img_b = render_sensor_view(
                        terrain, pair.sensor_b, abs_az, el,
                        use_b, center, comp
                    )
                except ValueError:
                    for m in matchers:
                        cell_n += 1
                        row = [pair.key, pair.sensor_a.name, pair.sensor_b.name,
                               f"{pair.scale_ratio:.1f}", m, daz, abs_az, el,
                               0, 0, "inf", False, 0.0, band_a, band_b,
                               "coarse", 0.0, 0, 0, 0.0,
                               datetime.now(timezone.utc).isoformat()]
                        with open(csv_path, "a", newline="") as f:
                            csv.writer(f).writerow(row)
                    continue

                ncc_val = cross_sensor_ncc(img_a, img_b)

                # Prepare for matching (resample to common GSD)
                try:
                    proc_a, proc_b, common_gsd, sf = prepare_cross_sensor_match(
                        img_a, img_b, pair.sensor_a, pair.sensor_b,
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
                        n_m, n_c, gerr, succ, rtime = _run_and_eval(
                            m, proc_a, proc_b, scratch
                        )

                    if succ:
                        successes += 1

                    gerr_str = f"{gerr:.4f}" if gerr != float("inf") else "inf"
                    ts = datetime.now(timezone.utc).isoformat()

                    row = [
                        pair.key, pair.sensor_a.name, pair.sensor_b.name,
                        f"{pair.scale_ratio:.1f}",
                        m, daz, abs_az, el,
                        n_m, n_c, gerr_str, succ,
                        f"{ncc_val:.6f}", band_a, band_b,
                        "coarse", f"{common_gsd:.1f}",
                        img_a.shape[0], img_b.shape[0],
                        f"{rtime:.3f}", ts,
                    ]
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow(row)

                    pct = 100 * cell_n / total
                    status = "OK" if succ else "FAIL"
                    print(
                        f"    [{cell_n:>3}/{total}] {pct:5.1f}%  "
                        f"daz={daz:>3} el={el:>2}  "
                        f"{m:<25s}  m={n_m:>5} c={n_c:>5} err={gerr_str:>9}  "
                        f"{status}  {rtime:.1f}s"
                    )

    print(f"\n  Sweep complete: {successes}/{total} cells succeeded "
          f"({100*successes/total:.1f}%)")
    print(f"  -> Saved: {csv_path}")
    return csv_path


def _run_and_eval(m_name, img1, img2, scratch):
    """Run matcher + evaluate (identity GT for cross-sensor)."""
    os.makedirs(scratch, exist_ok=True)
    p1 = os.path.join(scratch, "xm1.png")
    p2 = os.path.join(scratch, "xm2.png")
    cv2.imwrite(p1, img1)
    cv2.imwrite(p2, img2)

    t0 = time.time()
    try:
        if m_name == "sift":
            det = cv2.SIFT_create()
            kp1, d1 = det.detectAndCompute(img1, None)
            kp2, d2 = det.detectAndCompute(img2, None)
            if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
                return 0, 0, float("inf"), False, time.time() - t0
            bf = cv2.BFMatcher()
            raw = bf.knnMatch(d1, d2, k=2)
            good = [m for m, n in raw if m.distance < 0.75 * n.distance]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]) if good else np.array([])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]) if good else np.array([])
        elif m_name == "orb":
            det = cv2.ORB_create(nfeatures=10000)
            kp1, d1 = det.detectAndCompute(img1, None)
            kp2, d2 = det.detectAndCompute(img2, None)
            if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
                return 0, 0, float("inf"), False, time.time() - t0
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(d1, d2)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]) if matches else np.array([])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]) if matches else np.array([])
        else:
            import vismatch
            matcher = vismatch.get_matcher(m_name, device="cpu")
            res = matcher(p1, p2)
            if isinstance(res, tuple):
                pts1, pts2 = res[:2]
            else:
                pts1, pts2 = res["matched_kpts0"], res["matched_kpts1"]
    except Exception:
        return 0, 0, float("inf"), False, time.time() - t0

    rtime = time.time() - t0
    n = len(pts1) if pts1 is not None else 0
    if n < 10:
        return n, 0, float("inf"), False, rtime

    # Evaluate (identity GT)
    errs = np.hypot(pts1[:, 0] - pts2[:, 0], pts1[:, 1] - pts2[:, 1])
    n_correct = int(np.sum(errs <= 5.0))

    # Estimate affine
    M_est, inliers = cv2.estimateAffine2D(
        pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=5.0,
        maxIters=2000,
    )
    if M_est is None:
        return n, n_correct, float("inf"), False, rtime

    det = M_est[0, 0] * M_est[1, 1] - M_est[0, 1] * M_est[1, 0]
    if abs(det) < 0.1 or abs(det) > 10.0:
        return n, n_correct, float("inf"), False, rtime

    # Grid reprojection error
    sz = min(pts1[:, 0].max(), pts2[:, 0].max(), 512)
    if sz < 4:
        return n, n_correct, float("inf"), False, rtime
    gy, gx = np.mgrid[0:sz:sz/10, 0:sz:sz/10]
    gpts = np.column_stack([gx.ravel(), gy.ravel()])
    gpts_h = np.hstack([gpts, np.ones((len(gpts), 1))])
    grid_est = (M_est @ gpts_h.T).T
    grid_err = float(np.mean(np.hypot(
        gpts[:, 0] - grid_est[:, 0], gpts[:, 1] - grid_est[:, 1]
    )))

    success = (grid_err < 5.0) and (n_correct >= 10)
    return n, n_correct, grid_err, success, rtime


# ========================================================================
# STEP 4: Trust Routing Summary
# ========================================================================

def generate_routing_summary():
    """Show trust routing decisions for each sensor pair + sun angle."""
    print("\n" + "=" * 70)
    print("  STEP 4: Trust Routing Summary")
    print("=" * 70)

    sys.path.insert(0, os.path.join(_here, "src"))
    from router import get_best_matcher_cross_sensor, load_atlas

    atlas = load_atlas(os.path.join(_here, "results", "atlas.csv"))

    delta_azs = cfg["grid"]["delta_azimuth"]
    elevations = cfg["grid"]["elevation"]
    abs_azs = [int((REF_AZ + d) % 360) for d in delta_azs]

    csv_path = os.path.join(results_dir, "cross_sensor_routing.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sensor_pair", "scale_ratio", "delta_az", "el",
                     "matcher", "trust", "adj_rate", "penalty"])

    pair_names = [("OHRC", "TMC-2"), ("TMC-2", "IIRS"), ("OHRC", "IIRS")]

    for a_name, b_name in pair_names:
        sa, sb = get_sensor(a_name), get_sensor(b_name)
        ratio = max(sa.gsd_m, sb.gsd_m) / min(sa.gsd_m, sb.gsd_m)
        print(f"\n  {a_name} <-> {b_name} (scale: {ratio:.0f}x)")

        for daz, abs_az in zip(delta_azs, abs_azs):
            for el in elevations:
                matcher, protocol, trust, rate, sr, pk = \
                    get_best_matcher_cross_sensor(abs_az, el, a_name, b_name, atlas)
                symbol = {"GREEN": "[G]", "AMBER": "[A]", "RED": "[R]"}[trust]
                print(f"    daz={daz:>3} el={el:>2}: {symbol} {trust:6s} "
                      f"-> {matcher} ({rate:.0%})")

                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        f"{a_name}_x_{b_name}".replace("-", ""),
                        f"{ratio:.1f}", daz, el,
                        matcher, trust, f"{rate:.4f}", f"{sr:.1f}",
                    ])

    print(f"\n  -> Saved: {csv_path}")


# ========================================================================
# MAIN
# ========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  SunSweepAtlas Multi-Modal Pipeline Demo")
    print("  PS 26166: Multi-modal, Sun-angle-invariant image correspondence")
    print("  Sensors: OHRC (0.3m) + TMC-2 (5m) + IIRS (80m)")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Visualizations
    terrain, comp = generate_comparison_figure()

    # Step 2: Spectral analysis
    spectral_contrast_analysis(terrain, comp)

    # Step 3: Cross-sensor matching sweep
    run_cross_sensor_sweep(terrain, comp)

    # Step 4: Routing summary
    generate_routing_summary()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  All steps complete in {elapsed:.1f}s")
    print(f"  Results in: {results_dir}")
    print(f"{'=' * 70}")
