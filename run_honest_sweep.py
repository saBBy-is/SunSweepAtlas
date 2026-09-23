"""
run_honest_sweep.py — Executes the exact 9-step honest evaluation requested by the user.

1. Matchers: SIFT, ORB, LoFTR, SuperPoint-LightGlue, ALIKED-LightGlue, MINIMA.
2. Grid: 5 delta_az [0, 45, 90, 135, 180] x 2 elevations [10, 40] = 10 pairs.
   Reference geometry: az=270, el=40. Fixed terrain: synth_dem.py seed=42, 512x512, 20 m/px.
   Protocol: raw. Factor: A_identity.
   6 matchers x 10 pairs = 60 cells exactly once.
3. Persists results/atlas_ppt_grid.csv and results/ncc_reference_grid.csv.
4. Executes explicit wrong-location negative control (seed=9999 vs seed=42) for MINIMA and SIFT.
5. Documents provenance of demo_fallback/ images and exports fresh verified images to results/.
"""

import sys, os, time, hashlib, yaml, csv
from datetime import datetime, timezone
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# ── 1. Simulator Hash Check ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXPECTED_HASHES = {
    os.path.join(PROJECT_ROOT, "src", "sun_sim_v2.py"): "eb05f657168a53c99b8be71ee9c634d67eadcb82fb2a9da824afd36530b9a79b",
    os.path.join(PROJECT_ROOT, "src", "synth_dem.py"):   "841ba178ac8b1d1833ac9cb121083cd1446f895e5f0b6b96280e291b97dc1a9a",
}
for path, exp in EXPECTED_HASHES.items():
    with open(path, "rb") as fh:
        act = hashlib.sha256(fh.read()).hexdigest()
    if act != exp:
        raise RuntimeError(f"Integrity check failed for {os.path.basename(path)}!")
print("Simulator integrity checks PASSED.")

from sun_sim_v2 import horizon_map, render, to_uint8
from synth_dem import make_dem
import vismatch
from step3_sweep import run_matcher, evaluate, ncc
from router import negative_control_check

def run_all():
    # ── 2. Load Config ─────────────────────────────────────────────────────────
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    REF_AZ = float(cfg['reference']['azimuth'])     # 270.0
    REF_EL = float(cfg['reference']['elevation'])   # 40.0
    MAX_ERR = float(cfg['success']['max_grid_error_px'])  # 3.0
    MIN_COR = int(cfg['success']['min_correct_matches'])  # 20

    delta_azs = cfg['grid']['delta_azimuth']        # [0, 45, 90, 135, 180]
    elevations = cfg['grid']['elevation']           # [10, 40]
    abs_azs = [int((REF_AZ + d) % 360) for d in delta_azs]

    matchers = ['sift', 'orb', 'loftr', 'superpoint-lightglue', 'aliked-lightglue', 'minima']
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    scratch = os.path.join(PROJECT_ROOT, 'data', 'scratch')
    os.makedirs(scratch, exist_ok=True)

    # ── 3. Render Reference Terrain ────────────────────────────────────────────
    print("Generating reference terrain (seed=42, 512x512, 20 m/px)...")
    dem_ref = make_dem(512, 20)
    h_ref = horizon_map(dem_ref, 20.0, REF_AZ)
    img_ref = to_uint8(render(dem_ref, 20.0, REF_AZ, REF_EL, horizon=h_ref))

    # Precompute horizon maps for all azimuths on ref terrain
    h_cache = {int(REF_AZ): h_ref}
    for a in abs_azs:
        if a not in h_cache:
            h_cache[a] = horizon_map(dem_ref, 20.0, a)

    # ── 4. Fresh 60-Cell Sweep (protocol=raw, factor=A_identity) ───────────────
    out_csv = os.path.join(results_dir, "atlas_ppt_grid.csv")
    ncc_csv = os.path.join(results_dir, "ncc_reference_grid.csv")

    header = ("matcher,protocol,factor,delta_az,abs_az,el,"
              "matches,correct,grid_err,success,"
              "ncc_vs_ref,runtime,terrain_id,timestamp\n")
    with open(out_csv, 'w') as f:
        f.write(header)

    with open(ncc_csv, 'w') as f:
        f.write("delta_az,abs_az,el,ncc_vs_ref\n")

    M_identity = np.eye(2, 3)
    total_cells = len(delta_azs) * len(elevations) * len(matchers)
    cell_idx = 0

    print(f"\n--- Running 60-cell PPT Grid ({len(delta_azs)} az x {len(elevations)} el x {len(matchers)} matchers) ---")
    ncc_done = set()

    for daz, aaz in zip(delta_azs, abs_azs):
        for el in elevations:
            img_tgt = to_uint8(render(dem_ref, 20.0, aaz, el, horizon=h_cache[aaz]))
            ncc_val = ncc(img_ref, img_tgt)

            pair_key = (daz, aaz, el)
            if pair_key not in ncc_done:
                ncc_done.add(pair_key)
                with open(ncc_csv, 'a') as f:
                    f.write(f"{daz},{aaz},{el},{ncc_val:.6f}\n")

            for m in matchers:
                cell_idx += 1
                pts1, pts2, rtime = run_matcher(m, img_ref, img_tgt, scratch)
                if pts1 is None:
                    n_m, n_c, gerr, succ = 0, 0, float('inf'), False
                else:
                    n_m, n_c, gerr, succ = evaluate(pts1, pts2, M_identity, MAX_ERR, MIN_COR)

                gerr_str = f"{gerr:.4f}" if gerr != float('inf') else "inf"
                ts = datetime.now(timezone.utc).isoformat()
                row = (f"{m},raw,A_identity,{daz},{aaz},{el},"
                       f"{n_m},{n_c},{gerr_str},{succ},"
                       f"{ncc_val:.6f},{rtime:.3f},synth01,{ts}\n")
                with open(out_csv, 'a') as f:
                    f.write(row)

                print(f"[{cell_idx:>2}/{total_cells}] daz={daz:>3} aaz={aaz:>3} el={el:>2} | {m:<22} | "
                      f"m={n_m:>5} c={n_c:>5} err={gerr_str:>8} succ={'OK' if succ else 'FAIL'} ({rtime:.2f}s)")

    print(f"\nCompleted {cell_idx} cells. Output written to {out_csv}.")

    # ── 5. Wrong-Location Negative Control ──────────────────────────────────────
    print("\n--- Running Wrong-Location Negative Control ---")
    # Reference image: dem_ref (seed 42), az=270, el=40
    # Negative control query: dem_neg (seed 9999), az=270, el=40 (same illumination, different terrain)
    dem_neg = make_dem(512, 20, seed=9999)
    h_neg = horizon_map(dem_neg, 20.0, REF_AZ)
    img_neg = to_uint8(render(dem_neg, 20.0, REF_AZ, REF_EL, horizon=h_neg))

    neg_results = {}
    for test_m in ['minima', 'sift']:
        pts1, pts2, rtime = run_matcher(test_m, img_ref, img_neg, scratch)
        n_m = len(pts1) if pts1 is not None else 0

        # Check against identity transform (which is completely wrong for different terrain)
        n_m_eval, n_c_eval, gerr_eval, succ_eval = evaluate(pts1, pts2, M_identity, MAX_ERR, MIN_COR)

        # Check through router.negative_control_check
        is_valid_ransac, M_est, reason_ransac = negative_control_check(pts1, pts2, reproj_thresh=MAX_ERR, min_inliers=15)

        # Inlier count if M_est exists
        inlier_count = 0
        if M_est is not None and pts1 is not None and len(pts1) > 0:
            pts1_t = (M_est @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T
            errs = np.hypot(pts1_t[:, 0] - pts2[:, 0], pts1_t[:, 1] - pts2[:, 1])
            inlier_count = int(np.sum(errs <= MAX_ERR))

        neg_results[test_m] = {
            "raw_matches": n_m,
            "correct_matches_vs_identity": n_c_eval,
            "grid_err_vs_identity": str(gerr_eval),
            "success_criterion_met": succ_eval,
            "ransac_valid": is_valid_ransac,
            "ransac_inliers": inlier_count,
            "ransac_reason": reason_ransac,
            "produced_red_refusal": (not succ_eval) or (not is_valid_ransac),
            "runtime_s": round(rtime, 3)
        }
        print(f"Negative Control [{test_m}]:")
        print(f"  Raw matches returned      : {n_m}")
        print(f"  Correct vs Identity       : {n_c_eval}")
        print(f"  Grid error vs Identity    : {gerr_eval}")
        print(f"  Success criterion satisfied: {succ_eval}")
        print(f"  RANSAC check valid        : {is_valid_ransac}")
        print(f"  RANSAC reason             : {reason_ransac}")
        print(f"  Produced RED / Refusal     : {neg_results[test_m]['produced_red_refusal']}")

    neg_report_path = os.path.join(results_dir, "negative_control_report.json")
    import json
    with open(neg_report_path, 'w') as f:
        json.dump(neg_results, f, indent=2)

    # ── 6. Export Verified PPT Images ──────────────────────────────────────────
    print("\n--- Exporting Verified PPT Images ---")
    # Fresh renders from committed code (512x512, 20 m/px, seed=42)
    img_ref_fresh = img_ref
    img_flip_fresh = to_uint8(render(dem_ref, 20.0, 90.0, 40.0, horizon=h_cache[90]))
    img_grz10_fresh = to_uint8(render(dem_ref, 20.0, 270.0, 10.0, horizon=h_ref))
    img_grf10_fresh = to_uint8(render(dem_ref, 20.0, 90.0, 10.0, horizon=h_cache[90]))
    # Also el=5.0 from demo.py
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

    print("\nAll tasks in run_honest_sweep completed successfully.")

if __name__ == '__main__':
    run_all()
