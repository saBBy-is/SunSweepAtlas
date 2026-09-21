#!/usr/bin/env python
"""
demo.py — SIH 2026 Live Demo: Sun-Sweep Atlas proof-of-concept
═══════════════════════════════════════════════════════════════
Imports ONLY existing LunarAlign functions (read-only).
No GPU, no internet, no web server.  Runtime < 15 s.

Usage:
    python demo.py            # terminal output + saved fallback
    python demo.py --show     # also pop up matplotlib panel

Fallback outputs in  demo_fallback/
"""
import sys, os, time, argparse

# ── path setup (works from project root) ──────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")                       # safe default; swapped below if --show
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sun_sim_v2 import horizon_map, render, to_uint8
from synth_dem import make_dem


# ═══════════════════════════════════════════════════════════════════
#  CONFIG  — matches the slide parameters exactly
# ═══════════════════════════════════════════════════════════════════
DEM_SIZE   = 512
PX_M       = 20.0                           # metres per pixel

# Slide 2 — four illumination conditions (same DEM, nadir camera)
REF_AZ,  REF_EL  = 270.0, 40.0             # reference: Sun from West, mid-elevation
FLIP_AZ, FLIP_EL =  90.0, 40.0             # 180° azimuth flip
GRZ_AZ,  GRZ_EL  = 270.0, 5.0             # grazing Sun (same direction)
GRF_AZ,  GRF_EL  =  90.0, 5.0             # grazing + flipped

# Slide 3 — ORB matcher collapse
ROT0_AZ,  ROT0_EL  = 270.0, 40.0           # 0° Sun rotation (= reference)
ROT45_AZ, ROT45_EL = 315.0, 40.0           # 45° Sun rotation

OUT_DIR = os.path.join(_here, "demo_fallback")


# ═══════════════════════════════════════════════════════════════════
#  HELPERS  (pure functions, no side-effects on existing code)
# ═══════════════════════════════════════════════════════════════════

def ncc(a, b):
    """Pearson / normalised cross-correlation between two uint8 images."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean();  b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def orb_correct(img1, img2, nfeatures=10000, thresh_px=3.0):
    """ORB + cross-check BFMatcher.  Returns (total, correct, kp1, kp2, matches)."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, d1 = orb.detectAndCompute(img1, None)
    kp2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return 0, 0, kp1 or [], kp2 or [], []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)
    correct = sum(
        1 for m in matches
        if np.linalg.norm(
            np.array(kp1[m.queryIdx].pt) - np.array(kp2[m.trainIdx].pt)
        ) < thresh_px
    )
    return len(matches), correct, kp1, kp2, matches


def make_render(dem, az, el, hcache):
    """Render with per-azimuth horizon caching."""
    if az not in hcache:
        hcache[az] = horizon_map(dem, PX_M, az)
    return to_uint8(render(dem, PX_M, az, el, horizon=hcache[az]))


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main(show_panel=False):
    if show_panel:
        matplotlib.use("TkAgg")
        import importlib
        importlib.reload(plt)

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    hcache = {}

    hdr  = "=" * 62
    line = "-" * 62
    print(f"\n{hdr}")
    print("   🌕  LunarAlign — SIH 2026 Live Demo")
    print(f"{hdr}\n")

    # ── 1.  Generate scene ────────────────────────────────────────
    print("[1/4] Generating synthetic crater DEM (seed=42) …")
    dem = make_dem(DEM_SIZE, int(PX_M))

    # ── 2.  Slide 2: Sun-geometry renders + NCC ───────────────────
    print("[2/4] Rendering four illumination conditions …")
    img_ref  = make_render(dem, REF_AZ,  REF_EL,  hcache)
    img_flip = make_render(dem, FLIP_AZ, FLIP_EL, hcache)
    img_grz  = make_render(dem, GRZ_AZ,  GRZ_EL,  hcache)
    img_grf  = make_render(dem, GRF_AZ,  GRF_EL,  hcache)

    ncc_flip = ncc(img_ref, img_flip)
    ncc_grz  = ncc(img_ref, img_grz)
    ncc_grf  = ncc(img_ref, img_grf)

    print(f"\n  SLIDE 2 — NCC vs reference  (az={REF_AZ:.0f}°  el={REF_EL:.0f}°)")
    print(f"  {line}")
    print(f"  │ {'Condition':<26} │ {'NCC':>8} │")
    print(f"  {line}")
    print(f"  │ Reference  (self)         │ {'  +1.00':>8} │")
    print(f"  │ 180° flip  (az=90°)       │ {ncc_flip:>+8.2f} │")
    print(f"  │ Grazing    (el=10°)       │ {ncc_grz:>+8.2f} │")
    print(f"  │ Grazing + flip            │ {ncc_grf:>+8.2f} │")
    print(f"  {line}")
    print(f"  ↳ Negative NCC = shadow inversion.")
    print(f"    Drop from +1 = illumination destroys pixel agreement.\n")

    # ── 3.  Slide 3: ORB matcher collapse ─────────────────────────
    print("[3/4] Running ORB matcher (10 000 features, cross-check) …")
    img_r0  = make_render(dem, ROT0_AZ,  ROT0_EL,  hcache)   # same illum
    img_r45 = make_render(dem, ROT45_AZ, ROT45_EL, hcache)   # 45° rotated

    tot0,  cor0,  kp1a, kp2a, m0  = orb_correct(img_ref, img_r0)
    tot45, cor45, kp1b, kp2b, m45 = orb_correct(img_ref, img_r45)

    print(f"\n  SLIDE 3 — ORB correct matches  (identity geometry)")
    print(f"  {line}")
    print(f"  │ {'Sun rotation':<26} │ {'Correct':>8} / {'Total':<6} │")
    print(f"  {line}")
    print(f"  │ 0°   (same illumination)  │ {cor0:>8,d} / {tot0:<6,d} │")
    print(f"  │ 45°  (az 270→315)         │ {cor45:>8,d} / {tot45:<6,d} │")
    print(f"  {line}")
    ratio = cor0 / max(cor45, 1)
    print(f"  ↳ {ratio:.0f}× collapse with just 45° Sun rotation!\n")

    # ── 4.  Fallback outputs ──────────────────────────────────────
    print("[4/4] Saving fallback to demo_fallback/ …")

    # Individual images
    for name, img in [("reference", img_ref), ("180flip", img_flip),
                       ("grazing", img_grz),   ("grazing_flip", img_grf),
                       ("orb_rot0", img_r0),    ("orb_rot45", img_r45)]:
        cv2.imwrite(os.path.join(OUT_DIR, f"render_{name}.png"), img)

    # ── Matplotlib panel ──────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.12,
                   top=0.90, bottom=0.04, left=0.02, right=0.98)

    # Top row: four renders
    labels = [
        f"Reference\naz {REF_AZ:.0f}°  el {REF_EL:.0f}°\nNCC = +1.00",
        f"180° Flip\naz {FLIP_AZ:.0f}°  el {FLIP_EL:.0f}°\nNCC = {ncc_flip:+.2f}",
        f"Grazing\naz {GRZ_AZ:.0f}°  el {GRZ_EL:.0f}°\nNCC = {ncc_grz:+.2f}",
        f"Grazing + Flip\naz {GRF_AZ:.0f}°  el {GRF_EL:.0f}°\nNCC = {ncc_grf:+.2f}",
    ]
    images = [img_ref, img_flip, img_grz, img_grf]
    ncc_colours = ["#00ff88", "#ff4444", "#ffaa22", "#ff4444"]

    for i, (lab, img, col) in enumerate(zip(labels, images, ncc_colours)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(lab, color=col, fontsize=10, fontweight="bold",
                     fontfamily="monospace")
        ax.axis("off")

    # Bottom row: ORB match visualisations
    ref_bgr = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
    r0_bgr  = cv2.cvtColor(img_r0,  cv2.COLOR_GRAY2BGR)
    r45_bgr = cv2.cvtColor(img_r45, cv2.COLOR_GRAY2BGR)

    vis0  = cv2.drawMatches(ref_bgr, kp1a, r0_bgr,  kp2a, m0[:80],  None,
                            matchColor=(0, 255, 100),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis45 = cv2.drawMatches(ref_bgr, kp1b, r45_bgr, kp2b, m45[:80], None,
                            matchColor=(80, 80, 255),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    ax_l = fig.add_subplot(gs[1, :2])
    ax_l.imshow(cv2.cvtColor(vis0, cv2.COLOR_BGR2RGB))
    ax_l.set_title(f"ORB  0° Sun rotation  —  {cor0:,d} correct / {tot0:,d} total",
                   color="#00ff88", fontsize=11, fontweight="bold", fontfamily="monospace")
    ax_l.axis("off")

    ax_r = fig.add_subplot(gs[1, 2:])
    ax_r.imshow(cv2.cvtColor(vis45, cv2.COLOR_BGR2RGB))
    ax_r.set_title(f"ORB  45° Sun rotation  —  {cor45:,d} correct / {tot45:,d} total",
                   color="#ff4444", fontsize=11, fontweight="bold", fontfamily="monospace")
    ax_r.axis("off")

    fig.suptitle("LunarAlign  —  Sun-Sweep Atlas Demo   (SIH 2026)",
                 color="white", fontsize=15, fontweight="bold",
                 fontfamily="monospace", y=0.96)

    panel_path = os.path.join(OUT_DIR, "demo_panel.png")
    fig.savefig(panel_path, dpi=150, facecolor=fig.get_facecolor())

    # ── Text summary ──────────────────────────────────────────────
    txt = [
        "LunarAlign — SIH 2026 Demo Results",
        "=" * 40,
        "",
        f"SLIDE 2 — NCC vs reference (az={REF_AZ:.0f}° el={REF_EL:.0f}°):",
        f"  180° flip:       NCC = {ncc_flip:+.2f}",
        f"  Grazing (el=10): NCC = {ncc_grz:+.2f}",
        f"  Grazing + flip:  NCC = {ncc_grf:+.2f}",
        "",
        "SLIDE 3 — ORB correct matches (identity geometry):",
        f"  0° Sun rotation:  {cor0:,d} / {tot0:,d}",
        f"  45° Sun rotation: {cor45:,d} / {tot45:,d}",
        f"  Collapse ratio:   {ratio:.0f}x",
    ]
    txt_path = os.path.join(OUT_DIR, "demo_results.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt) + "\n")

    elapsed = time.time() - t0
    print(f"\n  ✅ Panel saved:   {panel_path}")
    print(f"  ✅ Text saved:    {txt_path}")
    print(f"  ⏱  Total time:   {elapsed:.1f} s")
    print(f"\n{hdr}\n")

    if show_panel:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="Pop up the matplotlib panel")
    args = ap.parse_args()
    main(show_panel=args.show)
