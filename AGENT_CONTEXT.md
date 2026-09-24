# AGENT_CONTEXT.md — read this before touching anything

Project: **LunarAlign** ("Sun-Sweep Atlas") — SIH26166, ISRO / Department of Space.
PS Title: Multi-modal, Sun angle and scale invariant image correspondence using
Chandrayaan-2 optical images (OHRC, TMC, IIRS). Theme: Space Technology.
Category: Software. Team: Selenographers.

If you are an AI agent picking this project up fresh: read this whole file before
running, editing, or "fixing" anything. This project has a documented history of
an earlier agent fabricating results (invented NCC values, invented match counts,
an invented 760x760px render that never existed). That was caught, reverted, and
replaced with a real, verified sweep. Do not repeat that pattern.

---

## CRITICAL AUDIT ALERT: THE CURRENT PDF DECK IS STALE

> [!WARNING]
> **`SIH26166_SunSweepAtlas_FINAL.pdf` is an UNFIXED, STALE export.**
> Despite being named `FINAL`, it still contains fabricated numbers, outdated 5°
> renders, and unverified captions from before commits `756505c` and `b8e8ca0`.
> **DO NOT trust or pitch numbers from that PDF.**
> All presentation numbers must be derived directly from the verified CSVs
> (`results/atlas_ppt_grid.csv` and `results/ncc_reference_grid.csv`).

---

## HARD RULES — do not violate these

1. **Never tune code, seeds, or parameters to reproduce old numbers.** The old
   PPT numbers below are CONFIRMED FABRICATED and VOID. They are listed here
   only so you recognize them and do not accidentally chase them:
   - NCC -0.74 at 180° flip — FABRICATED. Real value: **-0.541795**.
   - NCC +0.52 (grazing) / -0.70 (grazing+flip) — UNVERIFIED/STALE (from 5° bug).
     Real values at 10°: **+0.558803** and **-0.621754**.
   - ORB "6083 matches" at az=0/el=40 — FABRICATED. Real value: **2,080**.
   - ORB caption "240 matches, 13 correct, 353px off" — UNVERIFIED/FABRICATED.
     Real collapse at az=45/el=40: **560 matches, 27 correct** (77x collapse).
   - "760x760 px, 2 m/px" render caption — FABRICATED. Real config: **512x512 px, 20 m/px**.
   - Any AKAZE results — FABRICATED. AKAZE is NOT available in this environment.
   - "44% shadow at 5°, rising to 60% at 3°" — STALE (5° render bug).
     Real value (verified by direct pixel count on 2026-09-24): **21% shadow at 10° Sun elevation, up from 2.9% at 40°**.
2. **AKAZE is confirmed unavailable.** `cv2.AKAZE_create()` returns False in
   this environment's OpenCV build, and 'akaze' is not in the vismatch registry.
   Do not fake it, simulate it, or claim it ran. Drop it from all slides and tables.
3. **Only these 6 matchers are real here:** `sift`, `orb`, `loftr`,
   `superpoint-lightglue`, `aliked-lightglue`, `minima`. Do NOT reference RoMa,
   XoFTR, ASIFT, or a generic unqualified "LightGlue" as if they were run —
   none of those have ever actually been executed in this repo's frozen sweep.
   Drop RoMa and XoFTR badges from "Added by us" (paper citations may stay).
4. **Report raw numbers only.** No rounding in a favorable direction, no
   "should work," no summaries that omit bad-looking cells. If asked to
   validate something, show the actual command output, not a paraphrase.
5. **Old PPT text and code (760px, -0.74, 6083, 5-degree grazing, 44% shadow)
   must never be presented as current.** If you find any of these strings anywhere
   in the repo or the deck, flag it — it's a leftover, not a target.

---

## VERIFIED REAL STATE (as of git HEAD `b8e8ca07a75d9f8bf6b34854de9e39bc9fe3ee1b`)

- **Git HEAD commit:** `b8e8ca07a75d9f8bf6b34854de9e39bc9fe3ee1b`
  * Commits since baseline `249150e`:
    - `a3f30cb`: Added honest 10° 4-panel image strip (`results/slide2_4panel_honest_el10.png`).
    - `756505c`: Added `update_slide2_pptx.py` to patch slide 2 XML/media in PPTX.
    - `b8e8ca0`: Added `update_slide5_pptx.py` for honest 10°/40° shadow stats in PPTX.
- **config.yaml committed BEFORE the sweep** at commit `7b459d3eadd4961dc4ef6d09cd67c6b3f3e729ee`.
  Real frozen success criterion (the only one that is actually pre-committed):
  `max_grid_error_px: 3.0` AND `min_correct_matches: 20`.
  Grid: `delta_azimuth: [0,45,90,135,180]`, `elevation: [10,40]`,
  reference illumination `az=270, el=40`.
- **results/atlas_ppt_grid.csv** — the current source of truth. Exactly 60 rows
  (5 delta-az x 2 el x 6 matchers), one cell each, verified unique count.
  **results/atlas_full_sweep_backup.csv** is a separate, older/larger sweep
  (240 rows, different grid). Keep these files separate — do not merge them.
- **Determinism confirmed:** same (matcher, az, el) cell run twice back-to-back
  produced bit-for-bit identical output (SIFT and MINIMA both checked).
- **ncc_reference_grid.csv** — real persisted NCC values vs reference (az=270, el=40):
  * Reference (az=270, el=40): **+1.000000**
  * 180° flip (az=90, el=40): **-0.541795**
  * Grazing (az=270, el=10): **+0.558803**
  * Grazing + flip (az=90, el=10): **-0.621754**
  * 45° delta (az=315, el=40): **+0.635082** | el=10: **+0.405117**
  * 90° delta (az=0, el=40): **+0.121841**   | el=10: **+0.039134**
  * 135° delta (az=45, el=40): **-0.339122** | el=10: **-0.393198**
- **Negative control (real, already run):**
  * Evaluated in `results/negative_control_report.json` (seed=42 vs seed=9999).
  * SIFT correctly refused (18 raw matches < 20 match floor; autonomous refusal).
  * MINIMA matched 71 candidate keypoints and RANSAC found 22 inliers (det=1.061)
    by aligning coincidental crater rims. Against identity GT, error was 114.9 px!
  * **Ground-truth-free plausibility check exists in `src/router.py`**:
    `verify_transform_plausibility()` checks SVD singular values (scale $0.80 \le s \le 1.25$),
    anisotropy/shear ratio ($s_1/s_2 \le 1.20$), and inlier spatial dispersion coverage.
  * **CRITICAL AUDIT (2026-09-24):** Synthetic proxy testing confirmed the plausibility check
    does NOT catch MINIMA's negative-control false positive. With det=1.061, s1~1.035, s2~1.025,
    anisotropy~1.01, all geometric checks pass. The `produced_red_refusal: true` in the JSON
    was derived from the ground-truth-dependent `success_criterion_met` field, NOT from the
    plausibility filter. `ransac_valid: true` proves the router accepted the false match.
    Real MINIMA test was blocked by incomplete HuggingFace model download (missing minima_roma.pt
    and minima_xoftr.pt). **This open problem remains unsolved.**
- **SHA-256 Simulator Pinning: CONFIRMED IMPLEMENTED.**
  * `SIMULATOR_HASHES.txt` pins `src/sun_sim_v2.py` (`eb05f65...`) and `src/synth_dem.py` (`841ba17...`).
  * `demo.py`, `src/step3_sweep.py`, and `run_negative_control_and_images.py` actively hash both files
    on startup and exit with error if hashes do not match. The deck claim is TRUE.
- **Dashboard: CONFIRMED IMPLEMENTED.**
  * `dashboard.py` is a complete 481-line offline Streamlit application with Plotly heatmaps,
    trust-light indicators, and fallback-safe data loading.

---

## CURRENT PITCH FRAMING (locked)

NOT "we route to the best matcher per sun angle" — this is weakened by the
real data, since MINIMA alone succeeded on 10/10 illumination-only cells.
Current framing: **"we verify every fit and refuse what can't be verified."**
MINIMA usually being strong is precisely why routing alone isn't enough — the
negative control proves even the strongest matcher can be confidently wrong.

---

## DECK (PPTX / PDF) AUDIT — SLIDE-BY-SLIDE TRUTH TABLE

| Slide | Element | Stale PDF Value | Real Verified Value / Repo State | Action Required |
|---|---|---|---|---|
| **Slide 1** | Team ID | Blank / placeholder | Needs SPOC ID | Fill once assigned |
| **Slide 2** | Render Spec | 760×760 px · 2 m/px | **512×512 px · 20 m/px** | Replace text |
| **Slide 2** | Matcher List | Lists AKAZE | 6 matchers (SIFT, ORB, LoFTR, SP+LG, ALIKED+LG, MINIMA) | Drop AKAZE |
| **Slide 2** | 180° Flip NCC | -0.74 | **-0.541795** (~ -0.54) | Replace text |
| **Slide 2** | Grazing NCC | +0.52 (captioned el 05°) | **+0.558803** (~ +0.56 at el 10°) | Replace text & caption |
| **Slide 2** | Grazing+Flip NCC | -0.70 (captioned el 05°) | **-0.621754** (~ -0.62 at el 10°) | Replace text & caption |
| **Slide 2** | Thumbnails | Old 5° images (44%/42% shadow) | `ppt_render_grazing_az270_el10.png` & `_flip_az090_el10.png` (or `slide2_4panel_honest_el10.png`) | Swap images |
| **Slide 3** | Atlas Table | 30 cells (SIFT, AKAZE, ORB) | Full 60 rows from `results/atlas_ppt_grid.csv` (6 matchers × 5 Δaz × 2 el) | Replace entire table |
| **Slide 3** | ORB (Δaz 0°, el 40°) | 6083 matches | **2,080 / 2,080 matches** (0.0 px grid error) | Replace text |
| **Slide 3** | ORB Collapse Caption | "240 matches, 13 correct, 353px off" | **560 matches, 27 correct** (inf error, 77x collapse) | Replace text |
| **Slide 3** | Trust Thresholds | Not stated on slide | GREEN ≥85%, AMBER 40-85%, RED <40% are **heuristics** | Do not label as frozen |
| **Slide 4** | Classical Slice | Lists AKAZE | SIFT, ORB only | Drop AKAZE |
| **Slide 4** | SHA-256 Claim | "SHA-256 pinned, read-only" | **VERIFIED IMPLEMENTED** in `SIMULATOR_HASHES.txt` | Keep claim |
| **Slide 4** | Project Statuses | Learned sweep "in progress", Neg control "in progress", Dashboard "planned" | Learned sweep: **BUILT**, Neg control: **BUILT & TESTED**, Dashboard: **BUILT** | Update to show real completion |
| **Slide 5** | Shadow Stat | "44% at 5°, rising to 60% at 3°" | **21% at 10° Sun elevation, up from 2.9% at 40°** (verified by pixel count) | Replace text (`update_slide5_pptx.py`) |
| **Slide 5** | Astronomy Facts | 1.5° max pole el, 38m vs 1.2m shadow | Verified against NASA / orbital geometry | Keep unchanged |
| **Slide 6** | "Added by us" | Lists RoMa and XoFTR | RoMa and XoFTR were never run in frozen sweep | Drop badges (keep citations) |
| **Slide 6** | Reproducibility | "Every number comes from a script" | Becomes 100% true once deck reflects CSVs | Re-derive all numbers |

---

## OPEN PROBLEMS & IMPLEMENTATION STATUS (Ranked)

1. **Ground-truth-free rejection signal:**
   - *Problem:* Catch confidently-wrong fits like MINIMA's negative-control failure without ground truth.
   - *Status:* **PARTIALLY IMPLEMENTED** in `src/router.py` (`verify_transform_plausibility`).
     Performs SVD singular-value scale checking ($0.80 \le s \le 1.25$), shear anisotropy limit ($s_1/s_2 \le 1.20$),
     and spatial dispersion bounding box coverage ($\ge 10\%$).
   - *Audit finding (2026-09-24):* The check catches extreme geometric distortions but does NOT catch
     MINIMA's plausible-looking false positive (det=1.061, s1/s2=1.01 — all within thresholds).
     Possible fixes: cross-validation on inlier subsets, NCC consistency on warped patch, multi-matcher consensus.
2. **`router.py`:**
   - *Status:* **BUILT in `src/router.py`** (296 lines). Atlas-lookup cascade + per-pair RANSAC plausibility check.
     GREEN/AMBER/RED thresholds explicitly treated as operational heuristics.
3. **SHA-256 pinning on simulator:**
   - *Status:* **BUILT & VERIFIED.** `SIMULATOR_HASHES.txt` enforced at startup in `demo.py`, `step3_sweep.py`,
     and `run_negative_control_and_images.py`.
4. **Real lunar data:**
   - *Status:* Next milestone. Pipeline currently operates on synthetic DEMs (`synth_dem.py`).
     Target data: Chandrayaan-2 OHRC/TMC-2 pairs from PRADAN and LRO NAC crops from QuickMap.
5. **Dashboard:**
   - *Status:* **BUILT.** `dashboard.py` (481 lines, Streamlit + Plotly, dark mode, offline support).

---

## FILE MAP

- `results/atlas_ppt_grid.csv` — current source of truth, 60 rows
- `results/atlas_full_sweep_backup.csv` — older sweep, kept separate
- `ncc_reference_grid.csv` — persisted real NCC values
- `config.yaml` — pre-committed frozen success criterion
- `SIMULATOR_HASHES.txt` — SHA-256 integrity hashes for simulator files
- `results/negative_control_report.json` — verified negative control run data
- `results/ppt_render_*.png` — corrected 10° images for slide 2
- `results/slide2_4panel_honest_el10.png` — honest composite 4-panel image strip
- `src/router.py` — atlas-driven trust router with physical plausibility filter
- `dashboard.py` — offline Streamlit dashboard
- `update_slide2_pptx.py` & `update_slide5_pptx.py` — scripts to patch PPTX

---

Last verified against git commit `b8e8ca07a75d9f8bf6b34854de9e39bc9fe3ee1b`.
If you are a new agent: verify this commit is still HEAD before trusting any
number above. If it isn't, something changed without this file being updated —
say so plainly instead of assuming the file is still accurate.
