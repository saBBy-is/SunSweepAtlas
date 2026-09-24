# 🌕 sun sweep atlas — SIH 2026 Project (PS: 26166)

**Team**: Selenographers  
**Objective**: Multi-modal, Sun-angle and scale-invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC, IIRS) for ISRO / Department of Space.

## 🚀 The Differentiator: Sun-Sweep Atlas & Trust Router

While most AI matchers fail unpredictably when lighting changes on the lunar surface, **LunarAlign** guarantees reliability through a physics-informed pipeline:
1. **Shading Simulation**: We synthesize lunar terrain under 10 different lighting geometries (5 azimuths × 2 elevations, using a synthetic DEM with SHA-256 integrity pinning).
2. **Sun-Sweep Atlas**: We benchmark 6 state-of-the-art matchers (SIFT, ORB, LoFTR, SuperPoint-LightGlue, ALIKED-LightGlue, MINIMA) across all sun angles to map exactly when they break.
3. **The Trust Router**: A smart fallback system that analyzes the incoming sun angle and automatically selects the most reliable matcher — or raises a **RED Trust Light** if extreme shadows make matching unreliable.

## 📁 Repository Structure
- `src/` — Core pipeline (Router, Simulation, Matching)
- `results/atlas.csv` — **The source of truth**: 60-row honest, un-cherry-picked atlas (6 matchers × 10 geometries)
- `results/ncc_reference_grid.csv` — NCC values across the grid
- `results/negative_control_report.json` — Wrong-terrain rejection test results
- `dashboard.py` — Streamlit interactive UI to explore the Sun-Sweep Atlas
- `demo.py` — Terminal demo demonstrating matcher collapse at extreme sun angles
- `run_honest_sweep.py` — Script to reproduce the full 60-cell physics sweep
- `AGENT_CONTEXT.md` — Project integrity document: hard rules, verified state, audit trail

## 🛠️ Quickstart

```bash
# Install dependencies (requires Python 3.11+)
pip install -r requirements.txt

# Run the physics demo (~6 seconds on CPU)
python demo.py --show

# Run the full 60-cell sweep (requires model downloads, ~8-20 min depending on hardware)
python run_honest_sweep.py

# Launch the interactive Streamlit dashboard
streamlit run dashboard.py
```

## ⚠️ Known Limitations

1. **Synthetic terrain only**: The pipeline currently runs on `synth_dem.py` (512×512, 20 m/px, seed=42). Real Chandrayaan-2 OHRC/TMC-2 data processing is the next milestone.
2. **GDAL not included**: GDAL requires C++ build tools. It is listed in the deck but excluded from `requirements.txt`. Not needed for the current demo or sweep.
3. **AKAZE unavailable**: `cv2.AKAZE_create()` returns False in this OpenCV build. AKAZE is excluded from all results.
4. **Ground-truth-free rejection**: The SVD/anisotropy plausibility check (`verify_transform_plausibility` in `src/router.py`) catches extreme geometric distortions but does not catch plausible-looking false alignments (e.g. MINIMA aligning wrong craters with det≈1.06, s1/s2≈1.01). This remains an open research problem.
5. **No GPU required**: All matchers run on CPU. First run requires downloading model weights from HuggingFace (~140 MB total).

---
*Built for Smart India Hackathon 2026.*
