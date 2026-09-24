# 🌕 SunSweepAtlas — Lunar Image Matcher Trust Engine

**SIH 2026 · PS 26166 · Team Selenographers**
ISRO OHRC/TMC/IIRS Lunar Surface Image Alignment under Extreme Sun Angles

> **Live Dashboard**: `streamlit run dashboard.py` → [localhost:8501](http://localhost:8501)

---

## 🚀 Quick Start (Run on Your PC)

```bash
# 1. Clone the repo
git clone https://github.com/saBBy-is/SunSweepAtlas.git
cd SunSweepAtlas

# 2. Install dependencies (Python 3.10+)
pip install streamlit plotly numpy

# 3. Launch the dashboard
streamlit run dashboard.py
```

Open **http://localhost:8501** in your browser. That's it.

---

## What This Does

Lunar surface images look completely different depending on where the Sun is.
At **low sun angles (≤10°)**, shadows cover **21%** of pixels — matchers break.
At **high sun angles (≥40°)**, shadows drop to **2.9%** — matchers work.

**SunSweepAtlas** answers: *"Which image matcher should I trust at this sun angle?"*

### The Pipeline

```
LOLA DEM → Synthetic Sun Rendering → 6 Matchers × 10 Sun Geometries
    → 60-Cell Atlas → Trust Router → "Use THIS matcher, or REFUSE to match"
```

### Matchers Tested
| Matcher | Type | Overall Success |
|---------|------|----------------|
| SIFT | Classical | See atlas |
| ORB | Classical | See atlas |
| SuperPoint+LightGlue | Deep Learning | See atlas |
| ALIKED+LightGlue | Deep Learning | See atlas |
| LoFTR | Transformer | See atlas |
| MINIMA | Foundation Model | See atlas |

### Trust Levels
- 🟢 **GREEN** (≥85%) — Safe to use
- 🟡 **AMBER** (40–84%) — Use with caution
- 🔴 **RED** (<40%) — Refuse to match

---

## 📊 Dashboard Features

| Tab | What It Shows |
|-----|--------------|
| ☀ **SUN POSITION** | Interactive 3D sky hemisphere — drag to rotate, see sun at your chosen az/el |
| **HEATMAPS** | Per-matcher success rate across all sun geometries |
| **ALL MATCHERS** | Side-by-side comparison with trust-light cards |
| **TRUST ROUTER** | Pick an angle → get the best matcher recommendation |
| **3-D SURFACE** | Rotatable success-rate landscape |
| **RAW DATA** | Full atlas.csv table |

---

## 📁 Project Structure

```
SunSweepAtlas/
├── dashboard.py          # Main dashboard (streamlit + plotly)
├── demo.py               # Quick alignment demo
├── config.yaml           # Pipeline configuration
├── requirements.txt      # Python dependencies
├── src/
│   ├── router.py         # Trust router logic
│   ├── step1_dem.py      # DEM loading
│   ├── step2_render.py   # Sun-angle rendering
│   └── step3_sweep.py    # Matcher sweep engine
├── results/
│   └── atlas.csv         # 60-row ground-truth atlas (source of truth)
├── data/                 # DEM tiles and test images
├── AGENT_CONTEXT.md      # Development rules and audit log
├── MISSION_BRIEF.md      # Original problem statement
└── README.md             # This file
```

---

## Key Results

- **Shadow at 10° elevation**: 21.2% pixel coverage (pixel-counted, not estimated)
- **Shadow at 40° elevation**: 2.9% pixel coverage
- **Atlas size**: 60 cells (6 matchers × 5 Δaz × 2 el)
- **Plausibility check**: SVD-based sanity filter in router (catches extreme distortions)
- **Known limitation**: Plausibility check does NOT catch MINIMA false positives (documented in AGENT_CONTEXT.md)

---

## Requirements

- Python 3.10+
- `streamlit` (dashboard UI)
- `plotly` (charts + 3D)
- `numpy` (3D hemisphere math)

```bash
pip install streamlit plotly numpy
```

---

## License

Academic project — SIH 2026, Smart India Hackathon.
Team Selenographers · Problem Statement 26166 · ISRO
