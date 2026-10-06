# 🌕 SunSweepAtlas — Multi-Modal Lunar Image Matcher Trust Engine

**SIH 2026 · PS 26166 · Team Selenographers**
Multi-modal, Sun angle and scale invariant image correspondence using
Chandrayaan-2 optical images (OHRC, TMC-2, IIRS)

> **Live Dashboard**: `streamlit run dashboard.py` → [localhost:8501](http://localhost:8501)

---

## 🚀 Quick Start (Run on Your PC)

```bash
# 1. Clone the repo
git clone https://github.com/saBBy-is/SunSweepAtlas.git
cd SunSweepAtlas

# 2. Install dependencies (Python 3.10+)
pip install streamlit plotly numpy scipy opencv-python

# 3. Launch the dashboard
streamlit run dashboard.py

# 4. Run the demo (includes cross-sensor matching)
python demo.py
```

Open **http://localhost:8501** in your browser. That's it.

---

## What This Does

Lunar surface images look completely different depending on:
1. **Where the Sun is** — shadows cover 21% of pixels at 10° elevation vs 2.9% at 40°
2. **Which sensor took the image** — OHRC (0.3 m/px) sees fine crater rims that TMC-2 (5 m/px) and IIRS (80 m/px) cannot resolve

**SunSweepAtlas** answers: *"Which image matcher should I trust for THIS sun angle AND THIS sensor pair?"*

### The Multi-Modal Pipeline

```
                    ┌─── OHRC  (0.3 m/px) ──┐
LOLA DEM → Render → ├─── TMC-2 (5.0 m/px) ──┼→ 6 Matchers × 10 Sun Geometries
                    └─── IIRS  (80  m/px) ──┘         × 3 Sensor Pairs
                                                            │
                    60-Cell Single-Sensor Atlas ─────────────┤
                    Cross-Sensor Atlas ─────────────────────┤
                            │                               │
                    Trust Router (sun angle + scale ratio) ──→ "Use THIS matcher,
                                                                or REFUSE to match"
```

### Sensor Profiles (Chandrayaan-2)

| Sensor | Full Name | GSD | Spectral | Scale vs OHRC |
|--------|-----------|-----|----------|:---:|
| **OHRC** | Orbiter High Resolution Camera | 0.3 m/px | PAN 450–850 nm | 1× |
| **TMC-2** | Terrain Mapping Camera-2 | 5.0 m/px | PAN 500–850 nm | 17× |
| **IIRS** | Imaging IR Spectrometer | 80 m/px | VNIR+SWIR 800–5000 nm | 267× |

### Cross-Sensor Matching Challenge

| Sensor Pair | Scale Ratio | Difficulty |
|-------------|:-----------:|:---:|
| OHRC ↔ TMC-2 | 17× | 🟡 Moderate |
| TMC-2 ↔ IIRS | 16× | 🟡 Moderate |
| OHRC ↔ IIRS | 267× | 🔴 Extreme |

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

Trust is determined by **both** sun angle difficulty **and** cross-sensor scale ratio.

---

## 📊 Dashboard Features

| Tab | What It Shows |
|-----|--------------|
| ☀ **SUN POSITION** | Interactive 3D sky hemisphere — drag to rotate |
| **HEATMAPS** | Per-matcher success rate across all sun geometries |
| **ALL MATCHERS** | Side-by-side comparison with trust-light cards |
| **TRUST ROUTER** | Pick an angle + sensor pair → get best matcher |
| **3-D SURFACE** | Rotatable success-rate landscape |
| **RAW DATA** | Full atlas.csv table |

---

## 📁 Project Structure

```
SunSweepAtlas/
├── dashboard.py               # Main dashboard (streamlit + plotly)
├── demo.py                    # Quick demo (sun-angle + cross-sensor)
├── main.py                    # Multi-modal pipeline entry point
├── config.yaml                # Pipeline + sensor configuration
├── requirements.txt           # Python dependencies
├── src/
│   ├── sensors.py             # Chandrayaan-2 sensor profiles (OHRC, TMC-2, IIRS)
│   ├── multimodal.py          # Multi-resolution terrain + cross-sensor rendering
│   ├── router.py              # Trust router (sun-angle + sensor-pair aware)
│   ├── sun_sim_v2.py          # Lunar illumination renderer (SHA-256 pinned)
│   ├── synth_dem.py           # Synthetic DEM generator (SHA-256 pinned)
│   ├── data_loader.py         # PDS4 XML parser (sun angles + sensor ID)
│   ├── step3_sweep.py         # Single-sensor matcher sweep engine
│   ├── data/
│   │   ├── dem_loader.py      # GDAL DEM loader
│   │   └── image_loader.py    # Sensor-aware image loader
│   ├── pipeline/
│   │   └── router.py          # Simple trust router
│   └── matching/
│       └── benchmark.py       # SIFT/LoFTR matcher wrappers
├── run_cross_sensor_sweep.py  # Cross-sensor multi-modal sweep
├── results/
│   ├── atlas.csv              # 60-row single-sensor atlas (source of truth)
│   ├── cross_sensor_atlas.csv # Cross-sensor matching results
│   └── ncc_reference_grid.csv # Persisted NCC values
├── data/
│   └── dummy_metadata.xml     # Example PDS4 metadata
├── AGENT_CONTEXT.md           # Development rules and audit log
├── MISSION_BRIEF.md           # Problem statement
└── README.md                  # This file
```

---

## Key Results

- **Multi-modal**: Pipeline handles OHRC (0.3m), TMC-2 (5m), IIRS (80m) at native GSDs
- **Shadow at 10° elevation**: 21.2% pixel coverage (verified by pixel count)
- **Shadow at 40° elevation**: 2.9% pixel coverage
- **Single-sensor atlas**: 60 cells (6 matchers × 5 Δaz × 2 el)
- **Cross-sensor routing**: Trust level penalised by scale ratio (17×, 267×)
- **Plausibility check**: SVD-based sanity filter catches extreme distortions
- **Known limitation**: Plausibility check does NOT catch MINIMA false positives

---

## Requirements

- Python 3.10+
- `streamlit` (dashboard UI)
- `plotly` (charts + 3D)
- `numpy` (computation)
- `scipy` (multi-scale terrain generation)
- `opencv-python` (image processing + matching)

```bash
pip install streamlit plotly numpy scipy opencv-python
```

Optional (for deep-learning matchers):
```bash
pip install kornia torch torchvision vismatch
```

---

## License

Academic project — SIH 2026, Smart India Hackathon.
Team Selenographers · Problem Statement 26166 · ISRO
