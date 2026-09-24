"""
dashboard.py — LunarAlign Matcher Atlas Dashboard (SIH 2026)
═════════════════════════════════════════════════════════════
Visualises the Sun-Sweep Atlas (results/atlas.csv) with:
  • Graphical polar-compass azimuth selector
  • Per-matcher success-rate heatmaps (az × el)
  • 3-D success-rate surface
  • Trust-light summary for every matcher
  • Router recommendation for any user-specified sun angle
  • Raw-data table

Runs with: streamlit run dashboard.py
NOTE: Uses stdlib csv + plotly.graph_objects (no pandas DLL dependency).
"""

import os
import csv
import sys
import math
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict

# ── Path setup ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LunarAlign — Sun-Sweep Matcher Atlas",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — NASA/ISRO Mission-Control aesthetic ──────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

  html, body, .stApp { background: #070b14; color: #c8d8e8; font-family: 'Inter', sans-serif; }
  .block-container { padding: 1.2rem 2rem 2rem; max-width: 1600px; }

  [data-testid="stSidebar"] { background: #090e1a; border-right: 1px solid #1a2540; }
  [data-testid="stSidebar"] * { color: #c8d8e8 !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label {
    font-size: 0.78rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: #5b7ba8 !important;
  }

  [data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1525 0%, #111d30 100%);
    border: 1px solid #1e2e4a; border-top: 2px solid #2a6496;
    border-radius: 4px; padding: 14px 18px;
  }
  [data-testid="stMetricLabel"] {
    color: #5b7ba8 !important; font-size: 0.72rem;
    letter-spacing: 0.1em; text-transform: uppercase;
  }
  [data-testid="stMetricValue"] {
    color: #e8f4ff !important; font-size: 1.7rem;
    font-weight: 700; font-family: 'Share Tech Mono', monospace;
  }
  [data-testid="stMetricDelta"] { font-size: 0.78rem; }

  .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #1a2540; background: transparent; }
  .stTabs [data-baseweb="tab"] {
    background: transparent; color: #5b7ba8;
    border: none; border-bottom: 2px solid transparent;
    padding: 10px 20px; font-size: 0.82rem;
    letter-spacing: 0.05em; font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background: transparent; color: #79c0ff !important;
    border-bottom: 2px solid #2a6496 !important;
  }

  .trust-green { color: #39d353; font-weight: 700; font-family: 'Share Tech Mono', monospace; }
  .trust-amber { color: #e3b341; font-weight: 700; font-family: 'Share Tech Mono', monospace; }
  .trust-red   { color: #ff4b4b; font-weight: 700; font-family: 'Share Tech Mono', monospace; }

  h1 { color: #e8f4ff !important; font-weight: 700; letter-spacing: -0.02em; }
  h2, h3 { color: #9ab8d4 !important; font-weight: 600; }
  hr { border-color: #1a2540; margin: 1rem 0; }

  .mission-header {
    background: linear-gradient(90deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
    border: 1px solid #1a2e50; border-radius: 4px;
    padding: 16px 24px; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 20px;
  }
  .mission-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.4rem; color: #79c0ff; letter-spacing: 0.08em;
  }
  .mission-sub { font-size: 0.78rem; color: #5b7ba8; letter-spacing: 0.05em; }

  .status-dot-green { display:inline-block; width:9px; height:9px; border-radius:50%; background:#39d353; box-shadow:0 0 6px #39d353; margin-right:6px; }
  .status-dot-amber { display:inline-block; width:9px; height:9px; border-radius:50%; background:#e3b341; box-shadow:0 0 6px #e3b341; margin-right:6px; }
  .status-dot-red   { display:inline-block; width:9px; height:9px; border-radius:50%; background:#ff4b4b; box-shadow:0 0 6px #ff4b4b; margin-right:6px; }

  .router-box {
    background: #0d1a30; border: 1px solid #1e3558;
    border-left: 3px solid #2a6496; border-radius: 4px;
    padding: 16px 20px; margin: 10px 0;
    font-family: 'Share Tech Mono', monospace; font-size: 0.92rem;
  }
  .router-box-red {
    background: #170a0a; border: 1px solid #3d1515;
    border-left: 3px solid #ff4b4b; border-radius: 4px;
    padding: 16px 20px; margin: 10px 0;
    font-family: 'Share Tech Mono', monospace;
  }
  .compass-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.72rem;
    color: #5b7ba8; text-align: center; letter-spacing: 0.1em; margin-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (pure stdlib — no pandas)
# ══════════════════════════════════════════════════════════════════════════════

ATLAS_PATH = os.path.join(_here, "results", "atlas.csv")

_GREEN_THRESH = 0.85
_AMBER_THRESH = 0.40
_AZ_GRID  = [0, 45, 90, 135, 180, 225, 270, 315]
_EL_GRID  = [3, 5, 10, 20, 30, 45, 60, 80]
_MATCHER_CASCADE = ["minima", "superpoint-lightglue", "loftr", "sift", "orb", "aliked-lightglue"]

_COLOR_MAP = {
    "minima":               "#58a6ff",
    "superpoint-lightglue": "#3fb950",
    "loftr":                "#bc8cff",
    "sift":                 "#f0883e",
    "orb":                  "#ff4b4b",
    "aliked-lightglue":     "#79c0ff",
}


@st.cache_data(show_spinner="Loading atlas…")
def load_atlas_raw():
    """Return list-of-dicts from the CSV (stdlib only)."""
    if not os.path.exists(ATLAS_PATH):
        return None
    rows = []
    with open(ATLAS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            row["az"]      = int(row.get("az", row.get("abs_az", 0)))
            row["el"]      = int(row["el"])
            row["matches"] = int(row["matches"])
            row["correct"] = int(row["correct"])
            row["success"] = row["success"] == "True"
            try:
                row["grid_err"] = float(row["grid_err"]) if row["grid_err"] not in ("inf", "nan", "") else float("inf")
            except ValueError:
                row["grid_err"] = float("inf")
            row["runtime"] = float(row["runtime"])
            rows.append(row)
    return rows


def _trust(rate):
    if rate >= _GREEN_THRESH:
        return "GREEN", "●"
    if rate >= _AMBER_THRESH:
        return "AMBER", "●"
    return "RED", "●"


def _snap(value, grid):
    return min(grid, key=lambda g: abs(g - value))


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def pivot_success(rows, matcher, protocol, factor):
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["matcher"] == matcher and r["protocol"] == protocol and r["factor"] == factor:
            acc[(r["az"], r["el"])][1] += 1
            if r["success"]:
                acc[(r["az"], r["el"])][0] += 1
    azs = sorted(set(k[0] for k in acc))
    els = sorted(set(k[1] for k in acc))
    z = []
    for el in els:
        row_data = []
        for az in azs:
            s, t = acc.get((az, el), [0, 0])
            row_data.append(round(s / t * 100, 1) if t > 0 else None)
        z.append(row_data)
    return azs, els, z


def pivot_correct(rows, matcher, protocol, factor):
    acc = defaultdict(list)
    for r in rows:
        if r["matcher"] == matcher and r["protocol"] == protocol and r["factor"] == factor:
            acc[(r["az"], r["el"])].append(r["correct"])
    azs = sorted(set(k[0] for k in acc))
    els = sorted(set(k[1] for k in acc))
    z = []
    for el in els:
        row_data = []
        for az in azs:
            vals = acc.get((az, el), [])
            row_data.append(round(sum(vals) / len(vals)) if vals else None)
        z.append(row_data)
    return azs, els, z


def build_atlas_rates(rows):
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        m = r["matcher"]
        acc[m][(r["az"], r["el"])][1] += 1
        if r["success"]:
            acc[m][(r["az"], r["el"])][0] += 1
    return {m: {cell: s / t for cell, (s, t) in cells.items() if t > 0}
            for m, cells in acc.items()}


def make_polar_compass(az_deg, el_deg):
    """Premium multi-ring HUD compass with azimuth needle, elevation fill, and tick marks."""
    theta_rad = math.radians(az_deg)
    nx = math.sin(theta_rad)
    ny = math.cos(theta_rad)
    ring_t = list(range(0, 361))

    # Elevation normalization (high el = small fill, low el = large fill)
    el_norm = 1.0 - el_deg / 90.0
    # Color shifts: low elevation → warm amber, high → cool cyan
    el_hue_r = max(0, 255 - int(el_deg * 2.5))
    el_hue_g = min(255, 80 + int(el_deg * 2))
    el_fill = f"rgba({el_hue_r},{el_hue_g},200,0.12)"
    el_edge = f"rgba({el_hue_r},{el_hue_g},200,0.35)"

    fig = go.Figure()

    # ── Concentric rings (outer → inner) for depth ────────────────────────
    for r, c, w, d in [
        (1.00, "#1e3a5f", 1.8, None),
        (0.80, "#142a45", 1.0, "dot"),
        (0.60, "#0f2035", 0.8, "dot"),
        (0.40, "#0a1828", 0.6, "dot"),
        (0.20, "#08121e", 0.4, "dot"),
    ]:
        fig.add_trace(go.Scatter(
            x=[r * math.sin(math.radians(t)) for t in ring_t],
            y=[r * math.cos(math.radians(t)) for t in ring_t],
            mode="lines", line=dict(color=c, width=w, dash=d),
            showlegend=False, hoverinfo="skip",
        ))

    # ── 30° tick marks around outer ring ──────────────────────────────────
    for deg in range(0, 360, 30):
        inner_r, outer_r = (0.92, 1.00) if deg % 90 == 0 else (0.95, 1.00)
        sx, sy = math.sin(math.radians(deg)), math.cos(math.radians(deg))
        tick_color = "#2a5a8a" if deg % 90 == 0 else "#1a3050"
        fig.add_trace(go.Scatter(
            x=[sx * inner_r, sx * outer_r], y=[sy * inner_r, sy * outer_r],
            mode="lines", line=dict(color=tick_color, width=1.5 if deg % 90 == 0 else 0.8),
            showlegend=False, hoverinfo="skip",
        ))

    # ── 10° minor tick marks ──────────────────────────────────────────────
    for deg in range(0, 360, 10):
        if deg % 30 != 0:
            sx, sy = math.sin(math.radians(deg)), math.cos(math.radians(deg))
            fig.add_trace(go.Scatter(
                x=[sx * 0.97, sx * 1.00], y=[sy * 0.97, sy * 1.00],
                mode="lines", line=dict(color="#0f1e30", width=0.5),
                showlegend=False, hoverinfo="skip",
            ))

    # ── Elevation fill (filled disc showing sky coverage) ─────────────────
    fig.add_trace(go.Scatter(
        x=[el_norm * math.sin(math.radians(t)) for t in ring_t] + [0],
        y=[el_norm * math.cos(math.radians(t)) for t in ring_t] + [0],
        fill="toself", fillcolor=el_fill,
        line=dict(color=el_edge, width=1.2),
        showlegend=False, hoverinfo="skip", mode="lines",
    ))

    # ── Crosshair lines ───────────────────────────────────────────────────
    for a in [0, 90, 180, 270]:
        sx, sy = math.sin(math.radians(a)), math.cos(math.radians(a))
        fig.add_trace(go.Scatter(
            x=[0, sx * 0.92], y=[0, sy * 0.92],
            mode="lines", line=dict(color="#0c1828", width=0.8),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Cardinal labels ───────────────────────────────────────────────────
    for angle, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        fig.add_annotation(
            x=math.sin(math.radians(angle)) * 1.14,
            y=math.cos(math.radians(angle)) * 1.14,
            text=f"<b>{label}</b>", showarrow=False,
            font=dict(color="#4a90c8", size=11, family="Share Tech Mono"),
        )

    # ── Degree labels every 30° (skip cardinals) ──────────────────────────
    for deg in range(0, 360, 30):
        if deg % 90 != 0:
            fig.add_annotation(
                x=math.sin(math.radians(deg)) * 1.12,
                y=math.cos(math.radians(deg)) * 1.12,
                text=f"{deg}", showarrow=False,
                font=dict(color="#1e3a55", size=7, family="Share Tech Mono"),
            )

    # ── Azimuth sweep arc (shows the selected azimuth as a glowing arc) ──
    arc_pts = 60
    arc_half = 3  # ±3° glow band
    arc_angles = [az_deg - arc_half + i * (2 * arc_half) / arc_pts for i in range(arc_pts + 1)]
    fig.add_trace(go.Scatter(
        x=[0.88 * math.sin(math.radians(a)) for a in arc_angles],
        y=[0.88 * math.cos(math.radians(a)) for a in arc_angles],
        mode="lines", line=dict(color="rgba(88,166,255,0.6)", width=4),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Needle (main azimuth pointer) ─────────────────────────────────────
    # Glow layer
    fig.add_trace(go.Scatter(
        x=[0, nx * 0.87], y=[0, ny * 0.87],
        mode="lines", line=dict(color="rgba(88,166,255,0.25)", width=6),
        showlegend=False, hoverinfo="skip",
    ))
    # Core needle
    fig.add_trace(go.Scatter(
        x=[0, nx * 0.87], y=[0, ny * 0.87],
        mode="lines", line=dict(color="#58a6ff", width=2.5),
        showlegend=False, hoverinfo="skip",
    ))
    # Arrowhead
    fig.add_annotation(
        ax=nx * 0.4, ay=ny * 0.4, x=nx * 0.92, y=ny * 0.92,
        axref="x", ayref="y", xref="x", yref="y",
        arrowhead=2, arrowsize=1.3, arrowwidth=2.5,
        arrowcolor="#58a6ff", showarrow=True,
    )
    # Tail
    fig.add_trace(go.Scatter(
        x=[0, -nx * 0.25], y=[0, -ny * 0.25],
        mode="lines", line=dict(color="#1a3050", width=1.2),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Sun dot at needle tip ─────────────────────────────────────────────
    sun_r = el_norm * 0.92  # Position along needle scaled by elevation
    fig.add_trace(go.Scatter(
        x=[nx * sun_r], y=[ny * sun_r], mode="markers",
        marker=dict(color="#ffb938", size=10, symbol="circle",
                    line=dict(color="rgba(255,185,56,0.3)", width=4)),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Center hub ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(color="#1a3050", size=10, symbol="circle",
                    line=dict(color="#2a5a8a", width=1.5)),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(color="#58a6ff", size=4, symbol="circle"),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Digital readout ───────────────────────────────────────────────────
    fig.add_annotation(
        x=0, y=-0.42,
        text=f"<b>{az_deg:03d}°</b>",
        showarrow=False,
        font=dict(color="#79c0ff", size=16, family="Share Tech Mono"),
    )

    fig.update_layout(
        xaxis=dict(range=[-1.28, 1.28], visible=False, scaleanchor="y"),
        yaxis=dict(range=[-1.28, 1.28], visible=False),
        plot_bgcolor="#070b14",
        paper_bgcolor="#090e1a",
        margin=dict(l=0, r=0, t=0, b=0),
        height=250,
        showlegend=False,
    )
    return fig


def make_sun_hemisphere_3d(az_deg, el_deg):
    """3D interactive sky hemisphere with sun position, wireframe dome, and horizon plane."""
    import numpy as np

    # Sun position in 3D (spherical → cartesian)
    az_rad = math.radians(az_deg)
    el_rad = math.radians(el_deg)
    sun_x = math.cos(el_rad) * math.sin(az_rad)
    sun_y = math.cos(el_rad) * math.cos(az_rad)
    sun_z = math.sin(el_rad)

    fig = go.Figure()

    # ── Wireframe hemisphere (elevation arcs) ─────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 90)
    for el in [15, 30, 45, 60, 75]:
        r = math.cos(math.radians(el))
        z = math.sin(math.radians(el))
        fig.add_trace(go.Scatter3d(
            x=r * np.cos(theta), y=r * np.sin(theta),
            z=np.full_like(theta, z),
            mode="lines", line=dict(color="rgba(30,53,88,0.4)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Wireframe hemisphere (azimuth meridians) ──────────────────────────
    phi = np.linspace(0, np.pi / 2, 30)
    for az in range(0, 360, 30):
        az_r = math.radians(az)
        fig.add_trace(go.Scatter3d(
            x=np.cos(phi) * math.sin(az_r),
            y=np.cos(phi) * math.cos(az_r),
            z=np.sin(phi),
            mode="lines", line=dict(color="rgba(30,53,88,0.3)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Horizon circle ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta), y=np.sin(theta), z=np.zeros_like(theta),
        mode="lines", line=dict(color="#2a5a8a", width=2.5),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Cardinal markers on horizon ───────────────────────────────────────
    for az, label in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        r_az = math.radians(az)
        fig.add_trace(go.Scatter3d(
            x=[1.08 * math.sin(r_az)], y=[1.08 * math.cos(r_az)], z=[0],
            mode="text", text=[label],
            textfont=dict(color="#4a90c8", size=12, family="Share Tech Mono"),
            showlegend=False, hoverinfo="skip",
        ))

    # ── Sun ray line (origin → sun) ───────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[0, sun_x], y=[0, sun_y], z=[0, sun_z],
        mode="lines", line=dict(color="rgba(255,185,56,0.5)", width=3),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Sun projection on horizon (shadow line) ───────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[sun_x, sun_x], y=[sun_y, sun_y], z=[0, sun_z],
        mode="lines", line=dict(color="rgba(255,185,56,0.2)", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    # Shadow dot on horizon
    fig.add_trace(go.Scatter3d(
        x=[sun_x], y=[sun_y], z=[0],
        mode="markers",
        marker=dict(color="rgba(255,185,56,0.3)", size=5, symbol="circle"),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Sun orb (main, with glow layers) ──────────────────────────────────
    # Outer glow
    fig.add_trace(go.Scatter3d(
        x=[sun_x], y=[sun_y], z=[sun_z],
        mode="markers",
        marker=dict(color="rgba(255,200,60,0.15)", size=22, symbol="circle",
                    line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))
    # Mid glow
    fig.add_trace(go.Scatter3d(
        x=[sun_x], y=[sun_y], z=[sun_z],
        mode="markers",
        marker=dict(color="rgba(255,185,56,0.4)", size=14, symbol="circle",
                    line=dict(width=0)),
        showlegend=False, hoverinfo="skip",
    ))
    # Core sun
    fig.add_trace(go.Scatter3d(
        x=[sun_x], y=[sun_y], z=[sun_z],
        mode="markers+text",
        marker=dict(color="#ffb938", size=9, symbol="circle",
                    line=dict(color="#ffd060", width=2)),
        text=[f"☀ AZ:{az_deg}° EL:{el_deg}°"],
        textposition="top center",
        textfont=dict(color="#ffb938", size=10, family="Share Tech Mono"),
        showlegend=False,
        hovertemplate=f"Sun Position<br>Azimuth: {az_deg}°<br>Elevation: {el_deg}°<extra></extra>",
    ))

    # ── Elevation arc (from horizon to sun along its azimuth) ─────────────
    arc_el = np.linspace(0, el_rad, 30)
    fig.add_trace(go.Scatter3d(
        x=np.cos(arc_el) * math.sin(az_rad),
        y=np.cos(arc_el) * math.cos(az_rad),
        z=np.sin(arc_el),
        mode="lines", line=dict(color="#ffb938", width=3),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Zenith point ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[1.0],
        mode="markers+text",
        marker=dict(color="#2a5a8a", size=3),
        text=["ZENITH"], textposition="top center",
        textfont=dict(color="#1e3a55", size=8, family="Share Tech Mono"),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Camera angle (looking from a nice perspective) ────────────────────
    cam_az = math.radians(az_deg + 150)
    cam_dist = 2.2
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.3, 1.3]),
            yaxis=dict(visible=False, range=[-1.3, 1.3]),
            zaxis=dict(visible=False, range=[-0.1, 1.3]),
            bgcolor="#070b14",
            aspectmode="cube",
            camera=dict(
                eye=dict(
                    x=cam_dist * math.sin(cam_az) * 0.7,
                    y=cam_dist * math.cos(cam_az) * 0.7,
                    z=0.9,
                ),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="#0a1020",
        margin=dict(l=0, r=0, t=30, b=0),
        height=520,
        showlegend=False,
        title=dict(
            text=f"SKY HEMISPHERE — SUN @ AZ {az_deg:03d}° EL {el_deg:02d}°",
            font=dict(color="#5b7ba8", size=11, family="Share Tech Mono"),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

rows = load_atlas_raw()

# ── Mission header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="mission-header">
  <div>
    <div class="mission-title">🌕 LUNARIGN &nbsp;/&nbsp; SUN-SWEEP MATCHER ATLAS</div>
    <div class="mission-sub">SIH 2026 · PS 26166 · ISRO-OHRC/TMC/IIRS · TEAM: SELENOGRAPHERS</div>
  </div>
  <div style="margin-left:auto;text-align:right;">
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:#5b7ba8;letter-spacing:0.08em;">SYSTEM STATUS</div>
    <div><span class="status-dot-green"></span><span style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:#39d353;">ATLAS ONLINE</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

if rows is None:
    st.error("❌  `results/atlas.csv` not found. Run `python run_honest_sweep.py` to generate it.")
    st.stop()

# ── Global stats ──────────────────────────────────────────────────────────────
all_matchers  = sorted({r["matcher"]  for r in rows})
all_protocols = sorted({r["protocol"] for r in rows})
all_factors   = sorted({r["factor"]   for r in rows})

total_cells   = len(rows)
total_success = sum(1 for r in rows if r["success"])
avg_runtime   = sum(r["runtime"] for r in rows) / total_cells if total_cells else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("CELLS EVALUATED",    f"{total_cells:,}")
col2.metric("OVERALL SUCCESS",    f"{total_success / total_cells:.1%}")
col3.metric("MATCHERS TESTED",    str(len(all_matchers)))
col4.metric("GEOMETRY PAIRS",     "5 Δaz × 2 el = 10")
col5.metric("AVG RUNTIME / CELL", f"{avg_runtime:.1f}s")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;"
        "letter-spacing:0.1em;color:#2a6496;padding-bottom:6px;'>◈ FILTER PANEL</div>",
        unsafe_allow_html=True,
    )
    sel_matcher  = st.selectbox("Matcher",  all_matchers,  index=all_matchers.index("minima") if "minima" in all_matchers else 0)
    sel_protocol = st.selectbox("Protocol", all_protocols, index=0)
    sel_factor   = st.selectbox("Factor",   all_factors,   index=0)

    st.divider()

    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;"
        "letter-spacing:0.1em;color:#2a6496;padding-bottom:6px;'>◈ SUN ANGLE QUERY</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='compass-label'>AZIMUTH SELECTOR</div>", unsafe_allow_html=True)
    probe_az = st.slider("Azimuth (°)", 0, 359, 270, step=1, label_visibility="collapsed")
    probe_el = st.slider("Elevation (°)", 2, 85, 40, step=1)

    compass_fig = make_polar_compass(probe_az, probe_el)
    st.plotly_chart(compass_fig, use_container_width=True, config={"displayModeBar": False}, key="sidebar_compass")

    st.markdown(
        f"<div style='font-family:Share Tech Mono,monospace;font-size:0.72rem;"
        f"color:#5b7ba8;margin-top:-8px;text-align:center;'>"
        f"EL &nbsp;<b style='color:#79c0ff;font-size:1rem;'>{probe_el:02d}°</b>"
        f"&nbsp;&nbsp;AZ &nbsp;<b style='color:#79c0ff;font-size:1rem;'>{probe_az:03d}°</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:0.65rem;"
        "color:#1e3558;text-align:center;'>LUNARIGN v1.0 · SIH2026<br>OFFLINE · CPU-ONLY</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  ☀ SUN POSITION  ",
    "  HEATMAPS  ",
    "  ALL MATCHERS  ",
    "  TRUST ROUTER  ",
    "  3-D SURFACE  ",
    "  RAW DATA  ",
])

# ─── Tab 0: Interactive 3D Sun Position Hemisphere ────────────────────────────
with tab0:
    st.caption(
        "Interactive 3-D sky hemisphere showing the sun's position based on your sidebar "
        "azimuth and elevation inputs. Drag to rotate, scroll to zoom. "
        "The golden arc traces the sun's elevation from the horizon. "
        "Dashed line shows the shadow projection on the horizon plane."
    )

    hemi_left, hemi_right = st.columns([3, 1])

    with hemi_left:
        fig_hemi = make_sun_hemisphere_3d(probe_az, probe_el)
        st.plotly_chart(fig_hemi, use_container_width=True, key="sun_hemisphere")

    with hemi_right:
        # Sun angle telemetry panel
        st.markdown(
            f"<div style='background:#0d1525;border:1px solid #1e2e4a;"
            f"border-left:3px solid #ffb938;border-radius:4px;padding:16px;margin-top:20px;'>"
            f"<div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;"
            f"color:#5b7ba8;letter-spacing:0.1em;margin-bottom:12px;'>◈ SUN TELEMETRY</div>"
            f"<div style='font-family:Share Tech Mono,monospace;margin-bottom:10px;'>"
            f"<div style='font-size:0.68rem;color:#5b7ba8;'>AZIMUTH</div>"
            f"<div style='font-size:1.8rem;color:#ffb938;font-weight:700;'>{probe_az:03d}°</div></div>"
            f"<div style='font-family:Share Tech Mono,monospace;margin-bottom:10px;'>"
            f"<div style='font-size:0.68rem;color:#5b7ba8;'>ELEVATION</div>"
            f"<div style='font-size:1.8rem;color:#ffb938;font-weight:700;'>{probe_el:02d}°</div></div>"
            f"<div style='font-family:Share Tech Mono,monospace;margin-bottom:10px;'>"
            f"<div style='font-size:0.68rem;color:#5b7ba8;'>SUN HEIGHT</div>"
            f"<div style='font-size:1.1rem;color:#{'39d353' if probe_el > 20 else 'e3b341' if probe_el > 8 else 'ff4b4b'};'>"
            f"{'HIGH ▲' if probe_el > 45 else 'MEDIUM ■' if probe_el > 15 else 'GRAZING ▼'}</div></div>"
            f"<div style='font-family:Share Tech Mono,monospace;'>"
            f"<div style='font-size:0.68rem;color:#5b7ba8;'>SHADOW RISK</div>"
            f"<div style='font-size:1.1rem;color:#{'39d353' if probe_el > 30 else 'e3b341' if probe_el > 10 else 'ff4b4b'};'>"
            f"{'LOW' if probe_el > 30 else 'MODERATE' if probe_el > 10 else 'CRITICAL'}</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Compact compass below telemetry
        st.markdown(
            "<div style='font-family:Share Tech Mono,monospace;font-size:0.68rem;"
            "color:#2a6496;text-align:center;margin-top:12px;letter-spacing:0.08em;'>◈ BEARING</div>",
            unsafe_allow_html=True,
        )
        small_compass = make_polar_compass(probe_az, probe_el)
        st.plotly_chart(small_compass, use_container_width=True, config={"displayModeBar": False}, key="tab_compass")

# ─── Tab 1: Single-matcher heatmaps ──────────────────────────────────────────
with tab1:
    st.caption(
        "Each cell shows whether the selected matcher produced a correct alignment "
        "(≥20 inliers, ≤3 px grid error) at that sun azimuth and elevation. "
        "Green = reliable, Red = fails or refuses."
    )

    azs, els, z_succ = pivot_success(rows, sel_matcher, sel_protocol, sel_factor)

    if not azs:
        st.warning("No data for this combination of filters.")
    else:
        text_succ = [[f"{v:.0f}%" if v is not None else "–" for v in row_z] for row_z in z_succ]

        fig_s = go.Figure(go.Heatmap(
            x=[f"{a}°" for a in azs],
            y=[f"{e}°" for e in els],
            z=z_succ,
            text=text_succ,
            texttemplate="%{text}",
            colorscale=[[0, "#1a0505"], [0.4, "#5a1a0a"], [0.85, "#1a4a20"], [1.0, "#39d353"]],
            zmin=0, zmax=100,
            colorbar=dict(
                title=dict(text="SUCCESS %", font=dict(color="#5b7ba8", size=10, family="Share Tech Mono")),
                ticksuffix="%",
                tickfont=dict(color="#5b7ba8", family="Share Tech Mono", size=9),
                thickness=12, len=0.8,
            ),
        ))
        fig_s.update_layout(
            title=dict(
                text=f"SUCCESS RATE — {sel_matcher.upper()} / {sel_protocol} / {sel_factor}",
                font=dict(color="#5b7ba8", size=11, family="Share Tech Mono"),
            ),
            xaxis=dict(title="SUN AZIMUTH (°)", tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=10)),
            yaxis=dict(title="SUN ELEVATION (°)", tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=10)),
            plot_bgcolor="#070b14", paper_bgcolor="#0a1020",
            font_color="#c8d8e8", height=380,
            margin=dict(l=60, r=20, t=50, b=60),
        )
        st.plotly_chart(fig_s, use_container_width=True, key="heatmap_success")

        azs2, els2, z_corr = pivot_correct(rows, sel_matcher, sel_protocol, sel_factor)
        text_corr = [[str(v) if v is not None else "–" for v in row_z] for row_z in z_corr]
        fig_c = go.Figure(go.Heatmap(
            x=[f"{a}°" for a in azs2],
            y=[f"{e}°" for e in els2],
            z=z_corr,
            text=text_corr,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar=dict(
                title=dict(text="CORRECT MATCHES", font=dict(color="#5b7ba8", size=10, family="Share Tech Mono")),
                tickfont=dict(color="#5b7ba8", family="Share Tech Mono", size=9),
                thickness=12, len=0.8,
            ),
        ))
        fig_c.update_layout(
            title=dict(
                text=f"CORRECT MATCH COUNT — {sel_matcher.upper()}",
                font=dict(color="#5b7ba8", size=11, family="Share Tech Mono"),
            ),
            xaxis=dict(title="SUN AZIMUTH (°)", tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=10)),
            yaxis=dict(title="SUN ELEVATION (°)", tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=10)),
            plot_bgcolor="#070b14", paper_bgcolor="#0a1020",
            font_color="#c8d8e8", height=380,
            margin=dict(l=60, r=20, t=50, b=60),
        )
        st.plotly_chart(fig_c, use_container_width=True, key="heatmap_correct")


# ─── Tab 2: All-matchers comparison ──────────────────────────────────────────
with tab2:
    st.caption(
        "Compares all 6 matchers side-by-side. "
        "Green (≥85%), amber (40–84%), red (<40%). "
        "Below: per-elevation breakdown shows which matchers degrade at low Sun angles."
    )

    by_matcher = defaultdict(lambda: [0, 0])
    for r in rows:
        m = r["matcher"]
        by_matcher[m][1] += 1
        if r["success"]:
            by_matcher[m][0] += 1

    sorted_m = sorted(by_matcher.keys(), key=lambda m: by_matcher[m][0] / by_matcher[m][1], reverse=True)
    rates    = [by_matcher[m][0] / by_matcher[m][1] * 100 for m in sorted_m]
    bar_colors = [
        "#39d353" if r >= _GREEN_THRESH * 100 else "#e3b341" if r >= _AMBER_THRESH * 100 else "#ff4b4b"
        for r in rates
    ]

    fig_bar = go.Figure(go.Bar(
        x=sorted_m, y=rates,
        marker_color=bar_colors,
        marker_line=dict(color="#1a2540", width=1),
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
        textfont=dict(family="Share Tech Mono", color="#c8d8e8", size=11),
    ))
    fig_bar.add_hline(y=85, line_dash="dot", line_color="#39d353", line_width=1,
                      annotation_text="GREEN 85%",
                      annotation_font=dict(color="#39d353", size=9, family="Share Tech Mono"))
    fig_bar.add_hline(y=40, line_dash="dot", line_color="#e3b341", line_width=1,
                      annotation_text="AMBER 40%",
                      annotation_font=dict(color="#e3b341", size=9, family="Share Tech Mono"))
    fig_bar.update_layout(
        title=dict(text="OVERALL SUCCESS RATE BY MATCHER", font=dict(color="#5b7ba8", size=11, family="Share Tech Mono")),
        yaxis=dict(title="SUCCESS RATE (%)", range=[0, 115], ticksuffix="%",
                   tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=9),
                   gridcolor="#111d30"),
        xaxis=dict(tickfont=dict(family="Share Tech Mono", color="#c8d8e8", size=10)),
        plot_bgcolor="#070b14", paper_bgcolor="#0a1020",
        font_color="#c8d8e8", height=360, showlegend=False, bargap=0.3,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    st.plotly_chart(fig_bar, use_container_width=True, key="bar_matchers")

    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;"
        "letter-spacing:0.08em;color:#2a6496;margin:12px 0 6px;'>◈ MATCHER STATUS CARDS</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(sorted_m))
    for col, m in zip(cols, sorted_m):
        s, t = by_matcher[m]
        rate = s / t
        trust_name, _ = _trust(rate)
        border_c = "#39d353" if trust_name == "GREEN" else "#e3b341" if trust_name == "AMBER" else "#ff4b4b"
        val_c    = border_c
        col.markdown(
            f"<div style='background:#0d1525;border:1px solid #1e2e4a;"
            f"border-top:2px solid {border_c};border-radius:3px;padding:10px;text-align:center;'>"
            f"<div style='font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#5b7ba8;margin-bottom:4px;'>{m.upper()}</div>"
            f"<div style='font-family:Share Tech Mono,monospace;font-size:1.3rem;font-weight:700;color:{val_c};'>{rate:.0%}</div>"
            f"<div style='font-size:0.68rem;color:#5b7ba8;font-family:Share Tech Mono,monospace;'>{trust_name}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;"
        "letter-spacing:0.08em;color:#2a6496;margin:12px 0 6px;'>◈ DEGRADATION BY ELEVATION</div>",
        unsafe_allow_html=True,
    )
    el_matcher_rate = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        el_matcher_rate[r["el"]][r["matcher"]][1] += 1
        if r["success"]:
            el_matcher_rate[r["el"]][r["matcher"]][0] += 1

    els_sorted = sorted(el_matcher_rate.keys())
    fig_el = go.Figure()
    for m in sorted_m:
        y_vals = []
        for el in els_sorted:
            s, t = el_matcher_rate[el][m]
            y_vals.append(s / t * 100 if t > 0 else None)
        fig_el.add_trace(go.Scatter(
            x=[f"{e}°" for e in els_sorted], y=y_vals,
            name=m, mode="lines+markers",
            line=dict(color=_COLOR_MAP.get(m, "#888"), width=2),
            marker=dict(size=7, symbol="circle"),
        ))
    fig_el.add_hline(y=85, line_dash="dot", line_color="#39d353", line_width=1)
    fig_el.add_hline(y=40, line_dash="dot", line_color="#e3b341", line_width=1)
    fig_el.update_layout(
        xaxis=dict(title="SUN ELEVATION (°)", tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=9)),
        yaxis=dict(title="SUCCESS RATE (%)", range=[0, 108], ticksuffix="%",
                   tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=9),
                   gridcolor="#111d30"),
        plot_bgcolor="#070b14", paper_bgcolor="#0a1020",
        font_color="#c8d8e8",
        legend=dict(bgcolor="#0d1525", bordercolor="#1e2e4a",
                    font=dict(family="Share Tech Mono", size=10)),
        height=420,
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_el, use_container_width=True, key="elevation_lines")


# ─── Tab 3: Router probe ──────────────────────────────────────────────────────
with tab3:
    st.caption(
        "The trust router picks the best matcher for a given sun angle based on atlas data. "
        "GREEN = atlas shows ≥85% success — safe to trust. "
        "RED = <40% — refuse to match rather than risk a false positive."
    )

    atlas_rates = build_atlas_rates(rows)
    az_bin = _snap(probe_az, _AZ_GRID)
    el_bin = _snap(probe_el, _EL_GRID)

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown(
            f"<div style='font-family:Share Tech Mono,monospace;'>"
            f"<div style='font-size:0.7rem;color:#5b7ba8;letter-spacing:0.1em;'>◈ QUERY INPUT</div>"
            f"<div style='font-size:1.6rem;color:#79c0ff;margin:6px 0;'>AZ {probe_az:03d}° &nbsp; EL {probe_el:02d}°</div>"
            f"<div style='font-size:0.8rem;color:#2a6496;'>↳ Snapped → AZ <b style='color:#58a6ff;'>{az_bin}°</b>"
            f" &nbsp; EL <b style='color:#58a6ff;'>{el_bin}°</b></div></div>",
            unsafe_allow_html=True,
        )
        router_compass = make_polar_compass(probe_az, probe_el)
        st.plotly_chart(router_compass, use_container_width=True, config={"displayModeBar": False}, key="router_compass")

    with right_col:
        router_rows = []
        recommended = None
        for m in _MATCHER_CASCADE:
            if m not in atlas_rates:
                router_rows.append({"m": m, "rate": None, "trust": "N/A", "status": "not in atlas"})
                continue
            cell_rates = atlas_rates[m]
            rate = cell_rates.get((az_bin, el_bin), None)
            if rate is None:
                el_rates = [v for (a, e), v in cell_rates.items() if e == el_bin]
                rate = sum(el_rates) / len(el_rates) if el_rates else 0.0
            trust_name, _ = _trust(rate)
            status = "◀ RECOMMENDED" if recommended is None and trust_name != "RED" else ""
            if recommended is None and trust_name != "RED":
                recommended = (m, rate, trust_name)
            router_rows.append({"m": m, "rate": rate, "trust": trust_name, "status": status})

        if recommended:
            m_rec, rate_rec, trust_rec = recommended
            protocol_rec = "stretch_2_98" if el_bin <= 5 else "raw"
            trust_color = "#39d353" if trust_rec == "GREEN" else "#e3b341" if trust_rec == "AMBER" else "#ff4b4b"
            st.markdown(f"""
<div class="router-box">
  <div style='font-size:0.7rem;color:#2a6496;letter-spacing:0.1em;margin-bottom:6px;'>◈ ROUTER DECISION</div>
  <div style='font-size:1.1rem;color:#e8f4ff;'><span style='color:#5b7ba8;'>MATCHER:</span> <b style='color:#79c0ff;'>{m_rec}</b></div>
  <div style='font-size:0.9rem;color:#9ab8d4;margin-top:4px;'><span style='color:#5b7ba8;'>PROTOCOL:</span> {protocol_rec}</div>
  <div style='font-size:0.9rem;color:#9ab8d4;margin-top:2px;'><span style='color:#5b7ba8;'>SUCCESS RATE:</span>
    <b style='color:{trust_color};'>{rate_rec:.0%}</b> &nbsp;
    <span style='color:{trust_color};'>● {trust_rec}</span>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="router-box-red">
  <div style='font-size:0.7rem;color:#7a2020;letter-spacing:0.1em;margin-bottom:6px;'>◈ ROUTER DECISION</div>
  <div style='font-size:1rem;color:#ff4b4b;'>● RED LIGHT — NO MATCHER ACHIEVES ≥40% SUCCESS</div>
  <div style='font-size:0.8rem;color:#7a2020;margin-top:4px;'>Refusing to match. False correspondences likely at this sun geometry.</div>
</div>""", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-family:Share Tech Mono,monospace;font-size:0.7rem;"
            "color:#2a6496;margin:14px 0 4px;letter-spacing:0.1em;'>◈ FULL CASCADE</div>",
            unsafe_allow_html=True,
        )
        tbl = (
            "<table style='width:100%;border-collapse:collapse;"
            "font-family:Share Tech Mono,monospace;font-size:0.82rem;'>"
            "<thead><tr>"
            + "".join(
                f"<th style='padding:6px 10px;text-align:left;color:#2a6496;"
                f"border-bottom:1px solid #1a2540;font-weight:400;letter-spacing:0.06em;'>{h}</th>"
                for h in ["MATCHER", "SUCCESS", "TRUST", "STATUS"]
            )
            + "</tr></thead><tbody>"
        )
        for rr in router_rows:
            tn = rr["trust"]
            tc = "#39d353" if tn == "GREEN" else "#e3b341" if tn == "AMBER" else "#ff4b4b" if tn == "RED" else "#5b7ba8"
            rate_str = f"{rr['rate']:.0%}" if rr["rate"] is not None else "—"
            rec_c = "#58a6ff" if rr["status"] else "#5b7ba8"
            tbl += (
                f"<tr style='border-bottom:1px solid #0f1a2e;'>"
                f"<td style='padding:6px 10px;color:#c8d8e8;'>{rr['m']}</td>"
                f"<td style='padding:6px 10px;text-align:center;color:{tc};'>{rate_str}</td>"
                f"<td style='padding:6px 10px;text-align:center;color:{tc};'>● {tn}</td>"
                f"<td style='padding:6px 10px;color:{rec_c};'>{rr['status']}</td>"
                f"</tr>"
            )
        tbl += "</tbody></table>"
        st.markdown(tbl, unsafe_allow_html=True)


# ─── Tab 4: 3-D surface ──────────────────────────────────────────────────────
with tab4:
    st.caption(
        "3-D success-rate landscape over the sun-angle grid. "
        "Height = success rate (%). Rotate with mouse. "
        "Red valleys = dangerous sun geometries. Green peaks = reliable zones. "
        "Dots = real atlas measurement points."
    )

    azs_3d, els_3d, z_3d = pivot_success(rows, sel_matcher, sel_protocol, sel_factor)

    if not azs_3d:
        st.warning("No data for this combination of filters.")
    else:
        import numpy as np

        az_arr = np.array([float(a) for a in azs_3d])
        el_arr = np.array([float(e) for e in els_3d])
        z_arr  = np.array(
            [[v if v is not None else 0 for v in row_z] for row_z in z_3d],
            dtype=float,
        )
        az_grid, el_grid = np.meshgrid(az_arr, el_arr)

        fig_3d = go.Figure(data=[go.Surface(
            x=az_grid, y=el_grid, z=z_arr,
            colorscale=[[0, "#3d0505"], [0.4, "#6b2a0a"], [0.85, "#0d3318"], [1.0, "#39d353"]],
            cmin=0, cmax=100,
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="#79c0ff", project_z=True),
            ),
            colorbar=dict(
                title=dict(text="SUCCESS %", font=dict(color="#5b7ba8", size=10, family="Share Tech Mono")),
                ticksuffix="%",
                tickfont=dict(color="#5b7ba8", family="Share Tech Mono", size=9),
                thickness=12,
            ),
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.3, roughness=0.5),
            opacity=0.92,
        )])

        # Scatter dots at actual measurement points
        sx, sy, sz = [], [], []
        for el_i, el_v in enumerate(els_3d):
            for az_i, az_v in enumerate(azs_3d):
                v = z_3d[el_i][az_i]
                if v is not None:
                    sx.append(float(az_v))
                    sy.append(float(el_v))
                    sz.append(v)

        fig_3d.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz,
            mode="markers",
            marker=dict(
                size=5,
                color=sz,
                colorscale=[[0, "#ff4b4b"], [1, "#39d353"]],
                cmin=0, cmax=100,
                symbol="circle",
                line=dict(color="#1a2540", width=1),
            ),
            showlegend=False,
            hovertemplate="AZ: %{x}°<br>EL: %{y}°<br>Success: %{z:.0f}%<extra></extra>",
        ))

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(
                    title="AZIMUTH (°)",
                    tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=9),
                    gridcolor="#111d30", backgroundcolor="#070b14", zerolinecolor="#1a2540",
                ),
                yaxis=dict(
                    title="ELEVATION (°)",
                    tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=9),
                    gridcolor="#111d30", backgroundcolor="#070b14", zerolinecolor="#1a2540",
                ),
                zaxis=dict(
                    title="SUCCESS %", ticksuffix="%", range=[0, 100],
                    tickfont=dict(family="Share Tech Mono", color="#5b7ba8", size=9),
                    gridcolor="#111d30", backgroundcolor="#070b14",
                ),
                bgcolor="#070b14",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=580,
            paper_bgcolor="#0a1020",
            font_color="#c8d8e8",
            title=dict(
                text=f"3-D SUCCESS LANDSCAPE — {sel_matcher.upper()}",
                font=dict(color="#5b7ba8", size=11, family="Share Tech Mono"),
            ),
        )
        st.plotly_chart(fig_3d, use_container_width=True, key="surface_3d")


# ─── Tab 5: Raw data ──────────────────────────────────────────────────────────
with tab5:
    st.caption(
        "Every row from results/atlas.csv for the selected matcher/protocol/factor. "
        "✓ = success, ✗ = failed. These are the raw numbers behind every chart above."
    )

    filt_rows = [r for r in rows
                 if r["matcher"]  == sel_matcher
                 and r["protocol"] == sel_protocol
                 and r["factor"]   == sel_factor]

    display_cols = ["matcher", "protocol", "factor", "az", "el",
                    "matches", "correct", "grid_err", "success", "runtime"]

    th = (
        "<table style='border-collapse:collapse;width:100%;"
        "font-family:Share Tech Mono,monospace;font-size:0.78rem;'>"
        "<thead><tr>"
        + "".join(
            f"<th style='padding:6px 12px;text-align:left;color:#2a6496;"
            f"border-bottom:1px solid #1a2540;letter-spacing:0.06em;font-weight:400;'>{c.upper()}</th>"
            for c in display_cols
        )
        + "</tr></thead><tbody>"
    )
    td = "".join(
        "<tr style='border-bottom:1px solid #0f1a2e;'>"
        + "".join(
            f"<td style='padding:5px 12px;"
            f"color:{'#39d353' if col == 'success' and r.get(col) else '#ff4b4b' if col == 'success' else '#9ab8d4'};'>"
            f"{('✓' if r[col] else '✗') if col == 'success' else (f'{r[col]:.3f}' if isinstance(r.get(col), float) and col == 'grid_err' else f'{r[col]:.1f}s' if col == 'runtime' else r.get(col, ''))}"
            f"</td>"
            for col in display_cols
        )
        + "</tr>"
        for r in filt_rows[:500]
    )
    st.markdown(
        f"<div style='overflow-x:auto;background:#0a1020;border:1px solid #1a2540;"
        f"border-radius:4px;padding:4px;'>{th}{td}</tbody></table></div>"
        + (
            f"<p style='color:#5b7ba8;font-size:0.72rem;font-family:Share Tech Mono,monospace;"
            f"margin-top:6px;'>Showing first 500 of {len(filt_rows)} rows.</p>"
            if len(filt_rows) > 500 else ""
        ),
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#1e3558;font-size:0.72rem;"
    "font-family:Share Tech Mono,monospace;letter-spacing:0.06em;'>"
    "LUNARIGN · SIH 2026 · PS 26166 · "
    "PIPELINE: LOLA DEM → SUN-SWEEP ATLAS → TRUST ROUTER → IMAGE CORRESPONDENCE"
    "</p>",
    unsafe_allow_html=True,
)

