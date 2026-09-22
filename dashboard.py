"""
dashboard.py — LunarAlign Matcher Atlas Dashboard (SIH 2026)
═════════════════════════════════════════════════════════════
Visualises the Sun-Sweep Atlas (results/atlas.csv) with:
  • Per-matcher success-rate heatmaps (az × el)
  • Correct-match count heatmaps
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
    page_title="LunarAlign — Matcher Atlas",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background for the whole app */
    .stApp { background: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 1.5rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* Metric cards */
    [data-testid="stMetric"] { background: #161b22; border: 1px solid #21262d;
                                border-radius: 8px; padding: 12px; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }

    /* Trust badges */
    .trust-green { color:#2ea043; font-weight:700; font-size:1.1rem; }
    .trust-amber { color:#d29922; font-weight:700; font-size:1.1rem; }
    .trust-red   { color:#f85149; font-weight:700; font-size:1.1rem; }

    /* Headers */
    h1 { color: #e6edf3 !important; }
    h2, h3 { color: #c9d1d9 !important; }

    /* Divider */
    hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (pure stdlib — no pandas)
# ══════════════════════════════════════════════════════════════════════════════

ATLAS_PATH = os.path.join(_here, "results", "atlas.csv")

_GREEN_THRESH = 0.85
_AMBER_THRESH = 0.40
_AZ_GRID = [0, 45, 90, 135, 180, 225, 270, 315]
_EL_GRID = [3, 5, 10, 20, 30, 45, 60, 80]
_MATCHER_CASCADE = ["minima", "superpoint-lightglue", "loftr", "sift", "orb", "aliked-lightglue"]


@st.cache_data(show_spinner="Loading atlas…")
def load_atlas_raw():
    """Return list-of-dicts from the CSV (stdlib only)."""
    if not os.path.exists(ATLAS_PATH):
        return None
    rows = []
    with open(ATLAS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            row["az"] = int(row["az"])
            row["el"] = int(row["el"])
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
        return "GREEN", "🟢"
    if rate >= _AMBER_THRESH:
        return "AMBER", "🟡"
    return "RED", "🔴"


def _snap(value, grid):
    return min(grid, key=lambda g: abs(g - value))


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def pivot_success(rows, matcher, protocol, factor):
    """Return (az_list, el_list, z 2-D list [el][az]) for success-rate heatmap."""
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
    """Return (az_list, el_list, z) for mean-correct-matches heatmap."""
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
    """Build { matcher -> { (az, el) -> rate } } for the router panel."""
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        m = r["matcher"]
        acc[m][(r["az"], r["el"])][1] += 1
        if r["success"]:
            acc[m][(r["az"], r["el"])][0] += 1
    return {m: {cell: s / t for cell, (s, t) in cells.items() if t > 0}
            for m, cells in acc.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

rows = load_atlas_raw()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌕 LunarAlign — Sun-Sweep Matcher Atlas")
st.markdown(
    "Evaluating **6 image-matching algorithms** across a 8×8 sun-angle grid "
    "rendered on synthetic lunar terrain — the winning differentiator for SIH 2026."
)

if rows is None:
    st.error("❌  `results/atlas.csv` not found. Run `python -m src.step3_sweep` to generate it.")
    st.stop()

# ── Global stats ──────────────────────────────────────────────────────────────
all_matchers = sorted({r["matcher"] for r in rows})
all_protocols = sorted({r["protocol"] for r in rows})
all_factors   = sorted({r["factor"]   for r in rows})

total_cells = len(rows)
total_success = sum(1 for r in rows if r["success"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total cells evaluated", f"{total_cells:,}")
col2.metric("Overall success rate", f"{total_success/total_cells:.1%}")
col3.metric("Matchers tested", str(len(all_matchers)))
col4.metric("Sun-angle grid", f"{len(_AZ_GRID)} az × {len(_EL_GRID)} el")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    sel_matcher  = st.selectbox("Matcher",  all_matchers,  index=all_matchers.index("minima") if "minima" in all_matchers else 0)
    sel_protocol = st.selectbox("Protocol", all_protocols, index=0)
    sel_factor   = st.selectbox("Factor",   all_factors,   index=0)

    st.divider()
    st.markdown("### 🌞 Router Probe")
    probe_az = st.slider("Query azimuth (°)",   0, 359, 270, step=1)
    probe_el = st.slider("Query elevation (°)",  2,  85,  40, step=1)

# ══════════════════════════════════════════════════════════════════════════════
#  HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(["📊 Heatmaps", "🌡️ All Matchers", "🤖 Router", "📋 Raw Data"])

# ─── Tab 1: Single-matcher heatmaps ──────────────────────────────────────────
with tab1:
    st.subheader(f"Success Rate (%)  —  {sel_matcher.upper()}  |  {sel_protocol}  |  {sel_factor}")

    azs, els, z_succ = pivot_success(rows, sel_matcher, sel_protocol, sel_factor)
    if not azs:
        st.warning("No data for this combination of filters.")
    else:
        # Text annotations: rate + ✓/✗
        text_succ = []
        for row_z in z_succ:
            text_succ.append([f"{v:.0f}%" if v is not None else "–" for v in row_z])

        fig_s = go.Figure(go.Heatmap(
            x=[str(a) for a in azs],
            y=[str(e) for e in els],
            z=z_succ,
            text=text_succ,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0, zmax=100,
            colorbar=dict(title="Success %", ticksuffix="%"),
        ))
        fig_s.update_layout(
            xaxis_title="Sun Azimuth (°)",
            yaxis_title="Sun Elevation (°)",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#e6edf3",
            height=420,
        )
        st.plotly_chart(fig_s, use_container_width=True)

        st.subheader(f"Mean Correct Matches  —  {sel_matcher.upper()}")
        azs2, els2, z_corr = pivot_correct(rows, sel_matcher, sel_protocol, sel_factor)
        text_corr = [[str(v) if v is not None else "–" for v in row_z] for row_z in z_corr]
        fig_c = go.Figure(go.Heatmap(
            x=[str(a) for a in azs2],
            y=[str(e) for e in els2],
            z=z_corr,
            text=text_corr,
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar=dict(title="Correct matches"),
        ))
        fig_c.update_layout(
            xaxis_title="Sun Azimuth (°)",
            yaxis_title="Sun Elevation (°)",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font_color="#e6edf3",
            height=420,
        )
        st.plotly_chart(fig_c, use_container_width=True)

# ─── Tab 2: All-matchers comparison bar ──────────────────────────────────────
with tab2:
    st.subheader("Overall Success Rate by Matcher")

    # Success by matcher (all protocols, all factors)
    by_matcher = defaultdict(lambda: [0, 0])
    for r in rows:
        m = r["matcher"]
        by_matcher[m][1] += 1
        if r["success"]:
            by_matcher[m][0] += 1

    sorted_m = sorted(by_matcher.keys(), key=lambda m: by_matcher[m][0] / by_matcher[m][1], reverse=True)
    rates = [by_matcher[m][0] / by_matcher[m][1] * 100 for m in sorted_m]
    colors = ["#2ea043" if r >= _GREEN_THRESH * 100 else "#d29922" if r >= _AMBER_THRESH * 100 else "#f85149"
              for r in rates]

    fig_bar = go.Figure(go.Bar(
        x=sorted_m,
        y=rates,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
    ))
    fig_bar.update_layout(
        yaxis=dict(title="Success Rate (%)", range=[0, 108], ticksuffix="%"),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Per-matcher summary cards
    st.subheader("Matcher Summary Cards")
    cols = st.columns(len(sorted_m))
    for col, m in zip(cols, sorted_m):
        s, t = by_matcher[m]
        rate = s / t
        trust_name, trust_icon = _trust(rate)
        css_class = f"trust-{trust_name.lower()}"
        col.markdown(f"**{m}**")
        col.markdown(f"<span class='{css_class}'>{trust_icon} {trust_name}</span>", unsafe_allow_html=True)
        col.metric("Success", f"{rate:.1%}", f"{s}/{t} cells")

    st.divider()

    # Per-elevation breakdown
    st.subheader("Success Rate by Elevation (all matchers)")
    el_matcher_rate = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        el_matcher_rate[r["el"]][r["matcher"]][1] += 1
        if r["success"]:
            el_matcher_rate[r["el"]][r["matcher"]][0] += 1

    els_sorted = sorted(el_matcher_rate.keys())
    fig_el = go.Figure()
    color_map = {
        "minima": "#2ea043",
        "superpoint-lightglue": "#388bfd",
        "loftr": "#bc8cff",
        "sift": "#f0883e",
        "orb": "#f85149",
        "aliked-lightglue": "#79c0ff",
    }
    for m in sorted_m:
        y_vals = []
        for el in els_sorted:
            s, t = el_matcher_rate[el][m]
            y_vals.append(s / t * 100 if t > 0 else None)
        fig_el.add_trace(go.Scatter(
            x=[str(e) for e in els_sorted],
            y=y_vals,
            name=m,
            mode="lines+markers",
            line=dict(color=color_map.get(m, "#888"), width=2),
            marker=dict(size=8),
        ))
    fig_el.add_hline(y=_GREEN_THRESH * 100, line_dash="dash", line_color="#2ea043",
                     annotation_text="GREEN threshold", annotation_font_color="#2ea043")
    fig_el.add_hline(y=_AMBER_THRESH * 100, line_dash="dash", line_color="#d29922",
                     annotation_text="AMBER threshold", annotation_font_color="#d29922")
    fig_el.update_layout(
        xaxis_title="Sun Elevation (°)",
        yaxis=dict(title="Success Rate (%)", range=[0, 108], ticksuffix="%"),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        legend=dict(bgcolor="#161b22", bordercolor="#21262d"),
        height=450,
    )
    st.plotly_chart(fig_el, use_container_width=True)

# ─── Tab 3: Router probe ──────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 Atlas-Driven Trust Router")
    st.markdown(
        "Simulates the router decision for any sun-angle query. "
        "Use the sidebar sliders to set the query azimuth and elevation."
    )

    atlas_rates = build_atlas_rates(rows)

    az_bin = _snap(probe_az, _AZ_GRID)
    el_bin = _snap(probe_el, _EL_GRID)

    st.markdown(f"**Query:** az = {probe_az}° el = {probe_el}°  →  "
                f"Snapped to atlas cell az = **{az_bin}°** el = **{el_bin}°**")

    # Walk cascade
    router_rows = []
    recommended = None
    for m in _MATCHER_CASCADE:
        if m not in atlas_rates:
            router_rows.append({"Matcher": m, "Success Rate": "—", "Trust": "N/A", "Status": "not in atlas"})
            continue
        cell_rates = atlas_rates[m]
        rate = cell_rates.get((az_bin, el_bin), None)
        if rate is None:
            el_rates = [v for (a, e), v in cell_rates.items() if e == el_bin]
            rate = sum(el_rates) / len(el_rates) if el_rates else 0.0
        trust_name, trust_icon = _trust(rate)
        status = "✅ RECOMMENDED" if recommended is None and trust_name != "RED" else ""
        if recommended is None and trust_name != "RED":
            recommended = (m, rate, trust_name, trust_icon)
        router_rows.append({
            "Matcher": m,
            "Success Rate": f"{rate:.0%}",
            "Trust": f"{trust_icon} {trust_name}",
            "Status": status,
        })

    if recommended:
        m_rec, rate_rec, trust_rec, icon_rec = recommended
        css = f"trust-{trust_rec.lower()}"
        protocol_rec = "stretch_2_98" if el_bin <= 5 else "raw"
        st.markdown(f"""
<div style='background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px;margin:12px 0;'>
  <div style='font-size:1.3rem;font-weight:700;'>🤖 Router Decision</div>
  <div style='margin-top:8px;'>
    <b>Matcher:</b> <code>{m_rec}</code> &nbsp;|&nbsp;
    <b>Protocol:</b> <code>{protocol_rec}</code> &nbsp;|&nbsp;
    <b>Trust:</b> <span class='{css}'>{icon_rec} {trust_rec}</span><br/>
    <b>Estimated success rate:</b> {rate_rec:.0%}
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style='background:#1a0a0a;border:1px solid #f85149;border-radius:10px;padding:16px;margin:12px 0;'>
  <span class='trust-red'>🔴 RED LIGHT — No matcher achieves ≥40% success at this sun angle.<br/>
  Refusing to match to avoid false correspondences.</span>
</div>
""", unsafe_allow_html=True)

    # Show cascade table
    st.markdown("**Full cascade:**")
    header_cols = st.columns([2, 1.5, 1.5, 2])
    for h, col in zip(["Matcher", "Success Rate", "Trust", "Status"], header_cols):
        col.markdown(f"**{h}**")
    for rr in router_rows:
        c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 2])
        c1.markdown(f"`{rr['Matcher']}`")
        c2.markdown(rr["Success Rate"])
        c3.markdown(rr["Trust"])
        c4.markdown(rr["Status"])

# ─── Tab 4: Raw data ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("Raw Atlas Data")
    filt_rows = [r for r in rows
                 if r["matcher"] == sel_matcher
                 and r["protocol"] == sel_protocol
                 and r["factor"] == sel_factor]

    display_cols = ["matcher", "protocol", "factor", "az", "el",
                    "matches", "correct", "grid_err", "success", "runtime"]

    # Build a simple HTML table
    html_rows = "".join(
        "<tr>" + "".join(
            f"<td style='padding:4px 10px;border-bottom:1px solid #21262d;"
            f"color:{'#2ea043' if col=='success' and r.get(col) else '#f85149' if col=='success' else '#e6edf3'};'>"
            f"{('✓' if r[col] else '✗') if col=='success' else (f'{r[col]:.3f}' if isinstance(r.get(col), float) and col not in ('runtime',) else r.get(col, ''))}</td>"
            for col in display_cols
        ) + "</tr>"
        for r in filt_rows[:500]
    )
    html_header = "<tr>" + "".join(
        f"<th style='padding:4px 10px;text-align:left;color:#8b949e;border-bottom:2px solid #21262d;'>{c}</th>"
        for c in display_cols
    ) + "</tr>"
    st.markdown(
        f"<div style='overflow-x:auto;'><table style='border-collapse:collapse;width:100%;font-size:0.82rem;'>"
        f"<thead>{html_header}</thead><tbody>{html_rows}</tbody></table></div>"
        + (f"<p style='color:#8b949e;font-size:0.8rem;margin-top:6px;'>Showing first 500 of {len(filt_rows)} rows.</p>" if len(filt_rows) > 500 else ""),
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#484f58;font-size:0.8rem;'>"
    "LunarAlign · SIH 2026 · PS 26166 · "
    "Pipeline: LOLA DEM → Sun-Sweep Atlas → Trust Router → Image Correspondence"
    "</p>",
    unsafe_allow_html=True,
)
