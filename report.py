"""
report.py — Step 5: Verbatim, honest sweep report for PPT.
═══════════════════════════════════════════════════════════
Reads results/atlas.csv (produced by step3_sweep.py) and prints:
  1. The full table: all matchers × all geometries, no rounding-favorable
  2. The exact NCC value at 180° flip  
  3. The git commit hash of the baseline
  4. The SHA-256 hashes of the frozen simulator files
  5. A RED-trust example (if any), or explicit statement that none exist

Nothing here is cherry-picked: every cell is printed, failures included.
"""
import csv, os, sys, hashlib

REPO = os.path.dirname(os.path.abspath(__file__))
ATLAS = os.path.join(REPO, 'results', 'atlas.csv')

def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_head_commit():
    try:
        head_path = os.path.join(REPO, '.git', 'HEAD')
        with open(head_path) as f:
            head = f.read().strip()
        if head.startswith('ref:'):
            ref = head.split(' ')[1]
            ref_path = os.path.join(REPO, '.git', ref)
            with open(ref_path) as f:
                return f.read().strip()
        return head
    except Exception as e:
        return f'(error reading HEAD: {e})'

# ── Load atlas ─────────────────────────────────────────────────────────────────
rows = []
with open(ATLAS) as f:
    for row in csv.DictReader(f):
        rows.append(row)

matchers = ['sift', 'orb', 'loftr', 'superpoint-lightglue', 'aliked-lightglue', 'minima']
delta_azs = [0, 45, 90, 135, 180]
elevations = [10, 40]

print('=' * 80)
print('LUNARALIGN — HONEST SWEEP REPORT (Step 5, verbatim)')
print('SIH 2026, PS 26166')
print('=' * 80)

# ── Section A: Provenance ──────────────────────────────────────────────────────
print('\nA. PROVENANCE')
print('-' * 40)
commit = get_head_commit()
print(f'  Git HEAD commit: {commit}')
print()
print('  Simulator SHA-256 hashes:')
for relpath in ['src/sun_sim_v2.py', 'src/synth_dem.py']:
    fullpath = os.path.join(REPO, relpath)
    h = sha256(fullpath)
    print(f'    {relpath}')
    print(f'      {h}')
print()
print('  SIMULATOR_HASHES.txt:')
with open(os.path.join(REPO, 'SIMULATOR_HASHES.txt')) as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            print(f'    {line.rstrip()}')
print()
print('  Success criterion (from config.yaml, committed before any sweep):')
print('    grid_error < 3.0 px  AND  n_correct >= 20  (10x10 reprojection grid)')
print()
print(f'  Total rows in atlas.csv (excl. header): {len(rows)}')
print(f'  Expected (5 Δaz × 2 el × 6 matchers × 2 proto × 2 factors): 240')
if len(rows) != 240:
    print(f'  WARNING: row count mismatch ({len(rows)} != 240)')

# ── Section B: NCC at 180° flip ───────────────────────────────────────────────
print('\nB. NCC AT 180° FLIP (Δaz=180, el=40, reference az=270)')
print('-' * 40)
# The 180° flip means Δaz=180, abs_az=90, el=40
# All rows at that cell will have same ncc_vs_ref (computed once per render)
ncc_rows = [r for r in rows if r['delta_az']=='180' and r['el']=='40']
if ncc_rows:
    ncc_val = float(ncc_rows[0]['ncc_vs_ref'])
    print(f'  NCC (reference az=270,el=40 vs target az=90,el=40): {ncc_val:+.6f}')
    print(f'  Interpretation: {"Negative = shadow inversion (shadows on opposite sides)" if ncc_val < 0 else "Positive = similar illumination pattern"}')
else:
    print('  WARNING: no rows found for Δaz=180, el=40')

# ── Section C: Full table ─────────────────────────────────────────────────────
print('\nC. FULL SWEEP TABLE — ALL CELLS, ALL MATCHERS')
print('   Factor: A_identity (same geometry), Protocol: raw')
print('   (Failures shown explicitly — nothing omitted)')
print('-' * 80)

# Index by (matcher, delta_az, el, protocol, factor)
idx = {}
for row in rows:
    key = (row['matcher'], int(row['delta_az']), int(row['el']), row['protocol'], row['factor'])
    idx[key] = row

# Print compact table: one row per matcher, columns = Δaz,el combos
combos = [(daz, el) for el in elevations for daz in delta_azs]
header_parts = [f'Δaz={d:>3}° el={e:>2}°' for d, e in combos]
print(f'  {"Matcher":<25} | ' + ' | '.join(f'{h}' for h in header_parts))
sep = '-' * (25 + 3 + len(header_parts) * (len(header_parts[0]) + 3))
print(f'  {sep}')

for m in matchers:
    cells = []
    for daz, el in combos:
        key = (m, daz, el, 'raw', 'A_identity')
        if key in idx:
            r = idx[key]
            succ = r['success'] == 'True'
            n_c = int(r['correct'])
            n_m = int(r['matches'])
            gerr = float(r['grid_err']) if r['grid_err'] != 'inf' else float('inf')
            if succ:
                cells.append(f'OK({n_c:>4}c,{gerr:>5.2f}e)')
            else:
                cells.append(f'FAIL({n_c:>3}c,{"inf" if gerr==float("inf") else f"{gerr:>4.1f}"}e)')
        else:
            cells.append('MISSING'.center(14))
    print(f'  {m:<25} | ' + ' | '.join(f'{c}' for c in cells))

print()
print('  Key: OK/FAIL(correct_matches, grid_error_px)')

# ── Section D: Complete per-row data ─────────────────────────────────────────
print('\nD. RAW DATA — EVERY ROW (no cherry-picking)')
print('-' * 80)
print(f'  {"matcher":<25} {"proto":<15} {"factor":<12} {"Δaz":>4} {"az":>4} {"el":>3} {"m":>5} {"c":>5} {"gerr":>8} {"succ":<6} {"ncc":>8} {"rt":>6}')
print(f'  {"-"*110}')
for r in rows:
    gerr_str = r['grid_err'] if r['grid_err'] in ('inf',) else f'{float(r["grid_err"]):.4f}'
    flag = '  <--FAIL' if r['success'] == 'False' else ''
    print(f'  {r["matcher"]:<25} {r["protocol"]:<15} {r["factor"]:<12} {r["delta_az"]:>4} {r["abs_az"]:>4} {r["el"]:>3} '
          f'{r["matches"]:>5} {r["correct"]:>5} {gerr_str:>8} {r["success"]:<6} '
          f'{float(r["ncc_vs_ref"]):>+8.4f} {float(r["runtime"]):>6.2f}{flag}')

# ── Section E: Router trust analysis ─────────────────────────────────────────
print('\nE. ROUTER TRUST ANALYSIS — what would the system do?')
print('-' * 40)
print('   Based on measured success rates, would the router ever return RED?')
print()

from collections import defaultdict
cell_stats = defaultdict(lambda: [0,0])
for r in rows:
    if r['factor'] == 'A_identity' and r['protocol'] == 'raw':
        key = (r['matcher'], int(r['delta_az']), int(r['el']))
        cell_stats[key][1] += 1
        if r['success'] == 'True':
            cell_stats[key][0] += 1

# Per (delta_az, el) find best matcher's rate
print(f'  {"Δaz":>4} {"el":>3}  {"Best matcher":<25} {"rate":>6}  {"Trust":<8}')
print(f'  {"-"*55}')
any_red = False
for daz in delta_azs:
    for el in elevations:
        best_m = None
        best_rate = -1
        for m in matchers:
            key = (m, daz, el)
            if key in cell_stats:
                s, t = cell_stats[key]
                rate = s/t if t > 0 else 0
                if rate > best_rate:
                    best_rate = rate
                    best_m = m
        trust = 'GREEN' if best_rate >= 0.85 else 'AMBER' if best_rate >= 0.40 else 'RED'
        if trust == 'RED':
            any_red = True
        flag = '  <-- ROUTER REFUSES' if trust == 'RED' else ''
        print(f'  {daz:>4} {el:>3}  {best_m if best_m else "none":<25} {best_rate:>5.0%}  {trust:<8}{flag}')

print()
if not any_red:
    print('  RESULT: NO RED CELLS in this sweep.')
    print('  The router never refuses. This contradicts the "RED light" claim in the PPT.')
    print('  Explanation: The PPT grid (Δaz=[0..180], el=[10,40]) does not include any')
    print('  geometry hard enough to defeat the best available matcher (MINIMA/SPL).')
    print('  To obtain a real RED cell, you would need to test:')
    print('    - el <= 5° (grazing sun), OR')
    print('    - el >= 80° (overhead sun, shadow collapse), OR')
    print('    - Add very large DEMs with real albedo variation.')
    print('  The claim "system refuses at extreme angles" remains UNVALIDATED by this sweep.')
else:
    print('  RESULT: RED cells found — router will refuse at the flagged geometries.')

print()
print('=' * 80)
print('END OF HONEST REPORT')
print('=' * 80)
