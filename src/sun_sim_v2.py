"""
sun_sim_v2.py - lunar illumination renderer for the Sun-Sweep Atlas.

Why this replaces a plain matplotlib hillshade:
  * Uses a lunar-style reflectance model (Lommel-Seeliger or Lambert), not a display hillshade.
  * Casts real shadows (horizon test toward the sun). At low Sun these dominate lunar images.
  * No per-image contrast stretch, so brightness differences between sun angles are preserved.
  * The horizon map is computed once per AZIMUTH and reused for every ELEVATION (cheap sweeps).

Conventions
  dem[row, col] in metres; row increases SOUTH, col increases EAST.
  Azimuth: degrees clockwise from north. Elevation: degrees above the local horizon.
  Camera is assumed nadir-looking (view direction = local up).

Limits (say these on your slides): DEM-only renders have no albedo texture and no
roughness below DEM resolution. This is a shading-geometry stress test, not a photo.
"""
import numpy as np


def sun_vector(az_deg, el_deg):
    """Unit vector toward the Sun as (east, north, up)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])


def horizon_map(dem, pixel_m, az_deg, max_dist_px=256):
    """Highest terrain horizon angle (radians) seen from each pixel toward azimuth az_deg.
    A pixel is in cast shadow when this angle exceeds the sun elevation."""
    dem = dem.astype(np.float32)
    rows, cols = dem.shape
    az = np.radians(az_deg)
    d_col, d_row = np.sin(az), -np.cos(az)  # one step toward the Sun: east = +col, north = -row
    pad_n = max_dist_px + 1
    pad = np.pad(dem, pad_n, mode="constant", constant_values=np.nan)  # NaN = off-tile, ignored
    best = np.full(dem.shape, -np.pi / 2, dtype=np.float32)
    seen = set()
    for k in range(1, max_dist_px + 1):
        dr, dc = int(round(k * d_row)), int(round(k * d_col))
        if (dr, dc) == (0, 0) or (dr, dc) in seen:
            continue
        seen.add((dr, dc))
        shifted = pad[pad_n + dr: pad_n + dr + rows, pad_n + dc: pad_n + dc + cols]
        ang = np.arctan2(shifted - dem, np.hypot(dr, dc) * pixel_m)
        best = np.fmax(best, ang)  # fmax skips NaN
    return best


def render(dem, pixel_m, az_deg, el_deg, model="lommel_seeliger", horizon=None, albedo=1.0):
    """Return float32 reflectance in [0, ~1]. Pass horizon=horizon_map(...) to include cast shadows."""
    dem = dem.astype(np.float32)
    gy, gx = np.gradient(dem, pixel_m)          # gy = dz/d(row) (south +), gx = dz/d(col) (east +)
    nx, ny, nz = -gx, gy, np.ones_like(gx)      # surface normal as (east, north, up)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    s = sun_vector(az_deg, el_deg)
    mu0 = np.clip(nx * s[0] + ny * s[1] + nz * s[2], 0.0, None)  # cos(incidence)
    mu = nz                                                       # cos(emission), nadir camera
    if model == "lambert":
        refl = mu0
    elif model == "lommel_seeliger":
        refl = 2.0 * mu0 / (mu0 + mu + 1e-6)    # scaled so flat terrain under an overhead Sun = 1
    else:
        raise ValueError("model must be 'lambert' or 'lommel_seeliger'")
    lit = np.ones(dem.shape, dtype=bool) if horizon is None else horizon < np.radians(el_deg)
    return (albedo * refl * lit).astype(np.float32)


def to_uint8(img, gain=1.0):
    """Fixed-gain conversion. Do NOT stretch per image here; treat any normalisation as a
    separate, tested 'protocol' choice at the matcher input."""
    return (np.clip(img * gain, 0.0, 1.0) * 255.0).astype(np.uint8)


def _self_test():
    # 1) flat terrain: Lambert reflectance must equal sin(elevation)
    flat = np.zeros((64, 64), np.float32)
    out = render(flat, 20.0, 45.0, 30.0, model="lambert")
    assert np.allclose(out, np.sin(np.radians(30.0)), atol=1e-3), "flat-terrain check failed"

    # 2) a tall bump must cast a shadow on the side AWAY from the Sun, only when the Sun is low
    n = 200
    yy, xx = np.mgrid[0:n, 0:n]
    dem = 400.0 * np.exp(-(((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 6.0 ** 2)))  # ~120 m sigma at 20 m/px
    px = 20.0
    h_east = horizon_map(dem, px, az_deg=90.0)                    # Sun in the east
    low = render(dem, px, 90.0, 5.0, horizon=h_east)
    high = render(dem, px, 90.0, 60.0, horizon=h_east)
    west_of_bump = (slice(98, 103), slice(80, 92))                # west side = away from the Sun
    east_of_bump = (slice(98, 103), slice(110, 120))
    assert low[west_of_bump].max() == 0.0, "expected cast shadow west of bump at 5 deg"
    assert low[east_of_bump].mean() > 0.0, "sun-facing side should be lit"
    assert high[west_of_bump].mean() > 0.0, "no cast shadow expected at 60 deg"

    # 3) darker and more shadowed as the Sun gets lower
    lit_fracs = [float((horizon_map(dem, px, 90.0) < np.radians(e)).mean()) for e in (3, 10, 30, 60)]
    assert lit_fracs == sorted(lit_fracs), "lit fraction should rise with sun elevation"

    # 4) different azimuths must give visibly different images
    a = render(dem, px, 0.0, 15.0, horizon=horizon_map(dem, px, 0.0))
    b = render(dem, px, 180.0, 15.0, horizon=horizon_map(dem, px, 180.0))
    assert np.abs(a - b).mean() > 0.01, "azimuth change should change the image"
    print("sun_sim_v2 self-test passed. lit fraction vs elevation (3,10,30,60 deg):",
          [round(f, 3) for f in lit_fracs])


if __name__ == "__main__":
    _self_test()
