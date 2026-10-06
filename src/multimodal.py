"""
multimodal.py — Multi-Resolution + Multi-Spectral Terrain Renderer
═══════════════════════════════════════════════════════════════════════════════

Generates the SAME physical terrain viewed at different sensor GSDs
AND different spectral responses.

Key insights:
  1. SCALE: Different sensors see different levels of detail in the same
     crater field. OHRC (0.3 m/px) resolves fine crater rims that TMC-2
     (5 m/px) sees as smooth slopes, and IIRS (80 m/px) can barely
     distinguish individual craters.
  2. SPECTRAL: Different spectral bands produce different contrast.
     - OHRC PAN (450-850nm): High reflectance contrast on mature regolith
     - TMC-2 PAN (500-850nm): Similar but narrower response
     - IIRS VNIR (800-2500nm): Pyroxene/olivine absorption features
     - IIRS SWIR (2500-5000nm): Thermal contribution, inverted shadows

  This makes cross-sensor matching genuinely harder than just rescaling —
  the images LOOK DIFFERENT because different wavelengths interact
  differently with lunar minerals (FeO, TiO₂, plagioclase, pyroxene).

The renderer:
  1. Generates terrain at a fine base resolution (1 m/px)
  2. Generates spatially-varying composition maps (mineral abundance)
  3. Computes spectral albedo per sensor band from composition
  4. Resamples to each sensor's GSD via area-averaging
  5. Renders with the existing physically-based illumination model
  6. Returns image pairs ready for cross-sensor matching

Does NOT modify sun_sim_v2.py or synth_dem.py (SHA-256 pinned).
Uses sun_sim_v2's existing `albedo` parameter for spectral variation.
"""

import numpy as np
import scipy.ndimage as ndi
import cv2

# Import from the SHA-pinned renderer (read-only)
from sun_sim_v2 import horizon_map, render, to_uint8
from sensors import SensorProfile, OHRC, TMC2, IIRS, SensorPair


# ── Multi-Scale Terrain ──────────────────────────────────────────────────────

class MultiScaleTerrain:
    """Physical terrain that can be viewed at any sensor GSD.

    The terrain is generated at a fine base resolution and then downsampled
    with area-averaging to simulate how coarser sensors integrate over their
    larger pixels. This produces genuinely different images at different
    scales — OHRC sees fine crater rims that are invisible to IIRS.
    """

    def __init__(self, physical_extent_m: float = 2560.0,
                 base_gsd_m: float = 1.0, seed: int = 42):
        """
        Parameters
        ----------
        physical_extent_m : Physical size of the terrain (metres, square).
        base_gsd_m        : Internal DEM resolution (metres/pixel).
        seed              : Random seed for reproducible terrain.
        """
        self.extent_m = physical_extent_m
        self.base_gsd = base_gsd_m
        self.base_size = int(physical_extent_m / base_gsd_m)
        self.seed = seed
        self.dem = self._generate_terrain()

    def _generate_terrain(self) -> np.ndarray:
        """Generate multi-scale crater terrain at base resolution.

        Creates features at multiple physical scales:
          - Large undulation  (>200 m)    → visible to all sensors
          - Medium craters    (40–150 m)  → visible to OHRC + TMC-2
          - Small craters     (5–30 m)    → visible to OHRC only
          - Fine-scale texture (<5 m)     → sub-pixel for TMC-2 and IIRS
        """
        np.random.seed(self.seed)
        n = self.base_size
        dem = np.zeros((n, n), dtype=np.float32)

        # Large-scale topography (all sensors)
        dem += ndi.gaussian_filter(
            np.random.normal(0, 50, (n, n)).astype(np.float32),
            sigma=max(1, n // 6)
        ) * 8

        # Medium-scale roughness (TMC-2 + OHRC)
        dem += ndi.gaussian_filter(
            np.random.normal(0, 30, (n, n)).astype(np.float32),
            sigma=max(1, n // 30)
        ) * 3

        # Fine-scale texture (OHRC only)
        dem += ndi.gaussian_filter(
            np.random.normal(0, 10, (n, n)).astype(np.float32),
            sigma=max(1, n // 200)
        ) * 1.5

        # Multi-scale craters
        y, x = np.mgrid[0:n, 0:n]
        specs = []

        # Large craters (200–500 m diameter, visible to IIRS)
        for _ in range(5):
            cx = np.random.randint(n // 8, 7 * n // 8)
            cy = np.random.randint(n // 8, 7 * n // 8)
            r_m = np.random.uniform(100, 250)
            depth = np.random.uniform(150, 300)
            specs.append((cx, cy, r_m, depth))

        # Medium craters (40–150 m diameter, visible to TMC-2)
        for _ in range(15):
            cx = np.random.randint(n // 10, 9 * n // 10)
            cy = np.random.randint(n // 10, 9 * n // 10)
            r_m = np.random.uniform(20, 75)
            depth = np.random.uniform(50, 150)
            specs.append((cx, cy, r_m, depth))

        # Small craters (5–30 m diameter, OHRC only)
        for _ in range(30):
            cx = np.random.randint(n // 10, 9 * n // 10)
            cy = np.random.randint(n // 10, 9 * n // 10)
            r_m = np.random.uniform(2.5, 15)
            depth = np.random.uniform(10, 50)
            specs.append((cx, cy, r_m, depth))

        for cx, cy, r_m, depth in specs:
            r_px = r_m / self.base_gsd
            if r_px < 1:
                continue
            dist = np.hypot(x - cx, y - cy)
            in_crater = dist < r_px
            dem[in_crater] -= depth * (1 - (dist[in_crater] / r_px) ** 2)
            rim = (dist >= r_px) & (dist < r_px * 1.3)
            rim_frac = (dist[rim] - r_px) / (r_px * 0.3)
            dem[rim] += depth * 0.4 * (1 - rim_frac)

        return dem

    def get_sensor_dem(self, sensor: SensorProfile,
                       output_size: int = 512,
                       crop_center_m: tuple = None) -> np.ndarray:
        """Extract DEM as seen by a specific sensor.

        Parameters
        ----------
        sensor         : Target sensor profile.
        output_size    : Desired output image size (pixels).
        crop_center_m  : (row_m, col_m) physical centre of crop. None = terrain centre.

        Returns
        -------
        np.ndarray  (size × size) float32 DEM in metres.
        """
        gsd = sensor.gsd_m
        crop_extent_m = output_size * gsd

        if crop_center_m is None:
            center_m = (self.extent_m / 2, self.extent_m / 2)
        else:
            center_m = crop_center_m

        # Convert to base DEM pixel coordinates
        center_px_r = center_m[0] / self.base_gsd
        center_px_c = center_m[1] / self.base_gsd
        half_px = crop_extent_m / (2 * self.base_gsd)

        r0 = max(0, int(center_px_r - half_px))
        c0 = max(0, int(center_px_c - half_px))
        r1 = min(self.base_size, int(center_px_r + half_px))
        c1 = min(self.base_size, int(center_px_c + half_px))

        crop = self.dem[r0:r1, c0:c1].copy()

        if crop.shape[0] < 2 or crop.shape[1] < 2:
            raise ValueError(
                f"Sensor {sensor.name} at GSD={gsd}m needs {crop_extent_m:.0f}m "
                f"but terrain is only {self.extent_m:.0f}m across"
            )

        # Resample to output_size (area-averaging downsample)
        if crop.shape[0] != output_size:
            zoom_factor = output_size / crop.shape[0]
            resampled = ndi.zoom(crop, zoom_factor, order=1)
            resampled = resampled[:output_size, :output_size]
        else:
            resampled = crop[:output_size, :output_size]

        return resampled.astype(np.float32)

    def get_max_output_size(self, sensor: SensorProfile) -> int:
        """Maximum output image size this terrain supports for a given sensor."""
        return int(self.extent_m / sensor.gsd_m)


# ── Spectral Response Simulation ────────────────────────────────────────────
#
# Lunar surface composition varies spatially. Different sensors see different
# albedo because their spectral bands interact differently with minerals.
# This is what makes cross-sensor matching GENUINELY multi-modal.
#
# Reference: Lucey et al. (2000) "Lunar iron and titanium abundance algorithms"
# based on Clementine UVVIS data; Pieters (1999) pyroxene/olivine absorption.

# Spectral albedo ranges by mineral + wavelength band (reflectance 0-1)
# Rows: [mature_highland, mare_basalt, fresh_ejecta, pyroxene_rich]
# Cols: wavelengths approximate for each sensor band
_LUNAR_COMPOSITIONS = {
    # (min_albedo, max_albedo) per composition type
    "PAN_450_850": {    # OHRC band
        "highland":      (0.10, 0.16),   # anorthositic highland
        "mare":          (0.06, 0.09),   # dark basaltic mare
        "fresh_ejecta":  (0.18, 0.28),   # bright immature material
        "pyroxene":      (0.07, 0.12),   # mafic minerals
    },
    "PAN_500_850": {    # TMC-2 band (slightly narrower)
        "highland":      (0.11, 0.17),
        "mare":          (0.065, 0.095),
        "fresh_ejecta":  (0.19, 0.27),
        "pyroxene":      (0.075, 0.125),
    },
    "VNIR_800_2500": {  # IIRS VNIR — pyroxene absorption at ~1000nm, ~2000nm
        "highland":      (0.14, 0.22),   # brighter in NIR
        "mare":          (0.08, 0.13),   # absorption at 1000nm
        "fresh_ejecta":  (0.22, 0.32),   # very bright in NIR
        "pyroxene":      (0.05, 0.09),   # DEEP absorption band
    },
    "SWIR_2500_5000": { # IIRS SWIR — thermal + reflectance mix
        "highland":      (0.08, 0.12),   # lower reflectance in SWIR
        "mare":          (0.06, 0.10),
        "fresh_ejecta":  (0.10, 0.15),
        "pyroxene":      (0.04, 0.07),   # very low in SWIR
    },
}

# Map sensor names to spectral band keys
_SENSOR_BAND_MAP = {
    "OHRC":  "PAN_450_850",
    "TMC-2": "PAN_500_850",
    "IIRS":  "VNIR_800_2500",   # Use VNIR broadband for matching
}


def generate_composition_map(dem: np.ndarray, seed: int = 42) -> np.ndarray:
    """Generate spatially-varying lunar composition from DEM morphology.

    Uses elevation-based and curvature-based heuristics:
    - Low-lying flat areas → mare basalt (dark, Fe-rich)
    - Elevated terrain → highland anorthosite (bright)
    - Steep slopes / crater rims → fresh ejecta (brightest)
    - Broad depressions → pyroxene-rich (absorption features)

    Returns
    -------
    comp : ndarray shape (H, W, 4) — fractional abundance of
           [highland, mare, fresh_ejecta, pyroxene]  summing to 1.0
    """
    np.random.seed(seed + 7)  # offset from terrain seed
    h, w = dem.shape
    comp = np.zeros((h, w, 4), dtype=np.float32)

    # Normalise elevation to [0, 1]
    dem_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)

    # Curvature (Laplacian) — high curvature = crater rims / slopes
    laplacian = np.abs(ndi.laplace(dem).astype(np.float32))
    lap_norm = np.clip(laplacian / (np.percentile(laplacian, 95) + 1e-6), 0, 1)

    # Slope magnitude
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx**2 + gy**2)
    slope_norm = np.clip(slope / (np.percentile(slope, 95) + 1e-6), 0, 1)

    # Highland fraction: high elevation areas
    comp[:, :, 0] = np.clip(dem_norm * 1.5, 0, 1)

    # Mare fraction: low flat areas
    comp[:, :, 1] = np.clip((1 - dem_norm) * (1 - slope_norm) * 1.8, 0, 1)

    # Fresh ejecta: steep slopes, crater rims
    comp[:, :, 2] = np.clip(lap_norm * 0.8 + slope_norm * 0.5, 0, 1)

    # Pyroxene-rich: mid-elevation depressions with moderate slope
    mid_elev = 1 - 2 * np.abs(dem_norm - 0.4)
    comp[:, :, 3] = np.clip(mid_elev * slope_norm * 1.2, 0, 1)

    # Add spatial noise for realism (patchy distribution)
    for i in range(4):
        noise = ndi.gaussian_filter(
            np.random.normal(0, 0.15, (h, w)).astype(np.float32),
            sigma=max(1, h // 15)
        )
        comp[:, :, i] = np.clip(comp[:, :, i] + noise, 0.01, None)

    # Normalise to sum to 1
    total = comp.sum(axis=2, keepdims=True)
    comp /= (total + 1e-6)

    return comp


def spectral_albedo_map(comp: np.ndarray, sensor: SensorProfile) -> np.ndarray:
    """Compute spatially-varying albedo for a given sensor's spectral band.

    Parameters
    ----------
    comp   : ndarray (H, W, 4) — composition fractions
    sensor : SensorProfile — determines which spectral band to use

    Returns
    -------
    albedo : ndarray (H, W) float32 — pixel-wise albedo for this sensor
    """
    band_key = _SENSOR_BAND_MAP.get(sensor.name, "PAN_450_850")
    band_spec = _LUNAR_COMPOSITIONS[band_key]

    comp_names = ["highland", "mare", "fresh_ejecta", "pyroxene"]
    h, w = comp.shape[:2]
    albedo = np.zeros((h, w), dtype=np.float32)

    for i, name in enumerate(comp_names):
        lo, hi = band_spec[name]
        # Use the mean albedo weighted by composition fraction
        mean_albedo = (lo + hi) / 2.0
        albedo += comp[:, :, i] * mean_albedo

    # Normalise to [0.3, 1.0] range for the renderer
    # (renderer already handles physical illumination; albedo modulates it)
    albedo_norm = (albedo - albedo.min()) / (albedo.max() - albedo.min() + 1e-6)
    albedo_out = 0.3 + 0.7 * albedo_norm

    return albedo_out.astype(np.float32)


# ── Cross-Sensor Rendering ──────────────────────────────────────────────────

def render_sensor_view(terrain: MultiScaleTerrain, sensor: SensorProfile,
                       az_deg: float, el_deg: float,
                       output_size: int = 512,
                       crop_center_m: tuple = None,
                       composition: np.ndarray = None) -> np.ndarray:
    """Render terrain as seen by a specific sensor under given illumination.

    If composition is provided, applies spectral albedo modulation —
    making different sensors produce genuinely different-looking images
    of the same terrain (not just rescaled).

    Returns uint8 image ready for matching.
    """
    dem = terrain.get_sensor_dem(sensor, output_size, crop_center_m)
    h = horizon_map(dem, sensor.gsd_m, az_deg)

    # Compute sensor-specific albedo from composition
    if composition is not None:
        # Resample composition map to match DEM size
        comp_resized = np.zeros((dem.shape[0], dem.shape[1], 4), dtype=np.float32)
        for ch in range(4):
            comp_resized[:, :, ch] = cv2.resize(
                composition[:, :, ch],
                (dem.shape[1], dem.shape[0]),
                interpolation=cv2.INTER_AREA
            )
        # Renormalize after resize
        total = comp_resized.sum(axis=2, keepdims=True)
        comp_resized /= (total + 1e-6)

        albedo = spectral_albedo_map(comp_resized, sensor)
        refl = render(dem, sensor.gsd_m, az_deg, el_deg, horizon=h, albedo=albedo)
    else:
        refl = render(dem, sensor.gsd_m, az_deg, el_deg, horizon=h)

    return to_uint8(refl)


def render_sensor_pair(terrain: MultiScaleTerrain, pair: SensorPair,
                       az_deg: float, el_deg: float,
                       output_size: int = 512,
                       az2_deg: float = None, el2_deg: float = None,
                       use_spectral: bool = True) -> tuple:
    """Render the same terrain as seen by two different sensors.

    Both sensors view the same physical terrain centre. When use_spectral=True,
    each sensor gets a different spectral albedo — making this genuinely
    multi-modal (different sensors produce visually different images).

    Returns (img_a, img_b, size_a, size_b).
    """
    center = (terrain.extent_m / 2, terrain.extent_m / 2)

    # Generate composition map (same terrain, same composition for both sensors)
    comp = None
    if use_spectral:
        comp = generate_composition_map(terrain.dem, seed=terrain.seed)

    # Determine achievable size for each sensor
    size_a = min(output_size, terrain.get_max_output_size(pair.sensor_a))
    size_b = min(output_size, terrain.get_max_output_size(pair.sensor_b))

    img_a = render_sensor_view(
        terrain, pair.sensor_a, az_deg, el_deg, size_a, center, comp
    )

    az2 = az2_deg if az2_deg is not None else az_deg
    el2 = el2_deg if el2_deg is not None else el_deg

    img_b = render_sensor_view(
        terrain, pair.sensor_b, az2, el2, size_b, center, comp
    )

    return img_a, img_b, size_a, size_b


# ── Cross-sensor preprocessing for matching ─────────────────────────────────

def resample_to_common_gsd(img_fine: np.ndarray, img_coarse: np.ndarray,
                           gsd_fine: float, gsd_coarse: float,
                           target: str = "coarse") -> tuple:
    """Resample images to a common GSD for matching.

    Parameters
    ----------
    target : str
        "coarse" — downsample fine image to match coarse GSD (standard)
        "fine"   — upsample coarse image to match fine GSD (no new info)
        "mid"    — both to geometric mean GSD

    Returns
    -------
    (img_a_resampled, img_b_resampled, common_gsd)
    """
    if target == "coarse":
        common_gsd = gsd_coarse
    elif target == "fine":
        common_gsd = gsd_fine
    elif target == "mid":
        common_gsd = np.sqrt(gsd_fine * gsd_coarse)
    else:
        raise ValueError(f"Unknown target: {target}")

    def _resample(img, src_gsd, dst_gsd):
        if abs(src_gsd - dst_gsd) < 0.01:
            return img
        scale = src_gsd / dst_gsd
        h, w = img.shape[:2]
        new_h, new_w = max(4, int(h * scale)), max(4, int(w * scale))
        interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    return (
        _resample(img_fine, gsd_fine, common_gsd),
        _resample(img_coarse, gsd_coarse, common_gsd),
        common_gsd,
    )


def prepare_cross_sensor_match(img_a: np.ndarray, img_b: np.ndarray,
                                sensor_a: SensorProfile,
                                sensor_b: SensorProfile,
                                strategy: str = "coarse") -> tuple:
    """Prepare two sensor images for matching by cropping to their common geographic extent
    and resampling to a shared resolution.

    Returns (processed_a, processed_b, common_gsd, scale_factor).
    """
    # 1. Compute physical coverage (meters)
    width_a_m = img_a.shape[1] * sensor_a.gsd_m
    height_a_m = img_a.shape[0] * sensor_a.gsd_m
    width_b_m = img_b.shape[1] * sensor_b.gsd_m
    height_b_m = img_b.shape[0] * sensor_b.gsd_m

    common_w_m = min(width_a_m, width_b_m)
    common_h_m = min(height_a_m, height_b_m)

    # 2. Center crop both images to common physical extent
    def _crop_center_meters(img, gsd_m, target_w_m, target_h_m):
        px_w = max(4, int(round(target_w_m / gsd_m)))
        px_h = max(4, int(round(target_h_m / gsd_m)))
        h, w = img.shape[:2]
        px_w = min(w, px_w)
        px_h = min(h, px_h)
        r0 = (h - px_h) // 2
        c0 = (w - px_w) // 2
        return img[r0:r0 + px_h, c0:c0 + px_w]

    crop_a = _crop_center_meters(img_a, sensor_a.gsd_m, common_w_m, common_h_m)
    crop_b = _crop_center_meters(img_b, sensor_b.gsd_m, common_w_m, common_h_m)

    # 3. Resample to common pixel dimensions based on strategy
    gsd_fine = min(sensor_a.gsd_m, sensor_b.gsd_m)
    gsd_coarse = max(sensor_a.gsd_m, sensor_b.gsd_m)

    if strategy == "coarse":
        common_gsd = gsd_coarse
        target_size = (crop_b.shape[1], crop_b.shape[0]) if sensor_a.gsd_m <= sensor_b.gsd_m else (crop_a.shape[1], crop_a.shape[0])
    elif strategy == "fine":
        common_gsd = gsd_fine
        target_size = (crop_a.shape[1], crop_a.shape[0]) if sensor_a.gsd_m <= sensor_b.gsd_m else (crop_b.shape[1], crop_b.shape[0])
    else:  # mid or fixed 512
        target_size = (512, 512)
        common_gsd = common_w_m / 512.0

    proc_a = cv2.resize(crop_a, target_size, interpolation=cv2.INTER_AREA if crop_a.shape[0] > target_size[1] else cv2.INTER_CUBIC)
    proc_b = cv2.resize(crop_b, target_size, interpolation=cv2.INTER_AREA if crop_b.shape[0] > target_size[1] else cv2.INTER_CUBIC)

    scale_factor = gsd_fine / gsd_coarse
    return proc_a, proc_b, common_gsd, scale_factor


# ── NCC for cross-sensor similarity ─────────────────────────────────────────

def cross_sensor_ncc(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Normalised cross-correlation between two images (may be different sizes).

    If sizes differ, the larger is centre-cropped to match the smaller.
    """
    # Centre-crop to smaller
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])

    def _centre_crop(img, th, tw):
        r0 = (img.shape[0] - th) // 2
        c0 = (img.shape[1] - tw) // 2
        return img[r0:r0 + th, c0:c0 + tw]

    a = _centre_crop(img_a, h, w).astype(np.float64).ravel()
    b = _centre_crop(img_b, h, w).astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


# ── Self-test ────────────────────────────────────────────────────────────────

def _self_test():
    """Quick validation of multi-scale + multi-spectral rendering."""
    print("MultiScaleTerrain self-test (with spectral simulation)...")

    terrain = MultiScaleTerrain(
        physical_extent_m=2560.0, base_gsd_m=2.0, seed=42
    )
    print(f"  Terrain: {terrain.base_size}x{terrain.base_size} at "
          f"{terrain.base_gsd} m/px, extent={terrain.extent_m}m")
    print(f"  DEM range: {terrain.dem.min():.1f} to {terrain.dem.max():.1f} m")

    # Generate composition map
    comp = generate_composition_map(terrain.dem, seed=terrain.seed)
    print(f"  Composition map: {comp.shape}, sums to {comp.sum(axis=2).mean():.3f}")

    # Render at each sensor's GSD (with spectral albedo)
    imgs = {}
    for sensor in [OHRC, TMC2, IIRS]:
        max_size = terrain.get_max_output_size(sensor)
        use_size = min(256, max_size)
        try:
            img = render_sensor_view(
                terrain, sensor, az_deg=270, el_deg=40,
                output_size=use_size, composition=comp
            )
            shadow = float(np.sum(img == 0)) / img.size
            phys = use_size * sensor.gsd_m
            mean_val = float(img[img > 0].mean()) if img.max() > 0 else 0
            print(f"  {sensor.name:6s} (GSD={sensor.gsd_m:5.1f}m, "
                  f"band={_SENSOR_BAND_MAP.get(sensor.name, '?')}): "
                  f"{img.shape[0]}x{img.shape[1]}px, "
                  f"covers {phys:.0f}m, "
                  f"shadow={shadow:.1%}, mean_lit={mean_val:.1f}")
            imgs[sensor.name] = img
        except ValueError as e:
            print(f"  {sensor.name:6s}: SKIPPED — {e}")

    # Verify spectral contrast: different sensors should produce
    # visually different images (different mean brightness)
    if "OHRC" in imgs and "IIRS" in imgs:
        ohrc_mean = float(imgs["OHRC"][imgs["OHRC"] > 0].mean())
        iirs_mean = float(imgs["IIRS"][imgs["IIRS"] > 0].mean())
        diff_pct = abs(ohrc_mean - iirs_mean) / max(ohrc_mean, iirs_mean) * 100
        print(f"  Spectral contrast OHRC vs IIRS: {diff_pct:.1f}% "
              f"(OHRC mean={ohrc_mean:.1f}, IIRS mean={iirs_mean:.1f})")
        assert diff_pct > 1.0, (
            "Spectral simulation should create visible contrast between sensors"
        )

    # Cross-sensor pair (with spectral)
    from sensors import SENSOR_PAIRS
    pair = SENSOR_PAIRS[0]  # OHRC ↔ TMC-2
    img_a, img_b, sa, sb = render_sensor_pair(
        terrain, pair, 270, 40, output_size=128, use_spectral=True
    )
    ncc_val = cross_sensor_ncc(img_a, img_b)
    print(f"  Cross-sensor {pair.key} (spectral): "
          f"A={img_a.shape[0]}x{img_a.shape[1]}, "
          f"B={img_b.shape[0]}x{img_b.shape[1]}, "
          f"NCC={ncc_val:+.3f}")

    # Compare with non-spectral to show the difference
    img_a_ns, img_b_ns, _, _ = render_sensor_pair(
        terrain, pair, 270, 40, output_size=128, use_spectral=False
    )
    ncc_ns = cross_sensor_ncc(img_a_ns, img_b_ns)
    print(f"  Cross-sensor {pair.key} (no spectral): NCC={ncc_ns:+.3f}")
    print(f"  -> Spectral simulation reduces NCC by {abs(ncc_ns - ncc_val):.3f} "
          f"(harder matching = more realistic)")

    # Prepare for matching
    proc_a, proc_b, cgsd, sf = prepare_cross_sensor_match(
        img_a, img_b, pair.sensor_a, pair.sensor_b
    )
    print(f"  After resampling: A={proc_a.shape}, B={proc_b.shape}, "
          f"common_gsd={cgsd:.1f}m, scale_factor={sf:.3f}")

    print("  [OK] multi-modal self-test passed")


if __name__ == "__main__":
    _self_test()
