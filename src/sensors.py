"""
sensors.py — Chandrayaan-2 Sensor Profiles for Multi-Modal Matching
═══════════════════════════════════════════════════════════════════════

Defines the three target sensors from PS 26166:
  OHRC  — Orbiter High Resolution Camera  (0.3 m/px, panchromatic)
  TMC-2 — Terrain Mapping Camera          (5.0 m/px, panchromatic)
  IIRS  — Imaging IR Spectrometer         (80  m/px, VNIR+SWIR)

Sensor pairs for cross-modal matching:
  OHRC ↔ TMC-2  : 17× scale ratio  (typical co-registration task)
  TMC-2 ↔ IIRS  : 16× scale ratio  (spectral-spatial fusion)
  OHRC ↔ IIRS   : 267× scale ratio (extreme, likely RED-only)
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SensorProfile:
    """Immutable sensor specification matching ISRO documentation."""
    name: str
    full_name: str
    gsd_m: float            # Ground Sample Distance (metres/pixel)
    swath_km: float          # Swath width (km)
    spectral_range: str      # Wavelength range description
    band_count: int          # Number of spectral bands
    bit_depth: int           # Radiometric resolution (bits)
    orbit_alt_km: float      # Nominal orbit altitude (km)
    pds4_instrument_id: str  # PDS4 metadata identifier


# ── Official sensor specifications ──────────────────────────────────────────

OHRC = SensorProfile(
    name="OHRC",
    full_name="Orbiter High Resolution Camera",
    gsd_m=0.3,
    swath_km=3.0,
    spectral_range="PAN 450–850 nm",
    band_count=1,
    bit_depth=11,
    orbit_alt_km=100,
    pds4_instrument_id="ch2_ohrc",
)

TMC2 = SensorProfile(
    name="TMC-2",
    full_name="Terrain Mapping Camera-2",
    gsd_m=5.0,
    swath_km=20.0,
    spectral_range="PAN 500–850 nm",
    band_count=1,
    bit_depth=10,
    orbit_alt_km=100,
    pds4_instrument_id="ch2_tmc2",
)

IIRS = SensorProfile(
    name="IIRS",
    full_name="Imaging Infrared Spectrometer",
    gsd_m=80.0,
    swath_km=20.0,
    spectral_range="VNIR+SWIR 800–5000 nm",
    band_count=256,       # hyperspectral; broadband sum used for matching
    bit_depth=12,
    orbit_alt_km=100,
    pds4_instrument_id="ch2_iirs",
)

# Registry for lookup by name
SENSORS: Dict[str, SensorProfile] = {
    "OHRC":  OHRC,
    "TMC2":  TMC2,
    "TMC-2": TMC2,   # alias
    "IIRS":  IIRS,
}


def get_sensor(name: str) -> SensorProfile:
    """Look up a sensor by name (case-insensitive, dash-tolerant)."""
    key = name.upper().replace("_", "")
    for k, v in SENSORS.items():
        if k.upper().replace("_", "") == key:
            return v
    raise KeyError(f"Unknown sensor '{name}'. Known: {list(SENSORS.keys())}")


# ── Cross-sensor pairs ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class SensorPair:
    """A pair of sensors for cross-modal matching."""
    sensor_a: SensorProfile
    sensor_b: SensorProfile

    @property
    def scale_ratio(self) -> float:
        """GSD ratio (always >= 1, coarser / finer)."""
        a, b = self.sensor_a.gsd_m, self.sensor_b.gsd_m
        return max(a, b) / min(a, b)

    @property
    def name(self) -> str:
        return f"{self.sensor_a.name} \u2194 {self.sensor_b.name}"

    @property
    def key(self) -> str:
        """Filesystem-safe key for CSV / paths."""
        a = self.sensor_a.name.replace("-", "")
        b = self.sensor_b.name.replace("-", "")
        return f"{a}_x_{b}"


# Ordered by difficulty (easiest first)
SENSOR_PAIRS: List[SensorPair] = [
    SensorPair(OHRC, TMC2),     # 17× scale
    SensorPair(TMC2, IIRS),     # 16× scale
    SensorPair(OHRC, IIRS),     # 267× scale (extreme)
]


def get_all_pairs() -> List[SensorPair]:
    """Return all defined cross-sensor pairs."""
    return list(SENSOR_PAIRS)


def get_pair(name_a: str, name_b: str) -> SensorPair:
    """Construct a sensor pair by sensor names."""
    a = get_sensor(name_a)
    b = get_sensor(name_b)
    return SensorPair(a, b)
