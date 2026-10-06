import logging
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data.dem_loader import DEMLoader
from src.data.image_loader import ImageLoader
from src.pipeline.router import TrustRouter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing LunarAlign Multi-Modal Pipeline...")

    # ── Scenario: Cross-sensor matching (OHRC ↔ TMC-2) ─────────────────────
    # In production, these paths point to real PDS4 data from PRADAN/QuickMap.
    # For now, we demonstrate the multi-modal routing with dummy data.

    # Image 1: OHRC (0.3 m/px) — high-resolution reference
    ohrc_loader = ImageLoader(
        "data/raw/ohrc_image.tif",
        sensor_type="OHRC",
    )
    ohrc_image = ohrc_loader.load()
    ohrc_sun_el, ohrc_sun_az = ohrc_loader.get_sun_angles()
    ohrc_sensor, ohrc_gsd = ohrc_loader.get_sensor_info()

    logger.info(
        f"OHRC Image — Sensor: {ohrc_sensor}, GSD: {ohrc_gsd} m/px, "
        f"Sun El: {ohrc_sun_el}°, Az: {ohrc_sun_az}°"
    )

    # Image 2: TMC-2 (5 m/px) — wider-area, lower-resolution target
    tmc_loader = ImageLoader(
        "data/raw/tmc2_image.tif",
        sensor_type="TMC-2",
    )
    tmc_image = tmc_loader.load()
    tmc_sun_el, tmc_sun_az = tmc_loader.get_sun_angles()
    tmc_sensor, tmc_gsd = tmc_loader.get_sensor_info()

    logger.info(
        f"TMC-2 Image — Sensor: {tmc_sensor}, GSD: {tmc_gsd} m/px, "
        f"Sun El: {tmc_sun_el}°, Az: {tmc_sun_az}°"
    )

    # ── Multi-modal routing ─────────────────────────────────────────────────
    # The router considers BOTH sun angle difficulty AND sensor pair scale ratio.
    logger.info("=" * 60)
    logger.info("MULTI-MODAL ROUTING")
    logger.info("=" * 60)

    try:
        from src.router import route_and_match
        result, trust = route_and_match(
            ohrc_image, tmc_image,
            az1=ohrc_sun_az, el1=ohrc_sun_el,
            az2=tmc_sun_az, el2=tmc_sun_el,
            sensor1=ohrc_sensor, sensor2=tmc_sensor,
        )
        logger.info(f"Trust Light:    {trust}")
        logger.info(f"Cross-sensor:   {result.get('cross_sensor', False)}")
        if result.get('sensor_pair'):
            logger.info(f"Sensor Pair:    {result['sensor_pair']}")
            logger.info(f"Scale Ratio:    {result.get('scale_ratio', 'N/A')}×")
        logger.info(f"Selected Matcher: {result['matcher']}")
        if result.get('is_valid'):
            logger.info(f"Matches: {result.get('raw_matches', 0)}")
        else:
            logger.warning(f"Reason: {result.get('reason', 'unknown')}")

    except ImportError:
        logger.warning("Advanced router not available. Using simple pipeline router.")

        # Fallback to simple pipeline router
        router = TrustRouter()
        el_use = min(ohrc_sun_el, tmc_sun_el)
        match_result, trust_level = router.route_and_match(
            ohrc_image, tmc_image, el_use
        )
        logger.info(f"Trust Light: {trust_level}")
        if match_result:
            logger.info(f"Selected Matcher: {match_result['matcher']}")
            logger.info(f"Inliers found: {match_result['inliers']}")
        else:
            logger.warning("Matching refused (RED trust light).")

    # ── Sensor summary ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("─" * 60)
    logger.info("SENSOR PROFILES")
    logger.info("─" * 60)
    try:
        from src.sensors import SENSORS, SENSOR_PAIRS
        for name, s in [("OHRC", "OHRC"), ("TMC-2", "TMC2"), ("IIRS", "IIRS")]:
            p = SENSORS[s]
            logger.info(
                f"  {name:6s}  GSD={p.gsd_m:6.1f} m/px  "
                f"Swath={p.swath_km:5.1f} km  {p.spectral_range}"
            )
        logger.info("")
        for pair in SENSOR_PAIRS:
            logger.info(
                f"  {pair.name:20s}  Scale ratio: {pair.scale_ratio:>5.0f}×"
            )
    except ImportError:
        logger.info("  (sensor profiles not available)")


if __name__ == "__main__":
    main()
