import logging
import sys
import os

# Ensure src/ is on path
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from src.matching.benchmark import SIFTMatcher, LoFTRMatcher

logger = logging.getLogger(__name__)

# Try to load sensor profiles for multi-modal awareness
try:
    from sensors import get_sensor
    _SENSORS_OK = True
except ImportError:
    _SENSORS_OK = False


class TrustRouter:
    def __init__(self):
        self.matchers = {
            "SIFT": SIFTMatcher(),
            "LoFTR": LoFTRMatcher()
        }
        
    def determine_trust_level(self, elevation: float,
                               sensor_a: str = None, sensor_b: str = None):
        """
        Returns a trust level based on sun elevation angle and sensor pair.
        Red: Extreme angles (shadows dominate or no shadows).
        Amber: Challenging but possible.
        Green: Optimal lighting for feature matching.

        If cross-sensor, the trust level is downgraded based on the scale ratio.
        """
        # Base trust from sun angle
        if elevation < 10 or elevation > 85:
            base = "Red"
        elif 10 <= elevation < 30 or 75 < elevation <= 85:
            base = "Amber"
        else:
            base = "Green"

        # Cross-sensor penalty
        if sensor_a and sensor_b and _SENSORS_OK:
            try:
                sa = get_sensor(sensor_a)
                sb = get_sensor(sensor_b)
                ratio = max(sa.gsd_m, sb.gsd_m) / min(sa.gsd_m, sb.gsd_m)
                if ratio > 100:
                    # Extreme scale difference (OHRC↔IIRS: 267×) → always Red
                    return "Red"
                elif ratio > 10:
                    # Moderate (OHRC↔TMC-2, TMC-2↔IIRS) → downgrade one level
                    if base == "Green":
                        return "Amber"
            except KeyError:
                pass

        return base
            
    def select_matcher(self, elevation: float, trust_level: str):
        """
        Selects the best matcher based on the Sun-Sweep Atlas logic.
        Amber (challenging lighting) → LoFTR (robust to illumination change).
        Green (optimal lighting) → SIFT (fast and reliable in good conditions).
        Red → refuse to match.
        """
        if trust_level == "Red":
            logger.warning("Trust Level RED: Lighting conditions too extreme for reliable matching.")
            return None
        elif trust_level == "Amber":
            logger.info("Trust Level AMBER: Using robust deep learning matcher (LoFTR).")
            return self.matchers["LoFTR"]
        else:
            logger.info("Trust Level GREEN: Using fast classical matcher (SIFT).")
            return self.matchers["SIFT"]
            
    def route_and_match(self, img1, img2, elevation: float,
                        sensor_a: str = None, sensor_b: str = None):
        """
        Full pipeline: Determines trust, picks matcher, and executes.

        Parameters
        ----------
        sensor_a, sensor_b : str or None
            Sensor names for cross-sensor matching (e.g. "OHRC", "TMC-2").
        """
        trust = self.determine_trust_level(elevation, sensor_a, sensor_b)
        matcher = self.select_matcher(elevation, trust)
        
        if matcher is None:
            return None, trust
            
        pts1, pts2, inliers = matcher.match(img1, img2)
        result = {
            "points1": pts1,
            "points2": pts2,
            "inliers": inliers,
            "matcher": matcher.name,
        }
        if sensor_a and sensor_b:
            result["sensor_a"] = sensor_a
            result["sensor_b"] = sensor_b
            if _SENSORS_OK:
                try:
                    sa = get_sensor(sensor_a)
                    sb = get_sensor(sensor_b)
                    result["scale_ratio"] = max(sa.gsd_m, sb.gsd_m) / min(sa.gsd_m, sb.gsd_m)
                except KeyError:
                    pass
        return result, trust
