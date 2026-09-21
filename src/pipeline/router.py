import logging
from src.matching.benchmark import SIFTMatcher, LoFTRMatcher

logger = logging.getLogger(__name__)

class TrustRouter:
    def __init__(self):
        self.matchers = {
            "SIFT": SIFTMatcher(),
            "LoFTR": LoFTRMatcher()
        }
        
    def determine_trust_level(self, elevation: float):
        """
        Returns a trust level based on sun elevation angle.
        Red: Extreme angles (shadows dominate or no shadows).
        Amber: Challenging but possible.
        Green: Optimal lighting for feature matching.
        """
        if elevation < 10 or elevation > 85:
            return "Red"
        elif 10 <= elevation < 30 or 75 < elevation <= 85:
            return "Amber"
        else:
            return "Green"
            
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
            
    def route_and_match(self, img1, img2, elevation: float):
        """
        Full pipeline: Determines trust, picks matcher, and executes.
        """
        trust = self.determine_trust_level(elevation)
        matcher = self.select_matcher(elevation, trust)
        
        if matcher is None:
            return None, trust
            
        pts1, pts2, inliers = matcher.match(img1, img2)
        return {"points1": pts1, "points2": pts2, "inliers": inliers, "matcher": matcher.name}, trust
