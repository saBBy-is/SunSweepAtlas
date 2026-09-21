import cv2
import numpy as np
import logging

try:
    import torch
    import kornia.feature as KF
except ImportError:
    torch = None
    KF = None

logger = logging.getLogger(__name__)

class MatcherBase:
    def __init__(self, name: str):
        self.name = name
        
    def match(self, img1: np.ndarray, img2: np.ndarray):
        """Returns keypoints1, keypoints2, and matched point indices/coordinates."""
        raise NotImplementedError

class SIFTMatcher(MatcherBase):
    def __init__(self):
        super().__init__("SIFT")
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
    def match(self, img1: np.ndarray, img2: np.ndarray):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return [], [], []
            
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        # Extract points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Calculate robust inliers using RANSAC if enough matches
        inliers = 0
        if len(good_matches) > 4:
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = np.sum(mask)
                
        return pts1, pts2, inliers

class LoFTRMatcher(MatcherBase):
    def __init__(self):
        super().__init__("LoFTR")
        if KF is not None:
            self.matcher = KF.LoFTR(pretrained='outdoor')
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.matcher = self.matcher.to(self.device).eval()
        else:
            self.matcher = None
            logger.warning("Kornia not installed, LoFTR will be disabled.")
            
    def match(self, img1: np.ndarray, img2: np.ndarray):
        if self.matcher is None:
            return [], [], 0
            
        # Kornia expects torch tensors of shape (B, 1, H, W) normalized to [0, 1]
        img1_t = torch.from_numpy(img1).float()[None, None] / 255.0
        img2_t = torch.from_numpy(img2).float()[None, None] / 255.0
        
        img1_t = img1_t.to(self.device)
        img2_t = img2_t.to(self.device)
        
        input_dict = {"image0": img1_t, "image1": img2_t}
        with torch.no_grad():
            correspondences = self.matcher(input_dict)
            
        # Extract matched points
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        
        inliers = 0
        if len(mkpts0) > 4:
            _, mask = cv2.findHomography(mkpts0.reshape(-1, 1, 2), mkpts1.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = np.sum(mask)
                
        return mkpts0, mkpts1, inliers

def run_sun_sweep(base_image: np.ndarray, simulator, azimuth: float, elevations: list, matchers: list):
    """
    Evaluates different matchers across a sweep of sun elevations.
    
    Returns a dictionary simulating the "Sun-Sweep Atlas" results.
    """
    results = {matcher.name: [] for matcher in matchers}
    
    for el in elevations:
        synthetic_img = simulator.generate_synthetic_image(azimuth, el)
        
        for matcher in matchers:
            _, _, inliers = matcher.match(base_image, synthetic_img)
            results[matcher.name].append((el, inliers))
            
    return results
