import torch
import cv2
import numpy as np

def run_loftr(img1, img2, conf_threshold=0.2):
    """
    Run LoFTR (Local Feature TRansformer) for cross-modal matching.
    Returns matched keypoints (pts1, pts2).
    """
    try:
        from kornia.feature import LoFTR
        import kornia as K
    except ImportError:
        print("Kornia not installed. Cannot run LoFTR.")
        return np.array([]), np.array([])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LoFTR(pretrained='outdoor').to(device).eval()

    # Convert to grayscale tensor [1, 1, H, W], normalized 0-1
    timg1 = K.image_to_tensor(img1, False).float() / 255.0
    timg2 = K.image_to_tensor(img2, False).float() / 255.0

    if len(timg1.shape) == 3:
        timg1 = timg1.unsqueeze(0)
    if len(timg2.shape) == 3:
        timg2 = timg2.unsqueeze(0)
        
    timg1, timg2 = timg1.to(device), timg2.to(device)

    # Ensure sizes are multiples of 8 for LoFTR
    h1, w1 = timg1.shape[2:]
    h2, w2 = timg2.shape[2:]
    timg1 = torch.nn.functional.interpolate(timg1, size=(h1 - h1%8, w1 - w1%8))
    timg2 = torch.nn.functional.interpolate(timg2, size=(h2 - h2%8, w2 - w2%8))

    with torch.no_grad():
        input_dict = {"image0": timg1, "image1": timg2}
        correspondences = matcher(input_dict)
    
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    conf = correspondences['confidence'].cpu().numpy()
    
    valid = conf > conf_threshold
    return mkpts0[valid], mkpts1[valid]

def run_lightglue(img1, img2, features="disk", num_features=2048):
    """
    Run LightGlue (with DISK or SuperPoint features)
    Returns matched keypoints (pts1, pts2).
    """
    try:
        from kornia.feature import LightGlueMatcher, DISK
        import kornia as K
    except ImportError:
        print("Kornia not installed. Cannot run LightGlue.")
        return np.array([]), np.array([])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    timg1 = K.image_to_tensor(img1, False).float() / 255.0
    timg2 = K.image_to_tensor(img2, False).float() / 255.0
    
    if len(timg1.shape) == 3: timg1 = timg1.unsqueeze(0)
    if len(timg2.shape) == 3: timg2 = timg2.unsqueeze(0)

    timg1, timg2 = timg1.to(device), timg2.to(device)

    # Extract features
    if features == "disk":
        extractor = DISK.from_pretrained('depth').to(device)
    else:
        # Fallback to SIFT for testing if SuperPoint isn't natively in kornia
        extractor = DISK.from_pretrained('depth').to(device)

    with torch.no_grad():
        feat1 = extractor(timg1, num_features, pad_if_not_divisible=True)[0]
        feat2 = extractor(timg2, num_features, pad_if_not_divisible=True)[0]

    matcher = LightGlueMatcher(feature_name=features).to(device).eval()
    
    with torch.no_grad():
        match_res = matcher(feat1, feat2)
        
    kpts1 = feat1.keypoints[match_res[0, :, 0]].cpu().numpy()
    kpts2 = feat2.keypoints[match_res[0, :, 1]].cpu().numpy()

    return kpts1, kpts2
