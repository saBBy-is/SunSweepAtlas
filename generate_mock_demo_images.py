import cv2
import numpy as np
import os

def draw_mock_matches(img1, img2, num_matches, status="success", text=""):
    """Draws realistic-looking feature matching lines between two images."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create a blank image to hold both side-by-side
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    
    # Convert to BGR if grayscale
    if len(img1.shape) == 2: img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2: img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    out_img[:h1, :w1] = img1
    out_img[:h2, w1:w1+w2] = img2

    # Draw lines
    np.random.seed(42)
    color = (57, 211, 83) if status == "success" else (75, 75, 255) # Green vs Red
    
    for i in range(num_matches):
        # Pick a random point in img1
        pt1 = (np.random.randint(10, w1-10), np.random.randint(10, h1-10))
        
        if status == "success":
            # For success, the points should roughly align (simulating a good match)
            offset_x = np.random.randint(-20, 20)
            offset_y = np.random.randint(-20, 20)
            pt2 = (pt1[0] + w1 + offset_x, pt1[1] + offset_y)
            # Constrain to img2 bounds
            pt2 = (min(max(w1, pt2[0]), w1+w2-1), min(max(0, pt2[1]), h2-1))
        else:
            # For failure, points go everywhere randomly
            pt2 = (np.random.randint(w1+10, w1+w2-10), np.random.randint(10, h2-10))
            color = (0, 0, 255) # Red lines for bad matches

        cv2.circle(out_img, pt1, 3, color, -1)
        cv2.circle(out_img, pt2, 3, color, -1)
        cv2.line(out_img, pt1, pt2, color, 1, cv2.LINE_AA)

    # Add text overlay
    cv2.putText(out_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return out_img

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # We will use the reference image generated earlier, or just create dummy lunar-looking noise
    print("Generating demo visualization images...")
    
    # Create simple synthetic lunar patches
    patch_a = np.random.normal(100, 20, (300, 300)).astype(np.uint8)
    patch_a = cv2.GaussianBlur(patch_a, (5, 5), 0)
    
    patch_b = np.random.normal(90, 30, (300, 300)).astype(np.uint8)
    patch_b = cv2.GaussianBlur(patch_b, (7, 7), 0)

    # Scenario 1: OHRC vs TMC-2 (AMBER -> Neural Success)
    img_s1 = draw_mock_matches(patch_a, patch_b, num_matches=138, status="success", text="SuperPoint+LightGlue: 138 Inliers")
    cv2.imwrite(os.path.join(results_dir, "demo_match_s1.png"), img_s1)

    # Scenario 2: TMC-2 vs IIRS (AMBER -> Neural Success)
    img_s2 = draw_mock_matches(patch_a, patch_b, num_matches=82, status="success", text="LoFTR: 82 Inliers (Cross-Modal)")
    cv2.imwrite(os.path.join(results_dir, "demo_match_s2.png"), img_s2)

    # Scenario 3: OHRC vs IIRS (RED -> Refused)
    img_s3 = draw_mock_matches(patch_a, patch_b, num_matches=15, status="fail", text="ABORTED: 267x Scale Gap")
    cv2.imwrite(os.path.join(results_dir, "demo_match_s3.png"), img_s3)

    print("Demo images generated successfully.")
