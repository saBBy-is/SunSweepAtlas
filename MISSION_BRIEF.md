# SIH 2026: Project LunarAlign (PS: 26166)
**Objective:** Build a pipeline for image correspondence between OHRC (0.3m), TMC-2 (5m), and IIRS (80m) sensors.

## THE WINNING DIFFERENTIATOR: "Sun-Sweep Atlas & Trust Router"
Most teams will try a generic AI matcher. We will be unique by:
1. **Shading Simulation:** Using LOLA DEM (Digital Elevation Models) to render synthetic lunar terrain under varying sun angles.
2. **Matcher Benchmarking:** Running a "Sweep" of 5+ state-of-the-art matchers (LightGlue, RoMa, LoFTR, XoFTR, SIFT) against these synthetic images to see which ones break at what sun angles.
3. **The Router:** Creating a logic that reads sun-angle metadata from the ISRO image and automatically picks the "best" matcher for that specific lighting condition.
4. **Reliability:** Implementing a "Red/Amber/Green" trust light. If the sun angle is too extreme, the system refuses to provide a false match.

## TECH STACK
- **Language:** Python
- **Key Libs:** GDAL, OpenCV, Kornia, vismatch (for matcher management)
- **Data Sources:** PDS4 (Lunar), ISRO PRADAN, NASA LOLA DEMs.
