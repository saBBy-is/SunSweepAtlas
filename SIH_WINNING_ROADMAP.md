# LunarAlign - SIH 2026 Problem Statement 26166 Winning Strategy

## The Core Narrative for Judges
"We built LunarAlign, an intelligent Trust-Routing pipeline that prevents lunar rovers and mapping algorithms from trusting bad data."

Instead of just building one AI model that sometimes fails silently, we built an intelligent **Trust Router** (an AI referee). Given two sensors and the sun angles, it predicts whether a match is mathematically reliable (GREEN), risky (AMBER), or impossible (RED). We simulated the physical spectral albedo and resolution disparities between OHRC (0.3m visible), TMC-2 (5.0m visible), and IIRS (80m infrared), mathematically proving why traditional matchers like SIFT completely fail (0% success). 

---

## 🏆 Roadmap to Victory (Execution Plan)

### [x] Step 1: Ingest Real Chandrayaan-2 Data (ISSDC PDS4 Parser)
**Status:** ✅ COMPLETED
**Details:** 
- Built `src/pds4_parser.py`.
- This script natively parses ISRO's XML metadata (PDS4 format) from the Indian Space Science Data Centre (ISSDC).
- Automatically extracts Sun Azimuth, Sun Elevation (derived from Incidence Angle), and physical ground resolution (GSD) to instantly feed into our Trust Router. 
- **Why it wins:** Proves our solution isn't just synthetic; it interfaces directly with ISRO's actual data archival format.

### [x] Step 2: Implement Deep Learning Matchers (LightGlue, LoFTR)
**Status:** ✅ COMPLETED
**Details:** 
- Built `src/deep_matchers.py` integrating SuperPoint+LightGlue and LoFTR via Kornia.
- **Why it wins:** Proves we are using modern 2024-era AI capable of crossing extreme modality gaps, succeeding where classical math fails.

### [x] Step 3: Interactive "Upload & Route" Feature in the Dashboard
**Status:** ✅ COMPLETED
**Details:** 
- Added a new interactive `LIVE PDS4 DEMO` tab to `dashboard.py`.
- Judges can select real ISRO PDS4 scenarios or upload custom XML files.
- The dashboard simulates real-time Trust Routing, demonstrating exactly when it assigns AMBER risk and uses Neural Matchers, and when it assigns RED risk to refuse the match.
- **Why it wins:** Hackathons are won during the live demo. An interactive, real-time UI is 10x more impressive than static analytics.

---
*Note: This document is fully up to date.*
