# LunarAlign

Pipeline for image correspondence between OHRC, TMC-2, and IIRS sensors for SIH 2026.

## Differentiator
- **Sun-Sweep Atlas & Trust Router**: Matches based on sun angles and trust scoring.

## Setup
1. Create virtual environment.
2. Install requirements: `pip install -r requirements.txt`

## Structure
- `src/data/`: Data loading for DEMs and ISRO images.
- `src/simulation/`: Shading simulator for DEMs.
- `src/matching/`: Benchmark matching algorithms.
- `src/pipeline/`: Trust Router for choosing optimal matchers.
