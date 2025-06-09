
# Tonal Dispersion in Huoji Chinese

This repository contains the analysis scripts and visualizations used in the dissertation:  
**_Checked and Neutral Tones in Huoji Chinese: A Multidimensional Analysis of Pitch, Duration, and Phonation_**,  
submitted for the MPhil in Theoretical and Applied Linguistics, University of Cambridge, 2025.

## What This Project Does

This project documents and analyzes marginal tonal categories in the Huoji dialect (a Jin variety of Northern Chinese), focusing on:
- **Historical Checked Tone (T5)**
- **Emergent Neutral Tone (NT)**

It uses a combination of acoustic phonetic measurements—F0, duration, H1–H2, and HNR—to examine how these tones behave and interact in monosyllabic and disyllabic contexts. The project employs unsupervised clustering and multidimensional scaling (MDS) to visualize tonal dispersion.

## Why This Project Is Useful

- Offers a reproducible pipeline for analyzing tonal contrasts with phonation features.
- Bridges gaps between traditional tonal theory and non-modal phonation.
- Demonstrates how tone systems evolve via glottal and durational cues.
- Highlights overlooked dialects and tonal categories in Chinese phonology.

## How to Get Started

### Prerequisites

Ensure you have:
- Python 3.8+
- Installed packages:  
  `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `plotly`, `praat-parselmouth`
- External software:  
  [Praat](http://www.fon.hum.uva.nl/praat/) and [VoiceSauce](https://github.com/voicesauce/voicesauce)

### Example Workflow

1. **Segment audio files** using Praat (save as `.TextGrid`).
2. **Extract features**:
    - F0 → `f0_analysis.py`
    - Duration → `duration_analysis.py`
    - H1–H2 → `h1h2_analysis.py`
    - HNR → `HNR_analysis.py`
3. **Cluster tones**:
    - Run `tone_pattern_kmeans_3.py` for F0+duration+phonation features.
4. **Visualize results**:
    - Plot F0 and H1–H2 using `plot_f0_new.py` and `plot_h1h2_new.py`.
    - Explore MDS maps in `/MDS/`.

## Repository Structure

### Folders
- `/Duration`: Scripts analyzing TBU durations.
- `/MDS`: MDS-based tone dispersion visualization.
- `/energy`: Scripts for RMS energy extraction.
- `/f0_h1h2_animation`: Interactive exploration of tonal features.
- `/old`: Archive of earlier scripts and prototypes.

### Main Scripts
- `f0_analysis.py` – Extracts F0 from tokens.
- `duration_analysis.py` – Measures duration.
- `h1h2_analysis.py` – Extracts H1–H2.
- `HNR_analysis.py` – Extracts HNR using VoiceSauce output.
- `tone_pattern.py` – Clustering base script.
- `tone_pattern_kmeans_*.py` – Variants using different feature sets.
- `plot_f0_new.py`, `plot_h1h2_new.py` – For visualizing tone trajectories and spectral tilt.

## Where to Get Help

If you encounter issues or have questions about this repository, please reach out via GitHub Issues or contact [Xiaojing Du](https://github.com/duxiaojing-lingo).

## Citation

If you use this project in your own work, please cite:

> Du, Xiaojing. 2025. *Checked and Neutral Tones in Huoji Chinese: A Multidimensional Analysis of Pitch, Duration, and Phonation*. MPhil Thesis, University of Cambridge.

---

This repository is part of an academic project and is maintained by Xiaojing Du. Contributions, questions, and feedback are welcome!
