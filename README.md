#Final_DS_Project

# Neuro Deep Learning: Motor Imagery Classification

Final project for the Advanced Python / Deep Learning Workshop.
This repository implements the Deep ConvNet architecture (Schirrmeister et al., 2017) on the BCI Competition IV-2a dataset.

## Team
* **Helen:** Data Engineering & Preprocessing Pipeline
* **Peter:** Trial Segmentation & Cropped Training Loop
* **Margaret:** Deep ConvNet Architecture & Model Definition

## Quick Start

### 1. Install Dependencies
Run this command to install all required libraries:
```bash
pip install mne moabb torch scikit-learn pandas matplotlib seaborn

```

### 2. Run the Data Pipeline
To download the dataset (automatically handled via MOABB), filter, normalize, and visualize the data, run:
```bash
python -m src.test_run
```
Note: The first run will take a few minutes to download the BCI IV-2a dataset (~85MB per subject).

Output: Check the results/ folder for PSD plots and raw EEG traces.

### 3. Project structure:
**src/data_loader.py:** Handles downloading (MOABB), 4-38Hz filtering, and Z-score normalization.

**src/visualization.py:** Generates PSD (Power Spectral Density) and raw signal plots.

**src/config.py:** Central configuration (Sampling Rate, Channel Names, Filter settings) that can be adjusted to one's liking. 

**src/test_run.py:** The main script to run the pipeline and generate results.