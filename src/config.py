# src/config.py
import os

# --- Paths ---
# We don't need a hardcoded path for MOABB, it handles downloads automatically.
# But we can define where we want to save logs or results.
LOG_DIR = "logs"
RESULTS_DIR = "results"

# --- Data Parameters (Dataset 2a) ---
SAMPLING_RATE = 250  # Hz
LOW_CUTOFF = 4.0     # Hz (High-pass filter)
HIGH_CUTOFF = 38.0   # Hz (Low-pass filter)

# The dataset has 22 EEG channels. We list them to be explicit.
EEG_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 
    'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 
    'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
]

# MOABB Dataset ID for BCI Competition IV 2a
DATASET_NAME = "BNCI2014001"