import pytest
import numpy as np
import torch
import mne
from neuro_deep_learning import dataset

def test_preprocess_data_shape():
    """Smoke test: Ensure preprocessing returns MNE Raw object with correct channels."""
    # 1. Create dummy raw data (22 channels, 10 seconds, 250Hz)
    n_channels = 22
    sfreq = 250
    data = np.random.randn(n_channels, 10 * sfreq)
    info = mne.create_info(ch_names=[str(i) for i in range(n_channels)], sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    # 2. Run your preprocessing
    raw_processed = dataset.preprocess_data(raw)
    
    # 3. Assertions
    assert isinstance(raw_processed, mne.io.BaseRaw)
    assert raw_processed.info['sfreq'] == 250
    # Check if Z-score normalization happened (mean should be close to 0)
    data_out = raw_processed.get_data()
    assert np.isclose(data_out.mean(), 0, atol=0.1)

def test_crops_dataset_logic():
    """Test if the sliding window (cropping) logic works."""
    # 1. Create dummy trials: 10 trials, 22 chans, 1000 timepoints
    n_trials = 10
    n_chans = 22
    n_time = 1000
    X = np.random.randn(n_trials, n_chans, n_time)
    y = np.random.randint(0, 4, size=n_trials)
    
    # 2. Initialize your Dataset class
    crop_size = 500
    stride = 500
    ds = dataset.CropsDataset(X, y, crop_size=crop_size, stride=stride)
    
    # 3. Assertions
    # With 1000 timepoints and 500 crop size, we expect exactly 2 crops per trial
    expected_crops_per_trial = 2
    assert len(ds) == n_trials * expected_crops_per_trial
    
    # Check shape of one item
    crop_data, label, trial_idx = ds[0]
    assert crop_data.shape == (n_chans, crop_size)
    assert isinstance(crop_data, torch.Tensor)