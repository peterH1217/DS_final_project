import pytest
from pathlib import Path
from neuro_deep_learning.config import PATHS, SAMPLING_RATE

def test_paths_exist():
    """Test if the configuration resolves paths correctly."""
    assert isinstance(PATHS.project_root, Path)
    assert isinstance(PATHS.data, Path)
    assert isinstance(PATHS.results, Path)

def test_parameters():
    """Test key data parameters."""
    assert SAMPLING_RATE == 250
    # Ensure cutoff makes sense (Low < High or High is None)
    from neuro_deep_learning.config import LOW_CUTOFF, HIGH_CUTOFF
    if HIGH_CUTOFF is not None:
        assert LOW_CUTOFF < HIGH_CUTOFF