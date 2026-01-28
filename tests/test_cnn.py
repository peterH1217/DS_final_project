import pytest
import torch
from neuro_deep_learning.cnn import DeepConvNet

def test_model_forward_pass():
    """Test if the model accepts input and produces correct output shape."""
    # 1. Configuration
    n_channels = 22
    n_classes = 4
    input_window = 500
    batch_size = 5
    
    # 2. Instantiate Model
    model = DeepConvNet(n_channels=n_channels, n_classes=n_classes, input_window_samples=input_window)
    
    # 3. Create dummy input (Batch, 1, Channels, Time) <--- FIX: Added dimension
    # Schirrmeister architecture expects a 4D input where the second dim is "1" (like a grayscale image)
    dummy_input = torch.randn(batch_size, 1, n_channels, input_window)
    
    # 4. Forward pass
    output = model(dummy_input)
    
    # 5. Assertions
    # Output should be (Batch, n_classes)
    assert output.shape == (batch_size, n_classes)
    # Output should be logits (unbounded), not probabilities
    assert output.requires_grad == True

def test_model_on_cpu():
    """Ensure model initializes on CPU by default."""
    model = DeepConvNet(n_channels=22, n_classes=4, input_window_samples=500)
    assert next(model.parameters()).device.type == 'cpu'