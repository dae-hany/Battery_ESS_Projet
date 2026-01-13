import numpy as np
import iisignature
import torch

def extract_path_signature(voltage: np.ndarray, current: np.ndarray, time: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Extract Path Signature from Voltage, Current, and Time series data.
    
    Args:
        voltage (np.ndarray): Voltage array of shape (T,).
        current (np.ndarray): Current array of shape (T,).
        time (np.ndarray): Time array of shape (T,).
        level (int): Depth of the signature (default: 2).
        
    Returns:
        np.ndarray: Path Signature feature vector.
    """
    # 1. Stack data to form the path: Stream = [Time, Voltage, Current]
    # Normalizing time to start at 0 and roughly scale is good practice, 
    # but here we assume inputs are already reasonable or handled by caller.
    # For better numerics, we can normalize locally if needed, but let's stick to raw values first
    # or simple MinMax if the range varies wildly. 
    # Let's standardize mainly to ensure numerical stability for signatures.
    
    # Simple formatting
    time = time.reshape(-1, 1)
    voltage = voltage.reshape(-1, 1)
    current = current.reshape(-1, 1)
    
    path = np.hstack([time, voltage, current]) # Shape (T, 3)
    
    # 2. (Optional) Lead-Lag transformation could be added here if 'roughness' is critical,
    # but for battery data, direct linear interpolation (default in iisignature) is usually sufficient.
    
    # 3. Calculate Signature
    # iisignature.sig expects a 2D array of shape (N, d) where N is path length, d is dimensions.
    signature = iisignature.sig(path, level)
    
    return signature.astype(np.float32)

if __name__ == "__main__":
    # Simple test
    t = np.linspace(0, 10, 100)
    v = np.sin(t)
    i = np.cos(t)
    sig = extract_path_signature(v, i, t, level=2)
    print(f"Signature shape: {sig.shape}")
    print(f"Signature values: {sig}")
