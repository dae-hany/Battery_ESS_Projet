import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis

def interpolate_qv(voltage, capacity, v_grid):
    """
    Inteprolate Q(V) curve onto a common voltage grid.
    
    Args:
        voltage: Measured voltage array (must be monotonic ideally, or sorted)
        capacity: Measured capacity array
        v_grid: Common voltage grid
        
    Returns:
        Interpolated capacity on v_grid
    """
    # Remove process duplicates or non-monotonicity if needed
    # For discharge, Voltage decreases. We want to interpolate Q as func of V.
    # Uniqueness constraint for interpolation.
    
    # Sort by voltage (Discharge: High -> Low, Charge: Low -> High)
    # We generally treat V as x-axis.
    
    if len(voltage) < 2:
        return np.zeros_like(v_grid)
        
    # Remove duplicates
    _, unique_indices = np.unique(voltage, return_index=True)
    v_clean = voltage[unique_indices]
    q_clean = capacity[unique_indices]
    
    # Sort
    sort_idx = np.argsort(v_clean)
    v_clean = v_clean[sort_idx]
    q_clean = q_clean[sort_idx]
    
    try:
        f = interp1d(v_clean, q_clean, kind='linear', fill_value="extrapolate")
        q_interp = f(v_grid)
        return q_interp
    except Exception as e:
        print(f"Interpolation error: {e}")
        return np.zeros_like(v_grid)

def extract_delta_q(voltage: np.ndarray, capacity: np.ndarray, 
                   ref_q_interp: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
    """
    Calculate Delta Q(V) features for a single cycle given a reference Q(V) curve.
    
    Args:
        voltage (np.ndarray): Voltage vector of current cycle.
        capacity (np.ndarray): Capacity vector of current cycle.
        ref_q_interp (np.ndarray): Interpolated Q(V) of reference cycle (on v_grid).
        v_grid (np.ndarray): Common voltage grid used for interpolation.
        
    Returns:
        np.ndarray: Feature vector [var, min, mean, skew, kurtosis] of Delta Q(V).
    """
    # 1. Interpolate current cycle Q(V) onto grid
    curr_q_interp = interpolate_qv(voltage, capacity, v_grid)
    
    # 2. Calculate Delta Q(V) = Q_curr(V) - Q_ref(V)
    delta_q_curve = curr_q_interp - ref_q_interp
    
    # 3. Extract Statistics
    # Historically, Variance of Delta Q is highly correlated with SOH.
    feat_var = np.var(delta_q_curve)
    feat_min = np.min(delta_q_curve)
    feat_mean = np.mean(delta_q_curve)
    feat_skew = skew(delta_q_curve)
    feat_kurt = kurtosis(delta_q_curve)
    
    # Range (Max - Min)
    feat_range = np.max(delta_q_curve) - feat_min
    
    features = np.array([feat_var, feat_min, feat_mean, feat_skew, feat_kurt, feat_range])
    
    # Replace NaNs if any (due to flat constant curves etc)
    features = np.nan_to_num(features)
    
    return features
