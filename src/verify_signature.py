import os
import sys
import numpy as np
import pandas as pd

# 프로젝트 루트 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import DataConfig, BatteryDataProcessor

def verify_signature_extraction():
    print("=== Path Signature Integration Verification ===")
    
    # 1. Config Setup (Use 'advanced' feature set)
    config = DataConfig(
        base_dir=project_root,
        feature_set_name='advanced',  # <--- Key change
        min_cycles=30,  # Quick test
        window_size=10
    )
    
    # Check paths
    if not os.path.exists(config.data_dir):
        print(f"[Error] Data directory not found: {config.data_dir}")
        print("Please check the path or use the correct user path.")
        # Try to infer correct path if we are in a different env
        # For now, assume the hardcoded path in data_loader might be wrong for this environment if 'User' != 'daeho'
        # The user provided path: c:\Users\User\daehan_study\Battery_ESS_Projet
        # I should probably update the config to use the current working directory or the User's path if this fails.
        return

    print(f"[Info] Feature Set: {config.feature_set_name}")
    print(f"[Info] Cache Dir: {config.cache_dir}")
    
    processor = BatteryDataProcessor(config)
    
    # 2. Load and Process
    try:
        # Force no cache to test extraction logic? 
        # Or use cache if available (but likely not created yet)
        # Let's try with use_cache=False first to verify extraction logic.
        print("[Info] Starting extraction (use_cache=False)...")
        df, feature_names = processor.load_and_process_features(use_cache=False)
        
        print(f"[Success] Data loaded. Shape: {df.shape}")
        print(f"[Info] Feature Names ({len(feature_names)}): {feature_names}")
        
        # 3. Verify Signature Columns
        sig_cols = [c for c in feature_names if 'sig_' in c]
        if len(sig_cols) > 0:
            print(f"[Success] Found {len(sig_cols)} signature features.")
            print(f"Sample Signature (First row):")
            print(df[sig_cols].iloc[0].values)
            
            # Check for NaNs
            if df[sig_cols].isnull().any().any():
                print("[Warning] NaN values found in signature features!")
            else:
                print("[Success] No NaNs in signature features.")
        else:
            print("[Error] No signature features found! Check extraction logic.")
            
        # 4. Verify Dataset Creation
        print("[Info] Preparing Datasets...")
        train_ds, test_ds = processor.prepare_datasets(df)
        
        sample_x, sample_t, sample_e = train_ds[0]
        print(f"[Success] Dataset created.")
        print(f"Input Shape: {sample_x.shape}")
        print(f"Expected Input Dim: {len(feature_names)}")
        
        if sample_x.shape[1] == len(feature_names):
            print("[Pass] Input dimension matches feature count.")
        else:
            print(f"[Fail] Dimension Mismatch! Data: {sample_x.shape[1]}, Features: {len(feature_names)}")
            
    except Exception as e:
        print(f"[Error] Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Adjust base_dir dynamically for this environment
    # The user is in 'c:\Users\User\daehan_study\Battery_ESS_Projet'
    # The code has hardcoded 'C:\Users\daeho\...'
    # I should pass the correct base_dir.
    
    current_dir = os.getcwd() # likely c:\Users\User\daehan_study\Battery_ESS_Projet
    
    # Update DataConfig defaults in the function call is tricky as it's a dataclass
    # I will rely on the `base_dir` passed to DataConfig init.
    
    verify_signature_extraction()
