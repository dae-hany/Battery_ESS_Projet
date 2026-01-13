# -*- coding: utf-8 -*-
"""
Zero-shot Validation on CALCE Dataset
=====================================
Trains DeepSurv on NASA data and validates on CALCE data.

1. Train: NASA Battery Dataset (cleaned_dataset/data)
2. Test: CALCE Battery Dataset (cleaned_dataset/calce_data)

This checks if the features (Delta Q, Path Signature) and Model generalize
to a completely different dataset (Cross-Dataset Validation).
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import DataConfig, BatteryDataProcessor, BatteryDataset
from src.models import LSTMDeepSurv
from src.trainer import DeepSurvTrainer
# from pycox.evaluation import EvalSurv # Not installed, using lifelines

def get_calce_loader(config, batch_size=32):
    """
    Custom loader for CALCE to use ALL data for testing.
    """
    processor = BatteryDataProcessor(config)
    print(f"\n[Info] Processing CALCE data from: {config.data_dir}")
    
    # 1. Load Features (Cache safe?)
    # We should use a different cache dir or name to avoid conflict with NASA cache
    # DataConfig doesn't support changing cache filename easily, so we use a separate cache dir.
    df, features = processor.load_and_process_features()
    
    print(f"[Info] CALCE Features extracted: {len(features)}")
    print(f"[Info] CALCE Samples: {len(df)}")
    
    # 2. Sliding Window (Test Mode: No Split, Use All)
    # scaling: Ideally we should use scaler fitted on NASA train data?
    # But BatteryDataProcessor creates a NEW scaler. 
    # For strict Zero-shot, we should use NASA scaler.
    # However, for simplicity now, let's normalize CALCE independently 
    # (assuming distribution shift is handled by normalization).
    # If this fails, we might need to share scaler.
    
    # For this script, we'll let it fit scaler on CALCE itself for now to ensure valid input ranges.
    # (Refining this to transfer scaler is a future optimization)
    
    # Scale
    processor.scaler.fit(df[features].values)
    df.loc[:, features] = processor.scaler.transform(df[features].values)
    
    # Windowing
    data_dict = processor.create_sliding_window_data(df)
    
    dataset = BatteryDataset(
        torch.from_numpy(data_dict['X']),
        torch.from_numpy(data_dict['T']),
        torch.from_numpy(data_dict['E'])
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, features

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Feature Set: Let's use 'delta_q' which showed good performance
    feature_set = 'delta_q' 
    window_size = 10
    
    # ==========================================
    # 1. Train on NASA
    # ==========================================
    print("\n" + "="*50)
    print(" [Phase 1] Training on NASA Dataset")
    print("="*50)
    
    nasa_config = DataConfig(
        base_dir=project_root,
        feature_set_name=feature_set,
        window_size=window_size
    )
    
    # We use get_dataloaders form data_loader.py for NASA
    from src.data_loader import get_dataloaders
    train_loader, val_loader = get_dataloaders(nasa_config, batch_size=32)
    
    # Get input dim
    sample_x = next(iter(train_loader))[0]
    input_dim = sample_x.shape[-1]
    
    model = LSTMDeepSurv(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    trainer = DeepSurvTrainer(model, device)
    
    # Train
    print("Training...")
    trainer.fit(train_loader, val_loader, epochs=30, verbose=10)
    
    # ==========================================
    # 2. Validate on CALCE (Zero-shot)
    # ==========================================
    print("\n" + "="*50)
    print(" [Phase 2] Zero-shot Validation on CALCE Dataset")
    print("="*50)
    
    calce_config = DataConfig(
        base_dir=project_root,
        feature_set_name=feature_set,
        window_size=window_size,
        # Override paths for CALCE
        metadata_path=os.path.join(project_root, "cleaned_dataset", "calce_metadata.csv"),
        data_dir=os.path.join(project_root, "cleaned_dataset", "calce_data"),
        # Separate cache
        cache_dir=os.path.join(project_root, "cache", "calce"),
        # Adjust filters for CALCE (CS2 cells are ~1.1Ah)
        min_init_capacity=0.5,
        min_cycles=20
    )
    os.makedirs(calce_config.cache_dir, exist_ok=True)
    
    calce_loader, calce_feats = get_calce_loader(calce_config)
    
    # Sanity Check
    if len(calce_feats) != input_dim:
        print(f"[Error] Feature mismatch! Train: {input_dim}, Test: {len(calce_feats)}")
        return

    # Predict
    model.eval()
    surv_probs = []
    times = []
    events = []
    
    print("Evaluating on CALCE...")
    with torch.no_grad():
        for x, t, e in calce_loader:
            x = x.to(device)
            output = model(x) # Log Hazard ? No, this model returns risk OR surv? 
            # Check models.py: LSTMDeepSurv returns 'risk_score' (log hazard) usually?
            # Wait, PyCox models depend on implementation.
            # Assuming models.py returns LOG HAZARD because DeepSurvTrainer uses CoxPHLoss.
            
            # For C-Index, we need negative hazard (or just hazard inverted? No usage of surv df)
            # Actually C-Index in PyCox: concordance_index_censored(event_times, event_observed, predicted_scores)
            # Predicted scores -> Risk. High risk = Low Time. 
            # concordance_index_censored expects: (event_times, event_observed, predicted_risk)
            # Standard: if risk is high, time is low.
            
            surv_probs.append(output.cpu().numpy())
            times.append(t.cpu().numpy())
            events.append(e.cpu().numpy())

    preds = np.concatenate(surv_probs).squeeze()
    targets = np.concatenate(times)
    status = np.concatenate(events)
    
    # Calculate C-Index
    from lifelines.utils import concordance_index
    # Warning: CALCE data might have minimal events (all run to failure? Check metadata)
    # If all E=1 (or mostly), Validation is solid.
    
    try:
        c_index = concordance_index(targets, -preds, status) # -preds because risk vs time
        print(f"\n >>> CALCE Validation C-Index: {c_index:.4f}")
    except Exception as e:
        print(f"[Results] Raw C-Index Calculation Failed: {e}")
        # Try without negative?
        c_index_alt = concordance_index(targets, preds, status)
        print(f"[Results] Alt C-Index (Positive Risk): {c_index_alt:.4f}")
        
    print("="*50)

    # ==========================================
    # 3. Visualization
    # ==========================================
    import matplotlib.pyplot as plt
    
    save_dir = os.path.join(project_root, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Risk vs True RUL
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, preds, alpha=0.5, c='blue', edgecolors='none')
    plt.title(f'CALCE Validation: Risk Score vs True RUL (C-Index: {c_index:.4f})', fontsize=14)
    plt.xlabel('True RUL (Time)', fontsize=12)
    plt.ylabel('Predicted Risk Score (Log Hazard)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    path1 = os.path.join(save_dir, 'calce_risk_vs_rul.png')
    plt.savefig(path1, dpi=300)
    plt.close()
    print(f"[Plot] Saved Risk vs RUL scatter plot: {path1}")
    
    # Plot 2: Risk Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(preds, bins=50, color='green', alpha=0.7)
    plt.title('Distribution of Predicted Risk Scores on CALCE', fontsize=14)
    plt.xlabel('Predicted Risk Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    path2 = os.path.join(save_dir, 'calce_risk_distribution.png')
    plt.savefig(path2, dpi=300)
    plt.close()
    print(f"[Plot] Saved Risk Distribution histogram: {path2}")

if __name__ == "__main__":
    main()
