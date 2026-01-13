import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import DataConfig, BatteryDataProcessor, get_dataloaders
from src.models import LSTMDeepSurv
from src.trainer import DeepSurvTrainer
from src.metrics import BreslowEstimator, calculate_expected_rul, calculate_rmse

def train_and_evaluate(feature_set: str, epochs: int = 50) -> Dict[str, float]:
    """
    특정 Feature Set으로 모델을 학습하고 Best Validation C-Index와 RMSE를 반환합니다.
    """
    print(f"\n" + "="*50)
    print(f"Running Experiment with Feature Set: {feature_set}")
    print("="*50)
    
    # 1. Config & Data Loading
    config = DataConfig(
        base_dir=project_root,
        feature_set_name=feature_set,
        window_size=10,
        min_cycles=50
    )
    
    train_loader, val_loader = get_dataloaders(config, batch_size=32)
    
    # Check input dim
    sample_batch = next(iter(train_loader))[0]
    input_dim = sample_batch.shape[-1]
    
    # 2. Model Init
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMDeepSurv(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # 3. Training
    trainer = DeepSurvTrainer(
        model=model,
        device=device,
        lr=0.001,
        patience=15
    )
    
    history = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=10)
    
    # 4. Evaluation (Breslow & RMSE)
    # Need to get all training data risk scores to fit Breslow
    model.eval()
    
    # - Collect Train Data (for Breslow fit)
    train_risks = []
    train_times = []
    train_events = []
    
    with torch.no_grad():
        for bx, bt, be in train_loader:
            bx = bx.to(device)
            out = model(bx)
            train_risks.append(out.cpu().numpy())
            train_times.append(bt.numpy())
            train_events.append(be.numpy())
            
    train_risks = np.concatenate(train_risks)
    train_times = np.concatenate(train_times)
    train_events = np.concatenate(train_events)
    
    # - Fit Breslow
    breslow = BreslowEstimator()
    breslow.fit(train_risks, train_times, train_events)
    
    # - Predict on Validation (for RMSE)
    val_risks = []
    val_times = []
    
    with torch.no_grad():
        for bx, bt, be in val_loader:
            bx = bx.to(device)
            out = model(bx)
            val_risks.append(out.cpu().numpy())
            val_times.append(bt.numpy())
            
    val_risks = np.concatenate(val_risks)
    val_times = np.concatenate(val_times)
    
    # Survival Function
    surv_df = breslow.get_survival_function(val_risks)
    
    # Expected RUL
    pred_rul = calculate_expected_rul(surv_df)
    
    # RMSE
    rmse_val = calculate_rmse(val_times, pred_rul)
    
    best_idx = np.argmax(history['val_cindex'])
    best_cindex = history['val_cindex'][best_idx]
    best_epoch = best_idx + 1
    
    print(f"[Result] Best C-Index: {best_cindex:.4f}")
    print(f"[Result] Val RMSE: {rmse_val:.4f}")
    
    return {
        'c_index': best_cindex,
        'rmse': rmse_val,
        'best_epoch': best_epoch,
        'input_dim': input_dim,
        'history': history,
        'feature_set': feature_set
    }

def main():
    print("Optimization: Comparison of Path Signature Impact")
    
    # Experiment 1: Baseline (Dynamic Features)
    # Features: capacity, soh, derivatives...
    baseline_res = train_and_evaluate('dynamic')
    
    # Experiment 2: Advanced (Path Signature)
    # Features: Baseline + Path Signatures
    advanced_res = train_and_evaluate('advanced')

    # Experiment 3: All Features (Path Signature + Delta Q)
    # Features: Advanced + Delta Q(V)
    all_res = train_and_evaluate('all')
    
    # Summary
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON SUMMARY (C-Index & RMSE)")
    print("="*50)
    print(f"{'Metric':<25} | {'Baseline':<12} | {'+ Path Sig':<12} | {'+ Delta Q':<12}")
    print("-" * 75)
    print(f"{'Input Dimension':<25} | {baseline_res['input_dim']:<12} | {advanced_res['input_dim']:<12} | {all_res['input_dim']:<12}")
    print(f"{'Best Val C-Index':<25} | {baseline_res['c_index']:<12.4f} | {advanced_res['c_index']:<12.4f} | {all_res['c_index']:<12.4f}")
    print(f"{'Val RMSE (RUL)':<25} | {baseline_res['rmse']:<12.2f} | {advanced_res['rmse']:<12.2f} | {all_res['rmse']:<12.2f}")
    print("-" * 75)
    
    diff_sig = advanced_res['c_index'] - baseline_res['c_index']
    diff_delta_q = all_res['c_index'] - advanced_res['c_index']
    
    print(f"Impact of Path Sig: {diff_sig:+.4f}")
    print(f"Impact of Delta Q : {diff_delta_q:+.4f}")

if __name__ == "__main__":
    main()
