# -*- coding: utf-8 -*-
"""
Analysis of Path Signature Effect on DeepSurv Performance
===========================================================

This script compares the training performance (C-Index, Loss) of the DeepSurv model
with and without Path Signature features.

Scenarios:
1. Baseline: Feature Set 'full' (Basic + Dynamic + Stat)
2. With Signature: Feature Set 'advanced' (Baseline + Path Signature)

Output:
- comparison_results/path_signature_comparison.png

Author: ESS DeepSurv Research Team
Date: 2026
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import DataConfig, get_dataloaders
from src.models import LSTMDeepSurv
from src.trainer import DeepSurvTrainer

def run_experiment(feature_set, epochs=30):
    """Run training for a specific feature set configuration"""
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
    
    # Load Data (Cache will be used if available)
    train_loader, val_loader = get_dataloaders(config, batch_size=32)
    
    # Determine input dim
    sample_batch = next(iter(train_loader))[0]
    input_dim = sample_batch.shape[-1]
    print(f"Input Dimension: {input_dim}")
    
    # 2. Model Init
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using feature set '{feature_set}' on device: {device}")
    
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
        patience=10 # Slightly shorter patience for quick comparison
    )
    
    history = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=5)
    return history

def main():
    # Define experiment configurations
    # 'full' includes Basic, Dynamic, Stat features (No Signature)
    # 'advanced' includes Full + Path Signature
    
    print("Starting Comparison Analysis...")
    
    # 1. Baseline Experiment (Without Path Signature)
    print("\n[1/2] Baseline Model (Without Path Signature)")
    hist_baseline = run_experiment('full', epochs=50)
    
    # 2. Advanced Experiment (With Path Signature)
    print("\n[2/2] Advanced Model (With Path Signature)")
    hist_signature = run_experiment('advanced', epochs=50)
    
    # 3. Visualization
    print("\nGenerating Comparison Plots...")
    
    epochs_base = range(1, len(hist_baseline['val_cindex']) + 1)
    epochs_sig = range(1, len(hist_signature['val_cindex']) + 1)
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: C-Index Comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs_base, hist_baseline['val_cindex'], 'b--', label='Without Path Signature', linewidth=2)
    plt.plot(epochs_sig, hist_signature['val_cindex'], 'r-', label='With Path Signature', linewidth=2)
    plt.title('Validation C-Index Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('C-Index', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Loss Comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs_base, hist_baseline['val_loss'], 'b--', label='Without Path Signature', linewidth=2)
    plt.plot(epochs_sig, hist_signature['val_loss'], 'r-', label='With Path Signature', linewidth=2)
    plt.title('Validation Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Cox PH)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save Plot
    save_dir = os.path.join(project_root, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'path_signature_comparison.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Comparison plot saved to: {save_path}")

if __name__ == "__main__":
    main()
