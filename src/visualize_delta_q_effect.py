# -*- coding: utf-8 -*-
"""
Analysis of Delta Q(V) Effect on DeepSurv Performance
===========================================================

This script compares the training performance (C-Index, Loss) of the DeepSurv model
with and without Delta Q(V) features.

Scenarios:
1. Baseline: Feature Set 'full' (Basic + Dynamic + Stat)
2. With Delta Q: Feature Set 'delta_q' (Baseline + Delta Q Statistics)

Output:
- plots/delta_q_comparison.png

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
    # 'full' includes Basic, Dynamic, Stat features (No Delta Q)
    # 'delta_q' includes Full + Delta Q Features
    
    print("Starting Comparison Analysis (Delta Q Effect)...")
    
    # 1. Baseline Experiment (Without Delta Q)
    print("\n[1/2] Baseline Model (Without Delta Q)")
    hist_baseline = run_experiment('full', epochs=50)
    
    # 2. Advanced Experiment (With Delta Q)
    print("\n[2/2] Advanced Model (With Delta Q)")
    hist_delta = run_experiment('delta_q', epochs=50)
    
    # 3. Visualization
    print("\nGenerating Comparison Plots...")
    
    epochs_base = range(1, len(hist_baseline['val_cindex']) + 1)
    epochs_delta = range(1, len(hist_delta['val_cindex']) + 1)
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: C-Index Comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs_base, hist_baseline['val_cindex'], 'b--', label='Without Delta Q', linewidth=2)
    plt.plot(epochs_delta, hist_delta['val_cindex'], 'g-', label='With Delta Q', linewidth=2)
    plt.title('Validation C-Index Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('C-Index', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Loss Comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs_base, hist_baseline['val_loss'], 'b--', label='Without Delta Q', linewidth=2)
    plt.plot(epochs_delta, hist_delta['val_loss'], 'g-', label='With Delta Q', linewidth=2)
    plt.title('Validation Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Cox PH)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save Plot
    save_dir = os.path.join(project_root, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'delta_q_comparison.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n[Success] Comparison plot saved to: {save_path}")

    # Print Summary Metrics
    print("\n" + "="*50)
    print(" >>> Final Performance Comparison <<<")
    print("="*50)
    print(f"Baseline (Full)   | Max C-Index: {np.max(hist_baseline['val_cindex']):.4f} | Min Loss: {np.min(hist_baseline['val_loss']):.4f}")
    print(f"Advanced (Delta Q)| Max C-Index: {np.max(hist_delta['val_cindex']):.4f} | Min Loss: {np.min(hist_delta['val_loss']):.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
