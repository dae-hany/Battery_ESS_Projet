# -*- coding: utf-8 -*-
"""
Battery ESS Project - Model Performance Comparison
==================================================
Baseline (MLP-DeepSurv) vs Advanced (LSTM-DeepSurv)

Author: ESS DeepSurv Research Team
Date: 2025-12-24
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import medfilt

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.data_loader import DataConfig, get_dataloaders
from src.models import LSTMDeepSurv
from src.trainer import DeepSurvTrainer, SurvivalMetrics, cox_ph_loss

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# Baseline MLP Model
# ============================================================================

class BaselineDeepSurv(nn.Module):
    """MLP-based DeepSurv model"""
    
    def __init__(self, input_dim, hidden_layers=[64, 32, 16], dropout=0.3):
        super(BaselineDeepSurv, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class StaticDataset(Dataset):
    """Dataset for static features"""
    
    def __init__(self, features, times, events):
        self.features = torch.FloatTensor(features)
        self.times = torch.FloatTensor(times)
        self.events = torch.FloatTensor(events)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.times[idx], self.events[idx]


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_baseline_data(base_dir, test_size=0.2, random_seed=42):
    """Prepare static features for baseline model"""
    
    metadata_path = os.path.join(base_dir, "cleaned_dataset", "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    metadata.columns = [c.strip().lower() for c in metadata.columns]
    
    discharge_df = metadata[metadata['type'] == 'discharge'].copy()
    discharge_df['capacity'] = pd.to_numeric(discharge_df['capacity'], errors='coerce')
    discharge_df = discharge_df.dropna(subset=['capacity'])
    
    def preprocess_capacity(capacities, threshold=0.5):
        cleaned = capacities.copy()
        for i in range(1, len(cleaned)):
            prev_val = cleaned[i-1]
            curr_val = cleaned[i]
            if prev_val > 0:
                drop_ratio = (prev_val - curr_val) / prev_val
                if drop_ratio > threshold or curr_val < 0.1:
                    cleaned[i] = prev_val
        if len(cleaned) >= 3:
            cleaned = medfilt(cleaned, kernel_size=3)
        return cleaned
    
    all_samples = []
    min_init_capacity = 1.5
    min_cycles = 50
    soh_threshold = 0.8
    
    for bat_id in discharge_df['battery_id'].unique():
        bat_df = discharge_df[discharge_df['battery_id'] == bat_id].copy()
        bat_df = bat_df.sort_values('test_id')
        
        if len(bat_df) < min_cycles:
            continue
        
        init_cap = bat_df['capacity'].iloc[0]
        if init_cap < min_init_capacity:
            continue
        
        # Preprocess
        bat_df['capacity'] = preprocess_capacity(bat_df['capacity'].values)
        bat_df['capacity_smooth'] = bat_df['capacity'].rolling(
            window=7, min_periods=1, center=True
        ).mean().fillna(bat_df['capacity'])
        
        bat_df['soh'] = bat_df['capacity_smooth'] / init_cap
        
        # EOL
        eol_indices = bat_df.index[bat_df['soh'] < soh_threshold].tolist()
        if eol_indices:
            eol_cycle = bat_df.index.get_loc(eol_indices[0])
            event = 1
        else:
            eol_cycle = len(bat_df)
            event = 0
        
        # Features
        bat_df['soh_derivative'] = bat_df['soh'].diff().fillna(0)
        bat_df['capacity_derivative'] = bat_df['capacity'].diff().fillna(0)
        
        # Samples
        max_k = eol_cycle if event == 1 else len(bat_df)
        for k in range(max_k):
            row = bat_df.iloc[k]
            all_samples.append({
                'battery_id': bat_id,
                'capacity': row['capacity'],
                'soh': row['soh'],
                'soh_derivative': row['soh_derivative'],
                'capacity_derivative': row['capacity_derivative'],
                'time': eol_cycle - k,
                'event': event
            })
    
    df = pd.DataFrame(all_samples)
    feature_names = ['capacity', 'soh', 'soh_derivative', 'capacity_derivative']
    
    for col in feature_names:
        df[col] = df[col].fillna(0)
    
    # Split
    battery_ids = df['battery_id'].unique()
    train_bats, test_bats = train_test_split(
        battery_ids, test_size=test_size, random_state=random_seed
    )
    
    train_df = df[df['battery_id'].isin(train_bats)]
    test_df = df[df['battery_id'].isin(test_bats)]
    
    X_train = train_df[feature_names].values.astype(np.float32)
    T_train = train_df['time'].values.astype(np.float32)
    E_train = train_df['event'].values.astype(np.float32)
    
    X_test = test_df[feature_names].values.astype(np.float32)
    T_test = test_df['time'].values.astype(np.float32)
    E_test = test_df['event'].values.astype(np.float32)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    print(f"\n[Baseline Data]")
    print(f"  Train: {len(train_bats)} batteries, {len(X_train):,} samples")
    print(f"  Test: {len(test_bats)} batteries, {len(X_test):,} samples")
    
    return {
        'X_train': X_train, 'T_train': T_train, 'E_train': E_train,
        'X_test': X_test, 'T_test': T_test, 'E_test': E_test,
        'feature_names': feature_names
    }


# ============================================================================
# Training
# ============================================================================

def train_baseline(data, epochs=100, lr=1e-3, device='cpu'):
    """Train baseline MLP model"""
    
    print("\n" + "="*60)
    print("Training Baseline (MLP)")
    print("="*60)
    
    model = BaselineDeepSurv(
        input_dim=data['X_train'].shape[1],
        hidden_layers=[64, 32, 16],
        dropout=0.3
    ).to(device)
    
    train_dataset = StaticDataset(data['X_train'], data['T_train'], data['E_train'])
    test_dataset = StaticDataset(data['X_test'], data['T_test'], data['E_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {'train_loss': [], 'val_loss': [], 'train_cindex': [], 'val_cindex': []}
    best_val_cindex = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        train_risks, train_times, train_events = [], [], []
        
        for batch_x, batch_t, batch_e in train_loader:
            batch_x = batch_x.to(device)
            batch_t = batch_t.to(device)
            batch_e = batch_e.to(device)
            
            risk = model(batch_x)
            loss = cox_ph_loss(risk, batch_t, batch_e)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_risks.append(risk.detach().cpu().numpy())
            train_times.append(batch_t.cpu().numpy())
            train_events.append(batch_e.cpu().numpy())
        
        train_loss = np.mean(train_losses)
        train_risks = np.concatenate(train_risks)
        train_times = np.concatenate(train_times)
        train_events = np.concatenate(train_events)
        train_cindex = SurvivalMetrics.concordance_index(train_risks, train_times, train_events)
        
        # Validation
        model.eval()
        val_losses = []
        val_risks, val_times, val_events = [], [], []
        
        with torch.no_grad():
            for batch_x, batch_t, batch_e in test_loader:
                batch_x = batch_x.to(device)
                batch_t = batch_t.to(device)
                batch_e = batch_e.to(device)
                
                risk = model(batch_x)
                loss = cox_ph_loss(risk, batch_t, batch_e)
                
                val_losses.append(loss.item())
                val_risks.append(risk.cpu().numpy())
                val_times.append(batch_t.cpu().numpy())
                val_events.append(batch_e.cpu().numpy())
        
        val_loss = np.mean(val_losses)
        val_risks = np.concatenate(val_risks)
        val_times = np.concatenate(val_times)
        val_events = np.concatenate(val_events)
        val_cindex = SurvivalMetrics.concordance_index(val_risks, val_times, val_events)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_cindex'].append(train_cindex)
        history['val_cindex'].append(val_cindex)
        
        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train C={train_cindex:.4f}, Val C={val_cindex:.4f}")
        
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"Best Val C-Index: {best_val_cindex:.4f}")
    return model, history, best_val_cindex


def train_lstm(data_config, best_params, epochs=100, device='cpu'):
    """Train LSTM model"""
    
    print("\n" + "="*60)
    print("Training Advanced (LSTM)")
    print("="*60)
    
    data_config.window_size = best_params.get('window_size', 10)
    
    train_loader, val_loader = get_dataloaders(data_config, batch_size=32)
    sample = next(iter(train_loader))[0]
    input_dim = sample.shape[-1]
    
    model = LSTMDeepSurv(
        input_dim=input_dim,
        hidden_dim=best_params.get('hidden_dim', 64),
        num_layers=best_params.get('num_layers', 2),
        dropout=best_params.get('dropout', 0.3)
    ).to(device)
    
    trainer = DeepSurvTrainer(
        model=model,
        device=device,
        lr=best_params.get('learning_rate', 1e-3),
        patience=20
    )
    
    print(f"Params: window={data_config.window_size}, hidden={best_params.get('hidden_dim')}")
    
    best_cindex = 0
    for epoch in range(epochs):
        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        
        trainer.history['train_loss'].append(train_metrics['loss'])
        trainer.history['train_cindex'].append(train_metrics['c_index'])
        trainer.history['val_loss'].append(val_metrics['loss'])
        trainer.history['val_cindex'].append(val_metrics['c_index'])
        
        if val_metrics['c_index'] > best_cindex:
            best_cindex = val_metrics['c_index']
            trainer.best_model_state = model.state_dict().copy()
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train C={train_metrics['c_index']:.4f}, Val C={val_metrics['c_index']:.4f}")
        
        if trainer.patience_counter >= trainer.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if trainer.best_model_state is not None:
        model.load_state_dict(trainer.best_model_state)
    
    print(f"Best Val C-Index: {best_cindex:.4f}")
    return trainer, best_cindex


# ============================================================================
# Visualization
# ============================================================================

def save_plot(filename, save_dir):
    """Helper to save plots"""
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_losses(baseline_hist, lstm_hist, save_dir):
    """Plot loss curves"""
    
    # Training loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(baseline_hist['train_loss'])+1), baseline_hist['train_loss'], 
            'b-', label='Baseline', linewidth=2.5, alpha=0.7)
    ax.plot(range(1, len(lstm_hist['train_loss'])+1), lstm_hist['train_loss'], 
            'r-', label='LSTM', linewidth=2.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Training Loss', fontsize=13)
    ax.set_title('Training Loss Comparison', fontweight='bold', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('01_training_loss.png', save_dir)
    
    # Validation loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(baseline_hist['val_loss'])+1), baseline_hist['val_loss'], 
            'b--', label='Baseline', linewidth=2.5, alpha=0.7)
    ax.plot(range(1, len(lstm_hist['val_loss'])+1), lstm_hist['val_loss'], 
            'r--', label='LSTM', linewidth=2.5, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation Loss', fontsize=13)
    ax.set_title('Validation Loss Comparison', fontweight='bold', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('02_validation_loss.png', save_dir)


def plot_cindex(baseline_hist, lstm_hist, save_dir):
    """Plot C-Index curves"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(baseline_hist['val_cindex'])+1), baseline_hist['val_cindex'], 
            'b-', label='Baseline', linewidth=2.5, marker='o', markersize=4, alpha=0.7)
    ax.plot(range(1, len(lstm_hist['val_cindex'])+1), lstm_hist['val_cindex'], 
            'r-', label='LSTM', linewidth=2.5, marker='s', markersize=4, alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=2, label='Random')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation C-Index', fontsize=13)
    ax.set_title('C-Index Progress', fontweight='bold', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    save_plot('03_cindex_progress.png', save_dir)


def plot_final_comparison(baseline_cindex, lstm_cindex, save_dir):
    """Plot final C-Index bar chart"""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    models = ['Baseline\n(MLP)', 'Advanced\n(LSTM)']
    values = [baseline_cindex, lstm_cindex]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(models, values, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=2.5, width=0.6)
    ax.set_ylabel('C-Index', fontweight='bold', fontsize=13)
    ax.set_title('Final C-Index Comparison', fontweight='bold', fontsize=16)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                f'{val:.4f}', ha='center', va='bottom', 
                fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    save_plot('04_final_cindex.png', save_dir)


def plot_improvement(baseline_cindex, lstm_cindex, save_dir):
    """Plot improvement percentage"""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    improvement = ((lstm_cindex - baseline_cindex) / baseline_cindex) * 100
    
    ax.text(0.5, 0.7, 'Performance Improvement', 
            ha='center', va='center', fontsize=20, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.4, f'{improvement:+.2f}%', 
            ha='center', va='center', fontsize=50, fontweight='bold',
            color='green' if improvement > 0 else 'red',
            transform=ax.transAxes)
    ax.text(0.5, 0.2, f'ΔC-Index = {lstm_cindex - baseline_cindex:+.4f}', 
            ha='center', va='center', fontsize=16,
            transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    save_plot('05_improvement.png', save_dir)


def plot_architecture(save_dir):
    """Plot architecture comparison table"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    data = [
        ['Feature', 'Baseline (MLP)', 'Advanced (LSTM)'],
        ['Structure', 'Fully Connected', 'LSTM + MLP'],
        ['Input', '2D (Batch, Features)', '3D (Batch, Window, Features)'],
        ['Temporal Info', 'None (Static)', 'Time-series Learning'],
        ['Parameters', '~10K', '~50K'],
        ['Learning', 'Static Features', 'Temporal Patterns']
    ]
    
    table = ax.table(cellText=data, loc='center', cellLoc='left',
                    colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white', size=13)
    
    for i in range(1, len(data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax.set_title('Architecture Comparison', fontweight='bold', fontsize=16, pad=20)
    plt.tight_layout()
    save_plot('06_architecture.png', save_dir)


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  MODEL COMPARISON: Baseline vs Advanced  ")
    print("="*70)
    
    BASE_DIR = r"C:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    COMP_DIR = os.path.join(BASE_DIR, "comparison_results")
    os.makedirs(COMP_DIR, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load best params
    with open(os.path.join(RESULTS_DIR, "best_params.json"), 'r') as f:
        best_params = json.load(f)['best_params']
    
    print(f"Best params: {best_params}")
    
    # Prepare data
    print("\n" + "="*60)
    print("Preparing Data")
    print("="*60)
    
    baseline_data = prepare_baseline_data(BASE_DIR)
    
    lstm_config = DataConfig(
        base_dir=BASE_DIR,
        window_size=best_params.get('window_size', 10),
        feature_set_name='dynamic',
        test_size=0.2,
        random_seed=42
    )
    
    # Train models
    baseline_model, baseline_hist, baseline_cindex = train_baseline(
        baseline_data, epochs=100, device=device
    )
    
    lstm_trainer, lstm_cindex = train_lstm(
        lstm_config, best_params, epochs=100, device=device
    )
    
    # Visualize
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    plot_losses(baseline_hist, lstm_trainer.history, COMP_DIR)
    plot_cindex(baseline_hist, lstm_trainer.history, COMP_DIR)
    plot_final_comparison(baseline_cindex, lstm_cindex, COMP_DIR)
    plot_improvement(baseline_cindex, lstm_cindex, COMP_DIR)
    plot_architecture(COMP_DIR)
    
    # Save results
    results = {
        'baseline': {
            'architecture': 'MLP',
            'best_val_cindex': float(baseline_cindex),
            'epochs': len(baseline_hist['train_loss'])
        },
        'advanced': {
            'architecture': 'LSTM + MLP',
            'best_val_cindex': float(lstm_cindex),
            'epochs': len(lstm_trainer.history['train_loss']),
            'hyperparameters': best_params
        },
        'improvement': {
            'absolute': float(lstm_cindex - baseline_cindex),
            'relative_percent': float(((lstm_cindex - baseline_cindex) / baseline_cindex) * 100)
        }
    }
    
    with open(os.path.join(COMP_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY  ")
    print("="*70)
    print(f"\nBaseline (MLP): C-Index = {baseline_cindex:.4f}")
    print(f"Advanced (LSTM): C-Index = {lstm_cindex:.4f}")
    improvement = ((lstm_cindex - baseline_cindex) / baseline_cindex) * 100
    print(f"\n🎯 Improvement: {improvement:+.2f}% (ΔC-Index = {lstm_cindex - baseline_cindex:+.4f})")
    
    if lstm_cindex > baseline_cindex:
        print("\n✓ LSTM model outperforms baseline!")
    
    print(f"\n📁 Results saved to: {COMP_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
