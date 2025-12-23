# -*- coding: utf-8 -*-
"""
===================================================================================
DeepSurv for Battery Remaining Useful Life (RUL) Prediction
===================================================================================

A comprehensive survival analysis framework for journal publication.

This script implements:
1. Exploratory Data Analysis (EDA)
2. Feature Engineering & Preprocessing
3. DeepSurv Model with Cox Proportional Hazards Loss
4. Comprehensive Metrics: C-index, Brier Score, IBS, Time-dependent AUC
5. Publication-ready Visualizations
6. Baseline Model Comparisons (Cox PH, Random Survival Forest)

Author: ESS DeepSurv Research Team
Date: 2025
"""

import os
import warnings
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import medfilt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Plotting style for publication
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
        sns.set_style("whitegrid")

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ===================================================================================
# Configuration
# ===================================================================================

@dataclass
class Config:
    """Configuration class for the experiment."""
    # Paths
    base_dir: str = "/Users/geonminkim/ess_deepsurv"
    metadata_path: str = None
    data_dir: str = None
    output_dir: str = None
    
    # Data parameters
    min_init_capacity: float = 1.5  # Minimum initial capacity (Ah)
    min_cycles: int = 50  # Minimum number of cycles
    soh_threshold: float = 0.8  # End-of-Life threshold (SOH)
    
    # Model parameters
    hidden_layers: List[int] = None
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 200
    patience: int = 30  # Early stopping patience
    
    # Cross-validation
    n_folds: int = 5
    
    # Feature configuration for ablation study
    feature_set_name: str = "full"  # Options: "basic", "dynamic", "full"
    
    def __post_init__(self):
        self.metadata_path = os.path.join(self.base_dir, "cleaned_dataset", "metadata.csv")
        self.data_dir = os.path.join(self.base_dir, "cleaned_dataset", "data")
        self.output_dir = os.path.join(self.base_dir, "results")
        self.hidden_layers = [64, 32, 16] if self.hidden_layers is None else self.hidden_layers
        os.makedirs(self.output_dir, exist_ok=True)


config = Config()

# ===================================================================================
# PART 1: EXPLORATORY DATA ANALYSIS (EDA)
# ===================================================================================

class BatteryEDA:
    """Comprehensive Exploratory Data Analysis for Battery Dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        self.metadata = None
        self.discharge_df = None
        self.discharge_df_raw = None  # Keep raw data for comparison
        self.battery_stats = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess metadata."""
        print("=" * 60)
        print("PART 1: EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        self.metadata = pd.read_csv(self.config.metadata_path)
        
        # Standardize column names
        self.metadata.columns = [c.strip().lower() for c in self.metadata.columns]
        
        print(f"\n Dataset Overview:")
        print(f"   Total records: {len(self.metadata):,}")
        print(f"   Columns: {list(self.metadata.columns)}")
        
        # Filter discharge cycles
        self.discharge_df = self.metadata[self.metadata['type'] == 'discharge'].copy()
        self.discharge_df['capacity'] = pd.to_numeric(self.discharge_df['capacity'], errors='coerce')
        self.discharge_df = self.discharge_df.dropna(subset=['capacity'])
        
        # Keep raw copy for before/after comparison
        self.discharge_df_raw = self.discharge_df.copy()
        
        # Apply preprocessing to remove anomalies
        print("   Preprocessing capacity data to remove outliers...")
        self._preprocess_all_batteries()
    
    def _preprocess_capacity_eda(self, capacities: np.ndarray, 
                                 outlier_threshold: float = 0.5) -> np.ndarray:
        """
        Preprocess capacity data to remove outliers and anomalies (for EDA).
        Same logic as in BatteryDataProcessor.
        """
        cleaned = capacities.copy()
        
        for i in range(1, len(cleaned)):
            prev_val = cleaned[i-1]
            curr_val = cleaned[i]
            
            if prev_val > 0:
                drop_ratio = (prev_val - curr_val) / prev_val
                
                # Remove sudden drops (>outlier_threshold% decrease)
                if drop_ratio > outlier_threshold:
                    cleaned[i] = prev_val
                
                # Remove values near zero (likely measurement errors)
                if curr_val < 0.1:
                    cleaned[i] = prev_val
        
        # Apply median filter
        if len(cleaned) >= 3:
            cleaned = medfilt(cleaned, kernel_size=3)
        
        return cleaned
    
    def _preprocess_all_batteries(self):
        """Preprocess capacity data for all batteries."""
        for bat_id in self.discharge_df['battery_id'].unique():
            bat_mask = self.discharge_df['battery_id'] == bat_id
            bat_df = self.discharge_df[bat_mask].copy().sort_values('test_id')
            
            if len(bat_df) == 0:
                continue
            
            raw_capacities = bat_df['capacity'].values
            cleaned_capacities = self._preprocess_capacity_eda(raw_capacities, outlier_threshold=0.5)
            
            # Update in dataframe
            self.discharge_df.loc[bat_mask, 'capacity'] = cleaned_capacities
        
        print(f"   Discharge cycles: {len(self.discharge_df):,}")
        print(f"   Unique batteries: {self.discharge_df['battery_id'].nunique()}")
        
        return self.metadata
    
    def compute_battery_statistics(self) -> pd.DataFrame:
        """Compute comprehensive statistics for each battery."""
        stats_list = []
        
        for bat_id in self.discharge_df['battery_id'].unique():
            bat_df = self.discharge_df[self.discharge_df['battery_id'] == bat_id].copy()
            bat_df = bat_df.sort_values('test_id')
            
            if len(bat_df) < 5:
                continue
            
            capacities = bat_df['capacity'].values
            init_cap = capacities[0]
            final_cap = capacities[-1]
            
            # Calculate SOH
            soh = capacities / init_cap
            
            # Find EOL cycle (SOH < threshold)
            eol_indices = np.where(soh < self.config.soh_threshold)[0]
            eol_cycle = eol_indices[0] if len(eol_indices) > 0 else len(bat_df)
            reached_eol = len(eol_indices) > 0
            
            stats_list.append({
                'battery_id': bat_id,
                'total_cycles': len(bat_df),
                'eol_cycle': eol_cycle,
                'reached_eol': reached_eol,
                'init_capacity': init_cap,
                'final_capacity': final_cap,
                'degradation_rate': (init_cap - final_cap) / init_cap,
                'min_capacity': capacities.min(),
                'max_capacity': capacities.max(),
                'capacity_std': capacities.std(),
                'mean_capacity': capacities.mean(),
            })
        
        self.battery_stats = pd.DataFrame(stats_list)
        return self.battery_stats
    
    def plot_capacity_degradation(self, save: bool = True, 
                                  train_batteries: set = None,
                                  test_batteries: set = None) -> plt.Figure:
        """Plot capacity degradation curves for all batteries, highlighting train/test split."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Get unique batteries and split into groups
        batteries = self.discharge_df['battery_id'].unique()
        
        # Filter valid batteries (enough cycles and reasonable capacity)
        valid_batteries = []
        battery_data = {}
        
        for bat_id in batteries:
            bat_df = self.discharge_df[self.discharge_df['battery_id'] == bat_id].copy()
            bat_df = bat_df.sort_values('test_id')
            
            if len(bat_df) >= self.config.min_cycles:
                init_cap = bat_df['capacity'].iloc[0]
                if init_cap >= self.config.min_init_capacity:
                    valid_batteries.append(bat_id)
                    battery_data[bat_id] = bat_df['capacity'].values
        
        print(f"\n Valid batteries for analysis: {len(valid_batteries)}")
        
        # Determine battery split if provided
        if train_batteries is None:
            train_batteries = set()
        if test_batteries is None:
            test_batteries = set()
        
        # Split into 4 groups
        groups = np.array_split(valid_batteries, 4)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for idx, (ax, group) in enumerate(zip(axes.flatten(), groups)):
            for i, bat_id in enumerate(group):
                caps = battery_data[bat_id]
                cycles = np.arange(len(caps))
                
                # Color code by train/test split
                if bat_id in train_batteries:
                    color = 'blue'
                    linestyle = '-'
                    label_suffix = ' (Train)'
                elif bat_id in test_batteries:
                    color = 'red'
                    linestyle = '--'
                    label_suffix = ' (Test)'
                else:
                    color = colors[i % 10]
                    linestyle = '-'
                    label_suffix = ''
                
                ax.plot(cycles, caps, color=color, linestyle=linestyle, 
                       label=f'{bat_id}{label_suffix}', linewidth=1.5)
            
            # Calculate EOL threshold based on average initial capacity
            if len(group) > 0:
                init_caps = [self.discharge_df[self.discharge_df['battery_id'] == bid]['capacity'].iloc[0] 
                            for bid in group]
                avg_init_cap = np.mean(init_caps)
                eol_threshold = avg_init_cap * self.config.soh_threshold
                ax.axhline(y=eol_threshold, color='red', linestyle=':', 
                          alpha=0.7, linewidth=2, label=f'EOL Threshold ({self.config.soh_threshold*100:.0f}% SOH)')
            
            ax.set_xlabel('Cycle Number', fontsize=11)
            ax.set_ylabel('Capacity (Ah)', fontsize=11)
            ax.set_title(f'Battery Group {idx + 1}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Battery Capacity Degradation Curves (Train vs Test Split) - After Preprocessing', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.config.output_dir, 'capacity_degradation_after_preprocessing.png'))
            print(f"   Saved: capacity_degradation_after_preprocessing.png")
        
        return fig
    
    def plot_before_after_preprocessing(self, save: bool = True) -> None:
        """Plot before/after preprocessing comparison for all batteries."""
        batteries = self.discharge_df['battery_id'].unique()
        
        # Filter valid batteries
        valid_batteries = []
        for bat_id in batteries:
            bat_df = self.discharge_df[self.discharge_df['battery_id'] == bat_id].copy()
            if len(bat_df) >= self.config.min_cycles:
                init_cap = bat_df['capacity'].iloc[0]
                if init_cap >= self.config.min_init_capacity:
                    valid_batteries.append(bat_id)
        
        print(f"\n Generating before/after preprocessing plots for {len(valid_batteries)} batteries...")
        
        # Create directory
        before_after_dir = os.path.join(self.config.output_dir, 'before_after_preprocessing')
        os.makedirs(before_after_dir, exist_ok=True)
        
        # Plot each battery
        for bat_id in sorted(valid_batteries):
            # Get raw and processed data
            bat_df_raw = self.discharge_df_raw[self.discharge_df_raw['battery_id'] == bat_id].copy()
            bat_df_raw = bat_df_raw.sort_values('test_id')
            
            bat_df_clean = self.discharge_df[self.discharge_df['battery_id'] == bat_id].copy()
            bat_df_clean = bat_df_clean.sort_values('test_id')
            
            if len(bat_df_raw) == 0 or len(bat_df_clean) == 0:
                continue
            
            raw_capacities = bat_df_raw['capacity'].values
            clean_capacities = bat_df_clean['capacity'].values
            cycles = np.arange(len(raw_capacities))
            init_cap = clean_capacities[0]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot 1: Before preprocessing (raw data)
            ax1.plot(cycles, raw_capacities, 'r-', linewidth=2, marker='o', markersize=2, 
                    alpha=0.7, label='Raw Capacity')
            ax1.set_ylabel('Capacity (Ah)', fontsize=12, fontweight='bold')
            ax1.set_title(f'Battery {bat_id} - BEFORE Preprocessing (Raw Data)', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=11)
            
            # Mark anomalies
            for i in range(1, len(raw_capacities)):
                if raw_capacities[i] < 0.1 or (raw_capacities[i-1] > 0 and 
                    (raw_capacities[i-1] - raw_capacities[i]) / raw_capacities[i-1] > 0.5):
                    ax1.plot(i, raw_capacities[i], 'rx', markersize=12, 
                            markeredgewidth=2, label='Anomaly' if i == 1 else '')
            
            # Plot 2: After preprocessing (cleaned data)
            ax2.plot(cycles, clean_capacities, 'b-', linewidth=2, marker='o', markersize=2, 
                    alpha=0.7, label='Preprocessed Capacity')
            ax2.axhline(y=init_cap * self.config.soh_threshold, color='red', 
                       linestyle='--', alpha=0.6, linewidth=1.5, 
                       label=f'EOL Threshold ({self.config.soh_threshold*100:.0f}% SOH)')
            ax2.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Capacity (Ah)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Battery {bat_id} - AFTER Preprocessing (Cleaned Data)', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=11)
            
            plt.suptitle(f'Battery {bat_id} - Preprocessing Comparison', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            # Save
            filename = f'battery_{bat_id}_before_after.png'
            filepath = os.path.join(before_after_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"    Saved {len(valid_batteries)} before/after plots to: {before_after_dir}/")
        
        # Also create overview grid
        self._plot_before_after_grid(valid_batteries)
    
    def _plot_before_after_grid(self, valid_batteries):
        """Create overview grid showing before/after for all batteries."""
        n_batteries = len(valid_batteries)
        n_cols = 5
        n_rows = int(np.ceil(n_batteries / n_cols))
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(20, 6 * n_rows))
        if n_batteries == 1:
            axes = axes.reshape(-1, 1)
        else:
            # Ensure axes is 2D
            if axes.ndim == 1:
                axes = axes.reshape(-1, 1)
        
        for idx, bat_id in enumerate(sorted(valid_batteries)):
            # Calculate row and column indices correctly
            row = idx // n_cols  # Which row group (0, 1, 2, ...)
            col = idx % n_cols   # Which column (0-4)
            
            # Before plot (even rows: 0, 2, 4, ...)
            ax_before = axes[row * 2, col]
            # After plot (odd rows: 1, 3, 5, ...)
            ax_after = axes[row * 2 + 1, col]
            
            bat_df_raw = self.discharge_df_raw[self.discharge_df_raw['battery_id'] == bat_id].copy()
            bat_df_raw = bat_df_raw.sort_values('test_id')
            bat_df_clean = self.discharge_df[self.discharge_df['battery_id'] == bat_id].copy()
            bat_df_clean = bat_df_clean.sort_values('test_id')
            
            if len(bat_df_raw) > 0 and len(bat_df_clean) > 0:
                cycles = np.arange(len(bat_df_raw))
                
                # Before
                ax_before.plot(cycles, bat_df_raw['capacity'].values, 'r-', 
                             linewidth=1.5, alpha=0.7)
                ax_before.set_title(f'{bat_id} - Before', fontsize=10, fontweight='bold')
                ax_before.grid(True, alpha=0.2)
                
                # After
                ax_after.plot(cycles, bat_df_clean['capacity'].values, 'b-', 
                            linewidth=1.5, alpha=0.7)
                ax_after.set_title(f'{bat_id} - After', fontsize=10, fontweight='bold')
                ax_after.set_xlabel('Cycle', fontsize=9)
                ax_after.grid(True, alpha=0.2)
        
        # Hide empty subplots
        for idx in range(n_batteries, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if row * 2 < axes.shape[0] and col < axes.shape[1]:
                axes[row * 2, col].axis('off')
            if row * 2 + 1 < axes.shape[0] and col < axes.shape[1]:
                axes[row * 2 + 1, col].axis('off')
        
        plt.suptitle('Before/After Preprocessing Overview', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = os.path.join(self.config.output_dir, 'before_after_overview.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved overview grid: before_after_overview.png")
    
    def plot_pure_degradation_curves(self, save: bool = True) -> None:
        """Plot pure capacity degradation curves for ALL battery IDs (no train/test labels).
        
        This generates publication-ready individual plots without any split information.
        Perfect for scientific papers where clean, unlabeled curves are needed.
        """
        batteries = self.discharge_df['battery_id'].unique()
        
        # Filter valid batteries
        valid_batteries = []
        battery_data = {}
        
        for bat_id in batteries:
            bat_df = self.discharge_df[self.discharge_df['battery_id'] == bat_id].copy()
            bat_df = bat_df.sort_values('test_id')
            
            if len(bat_df) >= self.config.min_cycles:
                init_cap = bat_df['capacity'].iloc[0]
                if init_cap >= self.config.min_init_capacity:
                    valid_batteries.append(bat_id)
                    capacities = bat_df['capacity'].values
                    soh = capacities / init_cap
                    eol_idx = np.where(soh < self.config.soh_threshold)[0]
                    eol_cycle = eol_idx[0] if len(eol_idx) > 0 else len(bat_df)
                    
                    battery_data[bat_id] = {
                        'capacities': capacities,
                        'init_cap': init_cap,
                        'eol_cycle': eol_cycle,
                        'cycles': np.arange(len(capacities))
                    }
        
        print(f"\n Generating pure degradation curves for {len(valid_batteries)} batteries...")
        
        # Create directory for pure plots
        pure_plots_dir = os.path.join(self.config.output_dir, 'pure_degradation_curves')
        os.makedirs(pure_plots_dir, exist_ok=True)
        
        # Plot each battery individually - PURE version (no train/test labels)
        for bat_id in sorted(valid_batteries):
            data = battery_data[bat_id]
            capacities = data['capacities']
            cycles = data['cycles']
            init_cap = data['init_cap']
            eol_cycle = data['eol_cycle']
            eol_threshold = init_cap * self.config.soh_threshold
            
            # Create individual figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Pure degradation curve - no train/test distinction
            # Use a neutral color suitable for publications
            color = '#1f77b4'  # Publication blue
            ax.plot(cycles, capacities, color=color, linestyle='-', 
                   linewidth=2.5, label=f'Battery {bat_id}', marker='o', markersize=2, 
                   alpha=0.8, markeredgewidth=0.5)
            
            # Mark EOL threshold (optional - comment out if you want truly minimal)
            ax.axhline(y=eol_threshold, color='red', linestyle='--', 
                      alpha=0.5, linewidth=1.5, 
                      label=f'EOL Threshold ({self.config.soh_threshold*100:.0f}% SOH)')
            
            # Mark EOL cycle if reached (optional)
            if eol_cycle < len(capacities):
                ax.axvline(x=eol_cycle, color='orange', linestyle=':', 
                          alpha=0.5, linewidth=1.5)
                ax.plot(eol_cycle, capacities[eol_cycle], 'ro', markersize=8, 
                       markerfacecolor='red', markeredgecolor='darkred', 
                       markeredgewidth=1.5, zorder=5, label='EOL')
            
            # Clean, publication-ready styling
            ax.set_xlabel('Cycle Number', fontsize=13, fontweight='normal')
            ax.set_ylabel('Capacity (Ah)', fontsize=13, fontweight='normal')
            ax.set_title(f'Battery {bat_id}', fontsize=14, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax.legend(loc='best', fontsize=11, framealpha=0.9, frameon=True, 
                     fancybox=True, shadow=False)
            
            # Set reasonable y-limits
            cap_range = capacities.max() - capacities.min()
            if cap_range > 0:
                y_margin = max(0.1 * cap_range, 0.1)
                ax.set_ylim(max(0, capacities.min() - y_margin),
                           capacities.max() + y_margin)
            
            # Optional: Add minimal statistics (can be removed for pure curves)
            # Uncomment below if you want statistics box
            """
            final_cap = capacities[-1]
            degradation = (init_cap - final_cap) / init_cap * 100
            reached_eol = "Yes" if eol_cycle < len(capacities) else "No"
            
            stats_text = f'Initial: {init_cap:.3f} Ah\n'
            stats_text += f'Final: {final_cap:.3f} Ah\n'
            stats_text += f'Degradation: {degradation:.1f}%\n'
            stats_text += f'Total Cycles: {len(capacities)}\n'
            stats_text += f'EOL Reached: {reached_eol}'
            if eol_cycle < len(capacities):
                stats_text += f'\nEOL Cycle: {eol_cycle}'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props, family='monospace')
            """
            
            plt.tight_layout()
            
            # Save individual plot
            filename = f'battery_{bat_id}_pure.png'
            filepath = os.path.join(pure_plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Close to free memory
        
        print(f"    Saved {len(valid_batteries)} pure degradation curves to: {pure_plots_dir}/")
        
        # Also create a pure overview plot with all batteries (no train/test labels)
        self._plot_pure_overview_grid(valid_batteries, battery_data)
    
    def _plot_pure_overview_grid(self, valid_batteries, battery_data):
        """Create a clean grid overview of all batteries without train/test labels."""
        n_batteries = len(valid_batteries)
        n_cols = 5
        n_rows = int(np.ceil(n_batteries / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_batteries == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Use a colormap for variety but keep it publication-appropriate
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_batteries, 10)))
        
        for idx, bat_id in enumerate(sorted(valid_batteries)):
            ax = axes[idx]
            
            data = battery_data[bat_id]
            capacities = data['capacities']
            cycles = data['cycles']
            init_cap = data['init_cap']
            eol_cycle = data['eol_cycle']
            eol_threshold = init_cap * self.config.soh_threshold
            
            # Pure curve - no train/test distinction
            color = colors[idx % len(colors)]
            ax.plot(cycles, capacities, color=color, linestyle='-', linewidth=2, 
                   label=f'{bat_id}', marker='o', markersize=2, alpha=0.7)
            ax.axhline(y=eol_threshold, color='red', linestyle='--', 
                      alpha=0.4, linewidth=1.0)
            
            if eol_cycle < len(capacities):
                ax.axvline(x=eol_cycle, color='orange', linestyle=':', 
                          alpha=0.4, linewidth=1.0)
                ax.plot(eol_cycle, capacities[eol_cycle], 'ro', markersize=5)
            
            ax.set_xlabel('Cycle', fontsize=9)
            ax.set_ylabel('Capacity (Ah)', fontsize=9)
            ax.set_title(f'{bat_id}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, loc='best', framealpha=0.9)
            
            cap_range = capacities.max() - capacities.min()
            if cap_range > 0:
                y_margin = max(0.1 * cap_range, 0.1)
                ax.set_ylim(max(0, capacities.min() - y_margin),
                           capacities.max() + y_margin)
        
        # Hide empty subplots
        for idx in range(n_batteries, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Battery Capacity Degradation Curves', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = os.path.join(self.config.output_dir, 'pure_all_batteries_overview.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved pure overview grid plot: pure_all_batteries_overview.png")
    
    def plot_individual_battery_curves(self, train_batteries: set = None,
                                       test_batteries: set = None,
                                       save: bool = True) -> None:
        """Plot individual capacity degradation curve for EACH battery ID as separate files."""
        batteries = self.discharge_df['battery_id'].unique()
        
        # Filter valid batteries
        valid_batteries = []
        battery_data = {}
        
        for bat_id in batteries:
            bat_df = self.discharge_df[self.discharge_df['battery_id'] == bat_id].copy()
            bat_df = bat_df.sort_values('test_id')
            
            if len(bat_df) >= self.config.min_cycles:
                init_cap = bat_df['capacity'].iloc[0]
                if init_cap >= self.config.min_init_capacity:
                    valid_batteries.append(bat_id)
                    capacities = bat_df['capacity'].values
                    soh = capacities / init_cap
                    eol_idx = np.where(soh < self.config.soh_threshold)[0]
                    eol_cycle = eol_idx[0] if len(eol_idx) > 0 else len(bat_df)
                    
                    battery_data[bat_id] = {
                        'capacities': capacities,
                        'init_cap': init_cap,
                        'eol_cycle': eol_cycle,
                        'cycles': np.arange(len(capacities))
                    }
        
        print(f"\n Generating individual plots for {len(valid_batteries)} batteries...")
        
        # Determine split
        if train_batteries is None:
            train_batteries = set()
        if test_batteries is None:
            test_batteries = set()
        
        # Create individual directory for battery plots
        battery_plots_dir = os.path.join(self.config.output_dir, 'individual_batteries')
        os.makedirs(battery_plots_dir, exist_ok=True)
        
        # Plot each battery individually
        for bat_id in sorted(valid_batteries):
            data = battery_data[bat_id]
            capacities = data['capacities']
            cycles = data['cycles']
            init_cap = data['init_cap']
            eol_cycle = data['eol_cycle']
            eol_threshold = init_cap * self.config.soh_threshold
            
            # Create individual figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color and style by train/test
            if bat_id in train_batteries:
                color = '#2E86AB'  # Blue for train
                linestyle = '-'
                split_label = 'Train'
            elif bat_id in test_batteries:
                color = '#A23B72'  # Red/purple for test
                linestyle = '--'
                split_label = 'Test'
            else:
                color = '#6C757D'  # Gray if not in split
                linestyle = '-'
                split_label = 'Unknown'
            
            # Plot capacity degradation
            ax.plot(cycles, capacities, color=color, linestyle=linestyle, 
                   linewidth=2.5, label=f'{bat_id} ({split_label})', marker='o', markersize=3)
            
            # Mark EOL threshold
            ax.axhline(y=eol_threshold, color='red', linestyle=':', 
                      alpha=0.7, linewidth=2, 
                      label=f'EOL Threshold ({self.config.soh_threshold*100:.0f}% SOH = {eol_threshold:.3f} Ah)')
            
            # Mark EOL cycle if reached
            if eol_cycle < len(capacities):
                ax.axvline(x=eol_cycle, color='orange', linestyle=':', 
                          alpha=0.7, linewidth=2, label=f'EOL at Cycle {eol_cycle}')
                ax.plot(eol_cycle, capacities[eol_cycle], 'ro', markersize=10, 
                       markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2,
                       label='EOL Reached', zorder=5)
            
            # Add initial capacity marker
            ax.plot(0, init_cap, 'go', markersize=8, 
                   markerfacecolor='green', markeredgecolor='darkgreen', markeredgewidth=2,
                   label=f'Initial Capacity: {init_cap:.3f} Ah', zorder=5)
            
            # Styling
            ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
            ax.set_ylabel('Capacity (Ah)', fontsize=12, fontweight='bold')
            ax.set_title(f'Battery {bat_id} - Capacity Degradation Curve', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=10, framealpha=0.95, frameon=True)
            
            # Set reasonable y-limits
            cap_range = capacities.max() - capacities.min()
            if cap_range > 0:
                y_margin = max(0.1 * cap_range, 0.1)
                ax.set_ylim(max(0, capacities.min() - y_margin),
                           capacities.max() + y_margin)
            
            # Add statistics text box
            final_cap = capacities[-1]
            degradation = (init_cap - final_cap) / init_cap * 100
            reached_eol = "Yes" if eol_cycle < len(capacities) else "No"
            
            stats_text = f'Initial: {init_cap:.3f} Ah\n'
            stats_text += f'Final: {final_cap:.3f} Ah\n'
            stats_text += f'Degradation: {degradation:.1f}%\n'
            stats_text += f'Total Cycles: {len(capacities)}\n'
            stats_text += f'EOL Reached: {reached_eol}'
            if eol_cycle < len(capacities):
                stats_text += f'\nEOL Cycle: {eol_cycle}'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props, family='monospace')
            
            plt.tight_layout()
            
            # Save individual plot
            filename = f'battery_{bat_id}_degradation.png'
            filepath = os.path.join(battery_plots_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
        
        print(f"    Saved {len(valid_batteries)} individual battery plots to: {battery_plots_dir}/")
        
        # Also create a summary plot with all batteries (grid view)
        self._plot_all_batteries_grid(valid_batteries, battery_data, train_batteries, test_batteries)
    
    def _plot_all_batteries_grid(self, valid_batteries, battery_data, 
                                 train_batteries, test_batteries):
        """Create a grid view with all batteries for overview."""
        n_batteries = len(valid_batteries)
        n_cols = 5
        n_rows = int(np.ceil(n_batteries / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_batteries == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, bat_id in enumerate(sorted(valid_batteries)):
            ax = axes[idx]
            
            data = battery_data[bat_id]
            capacities = data['capacities']
            cycles = data['cycles']
            init_cap = data['init_cap']
            eol_cycle = data['eol_cycle']
            eol_threshold = init_cap * self.config.soh_threshold
            
            # Color by train/test
            if bat_id in train_batteries:
                color = '#2E86AB'
                linestyle = '-'
                label = f'{bat_id} (Train)'
            elif bat_id in test_batteries:
                color = '#A23B72'
                linestyle = '--'
                label = f'{bat_id} (Test)'
            else:
                color = '#6C757D'
                linestyle = '-'
                label = bat_id
            
            ax.plot(cycles, capacities, color=color, linestyle=linestyle, linewidth=2, label=label)
            ax.axhline(y=eol_threshold, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
            
            if eol_cycle < len(capacities):
                ax.axvline(x=eol_cycle, color='orange', linestyle=':', alpha=0.6, linewidth=1.5)
                ax.plot(eol_cycle, capacities[eol_cycle], 'ro', markersize=6)
            
            ax.set_xlabel('Cycle', fontsize=9)
            ax.set_ylabel('Capacity (Ah)', fontsize=9)
            ax.set_title(f'{bat_id}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best', framealpha=0.9)
            
            cap_range = capacities.max() - capacities.min()
            if cap_range > 0:
                y_margin = max(0.1 * cap_range, 0.1)
                ax.set_ylim(max(0, capacities.min() - y_margin),
                           capacities.max() + y_margin)
        
        # Hide empty subplots
        for idx in range(n_batteries, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('All Battery Degradation Curves - Overview (Train vs Test)', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = os.path.join(self.config.output_dir, 'all_batteries_overview.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved overview grid plot: all_batteries_overview.png")
    
    def plot_individual_batteries(self, train_batteries: set = None,
                                  test_batteries: set = None,
                                  save: bool = True) -> plt.Figure:
        """Plot individual degradation curve for each battery with train/test labels."""
        batteries = self.discharge_df['battery_id'].unique()
        
        # Filter valid batteries
        valid_batteries = []
        battery_data = {}
        battery_eol = {}
        
        for bat_id in batteries:
            bat_df = self.discharge_df[self.discharge_df['battery_id'] == bat_id].copy()
            bat_df = bat_df.sort_values('test_id')
            
            if len(bat_df) >= self.config.min_cycles:
                init_cap = bat_df['capacity'].iloc[0]
                if init_cap >= self.config.min_init_capacity:
                    valid_batteries.append(bat_id)
                    capacities = bat_df['capacity'].values
                    soh = capacities / init_cap
                    eol_idx = np.where(soh < self.config.soh_threshold)[0]
                    eol_cycle = eol_idx[0] if len(eol_idx) > 0 else len(bat_df)
                    
                    battery_data[bat_id] = {
                        'capacities': capacities,
                        'init_cap': init_cap,
                        'eol_cycle': eol_cycle
                    }
                    battery_eol[bat_id] = init_cap * self.config.soh_threshold
        
        print(f"\n Plotting {len(valid_batteries)} individual battery curves...")
        
        # Determine split
        if train_batteries is None:
            train_batteries = set()
        if test_batteries is None:
            test_batteries = set()
        
        # Calculate grid size
        n_batteries = len(valid_batteries)
        n_cols = 5
        n_rows = int(np.ceil(n_batteries / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten() if n_batteries > 1 else [axes]
        
        for idx, bat_id in enumerate(sorted(valid_batteries)):
            ax = axes[idx] if n_batteries > 1 else axes[0]
            
            data = battery_data[bat_id]
            capacities = data['capacities']
            cycles = np.arange(len(capacities))
            init_cap = data['init_cap']
            eol_cycle = data['eol_cycle']
            eol_threshold = battery_eol[bat_id]
            
            # Color and style by train/test
            if bat_id in train_batteries:
                color = '#2E86AB'  # Blue for train
                linestyle = '-'
                label = f'{bat_id} (Train)'
            elif bat_id in test_batteries:
                color = '#A23B72'  # Red/purple for test
                linestyle = '--'
                label = f'{bat_id} (Test)'
            else:
                color = '#6C757D'  # Gray if not in split
                linestyle = '-'
                label = bat_id
            
            ax.plot(cycles, capacities, color=color, linestyle=linestyle, 
                   linewidth=2, label=label)
            
            # Mark EOL threshold
            ax.axhline(y=eol_threshold, color='red', linestyle=':', 
                      alpha=0.6, linewidth=1.5)
            
            # Mark EOL cycle if reached
            if eol_cycle < len(capacities):
                ax.axvline(x=eol_cycle, color='orange', linestyle=':', 
                          alpha=0.6, linewidth=1.5)
                ax.plot(eol_cycle, capacities[eol_cycle], 'ro', markersize=8, 
                       label='EOL Reached')
            
            ax.set_xlabel('Cycle Number', fontsize=9)
            ax.set_ylabel('Capacity (Ah)', fontsize=9)
            ax.set_title(f'{bat_id}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best', framealpha=0.9)
            
            # Set reasonable y-limits
            cap_range = capacities.max() - capacities.min()
            ax.set_ylim(max(0, capacities.min() - 0.1 * cap_range),
                       capacities.max() + 0.1 * cap_range)
        
        # Hide empty subplots
        for idx in range(n_batteries, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Individual Battery Degradation Curves (Train vs Test Split)', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.config.output_dir, 'individual_battery_degradation.png'))
            print(f"   Saved: individual_battery_degradation.png")
        
        return fig
    
    def plot_soh_distribution(self, save: bool = True) -> plt.Figure:
        """Plot SOH distribution and EOL analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. EOL cycle distribution
        ax1 = axes[0, 0]
        eol_data = self.battery_stats[self.battery_stats['reached_eol'] == True]['eol_cycle']
        if len(eol_data) > 0:
            ax1.hist(eol_data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax1.axvline(eol_data.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {eol_data.mean():.1f}')
        ax1.set_xlabel('EOL Cycle')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of End-of-Life Cycles')
        ax1.legend()
        
        # 2. Initial capacity distribution
        ax2 = axes[0, 1]
        init_caps = self.battery_stats['init_capacity']
        ax2.hist(init_caps, bins=15, color='forestgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(init_caps.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {init_caps.mean():.2f} Ah')
        ax2.set_xlabel('Initial Capacity (Ah)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Initial Capacity')
        ax2.legend()
        
        # 3. Degradation rate distribution
        ax3 = axes[1, 0]
        deg_rates = self.battery_stats['degradation_rate']
        valid_deg = deg_rates[(deg_rates > 0) & (deg_rates < 1)]
        ax3.hist(valid_deg, bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Degradation Rate')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Degradation Rates')
        
        # 4. Censoring status
        ax4 = axes[1, 1]
        censoring_counts = self.battery_stats['reached_eol'].value_counts()
        labels = ['Uncensored (EOL reached)', 'Censored (EOL not reached)']
        colors_pie = ['#ff6b6b', '#4ecdc4']
        ax4.pie(censoring_counts.values, labels=labels, colors=colors_pie, 
                autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
        ax4.set_title('Event Censoring Status')
        
        plt.suptitle('Battery Dataset Statistics', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.config.output_dir, 'dataset_statistics.png'))
            print(f"   Saved: dataset_statistics.png")
        
        return fig
    
    def run_full_eda(self):
        """Run complete EDA pipeline."""
        self.load_data()
        self.compute_battery_statistics()
        
        print(f"\n Battery Statistics Summary:")
        print(self.battery_stats.describe().to_string())
        
        # Plot before/after preprocessing comparison
        self.plot_before_after_preprocessing()
        
        # Plot after preprocessing
        self.plot_capacity_degradation()
        self.plot_soh_distribution()
        
        # Generate pure degradation curves (for publication)
        self.plot_pure_degradation_curves()
        
        # Save statistics
        stats_path = os.path.join(self.config.output_dir, 'battery_statistics.csv')
        self.battery_stats.to_csv(stats_path, index=False)
        print(f"   Saved: battery_statistics.csv")
        
        return self.battery_stats


# ===================================================================================
# PART 2: DATA PREPROCESSING & FEATURE ENGINEERING
# ===================================================================================

class SurvivalDataset(Dataset):
    """PyTorch Dataset for Survival Analysis."""
    
    def __init__(self, features: np.ndarray, times: np.ndarray, events: np.ndarray,
                 battery_ids: np.ndarray = None):
        self.features = torch.FloatTensor(features)
        self.times = torch.FloatTensor(times)
        self.events = torch.FloatTensor(events)
        self.battery_ids = battery_ids
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.times[idx], self.events[idx]


class BatteryDataProcessor:
    """Complete data processing pipeline for battery survival analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def _get_feature_names(self, feature_set_name: str) -> List[str]:
        """
        Get feature names based on feature set configuration.
        
        Args:
            feature_set_name: One of 'basic', 'dynamic', 'full'
                - 'basic': capacity, soh (minimal features)
                - 'dynamic': basic + derivatives (degradation rate information)
                - 'full': dynamic + rolling statistics (variability information)
        
        Returns:
            List of feature names
        """
        all_features = {
            'capacity': 'capacity',
            'capacity_smooth': 'capacity_smooth',
            'soh': 'soh',
            'soh_derivative': 'soh_derivative',
            'capacity_derivative': 'capacity_derivative',
            'capacity_rolling_std': 'capacity_rolling_std'
        }
        
        if feature_set_name == 'basic':
            # Case 1: Basic features only (capacity, soh)
            return ['capacity', 'soh']
        elif feature_set_name == 'dynamic':
            # Case 2: Basic + dynamic information (derivatives)
            return ['capacity', 'soh', 'soh_derivative', 'capacity_derivative']
        elif feature_set_name == 'full':
            # Case 3: Full feature set (includes variability)
            return list(all_features.values())
        else:
            raise ValueError(f"Unknown feature_set_name: {feature_set_name}. "
                           f"Must be one of: 'basic', 'dynamic', 'full'")
        
    def load_and_process(self) -> Tuple[pd.DataFrame, List[str]]:
        """Load metadata and extract features."""
        print("\n" + "=" * 60)
        print("PART 2: DATA PREPROCESSING")
        print("=" * 60)
        
        metadata = pd.read_csv(self.config.metadata_path)
        metadata.columns = [c.strip().lower() for c in metadata.columns]
        
        # Filter discharge cycles
        discharge_df = metadata[metadata['type'] == 'discharge'].copy()
        discharge_df['capacity'] = pd.to_numeric(discharge_df['capacity'], errors='coerce')
        discharge_df = discharge_df.dropna(subset=['capacity'])
        
        # Process each battery
        all_samples = []
        
        for bat_id in discharge_df['battery_id'].unique():
            bat_df = discharge_df[discharge_df['battery_id'] == bat_id].copy()
            bat_df = bat_df.sort_values('test_id').reset_index(drop=True)
            
            # Apply filters
            if len(bat_df) < self.config.min_cycles:
                continue
            
            init_cap = bat_df['capacity'].iloc[0]
            if init_cap < self.config.min_init_capacity:
                continue
            
            samples = self._process_battery(bat_df, bat_id, init_cap)
            all_samples.extend(samples)
        
        df = pd.DataFrame(all_samples)
        
        # Define feature columns based on feature_set_name
        # NOTE: Removed cycle_normalized and degradation_rate to prevent data leakage
        # cycle_normalized = k/max_cycle is directly correlated with RUL = eol_cycle - k
        # degradation_rate = (init_cap - capacity)/init_cap contains similar info to SOH
        self.feature_names = self._get_feature_names(self.config.feature_set_name)
        
        # Fill NaN values
        for col in self.feature_names:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        print(f"\n Processed Dataset:")
        print(f"   Total samples: {len(df):,}")
        print(f"   Unique batteries: {df['battery_id'].nunique()}")
        print(f"   Features: {self.feature_names}")
        print(f"   Event rate: {df['event'].mean():.2%}")
        
        return df, self.feature_names
    
    def _preprocess_capacity(self, capacities: np.ndarray, 
                             outlier_threshold: float = 0.5) -> np.ndarray:
        """
        Preprocess capacity data to remove outliers and anomalies.
        
        Removes:
        1. Sudden drops (>outlier_threshold% of previous value)
        2. Values near zero (< 0.1 Ah) that are likely measurement errors
        
        Args:
            capacities: Raw capacity array
            outlier_threshold: Maximum allowed drop ratio (0.5 = 50% drop)
        
        Returns:
            Cleaned capacity array
        """
        cleaned = capacities.copy()
        
        for i in range(1, len(cleaned)):
            # Detect sudden drops (>outlier_threshold% decrease)
            prev_val = cleaned[i-1]
            curr_val = cleaned[i]
            
            if prev_val > 0:
                drop_ratio = (prev_val - curr_val) / prev_val
                
                # If drop is too large, interpolate or use previous value
                if drop_ratio > outlier_threshold:
                    # Use linear interpolation between previous and next valid point
                    # Or use previous value if this is clearly an outlier
                    cleaned[i] = prev_val  # Conservative: use previous value
                    # Alternative: could interpolate if we have future values
                
                # Remove values near zero (likely measurement errors)
                if curr_val < 0.1:
                    cleaned[i] = prev_val
        
        # Apply median filter to further smooth (removes remaining small spikes)
        if len(cleaned) >= 3:
            cleaned = medfilt(cleaned, kernel_size=3)
        
        return cleaned
    
    def _process_battery(self, bat_df: pd.DataFrame, bat_id: str, 
                        init_cap: float) -> List[Dict]:
        """Process a single battery and extract features."""
        samples = []
        
        # Store raw capacity for visualization
        raw_capacity = bat_df['capacity'].values.copy()
        
        # Preprocess to remove outliers and anomalies
        cleaned_capacity = self._preprocess_capacity(raw_capacity, outlier_threshold=0.5)
        bat_df['capacity'] = cleaned_capacity
        bat_df['capacity_raw'] = raw_capacity  # Keep raw for comparison
        
        # Apply smoothing to cleaned data (more aggressive smoothing after cleaning)
        bat_df['capacity_smooth'] = bat_df['capacity'].rolling(window=7, min_periods=1, center=True).mean()
        # Fill any remaining NaN with capacity itself
        bat_df['capacity_smooth'] = bat_df['capacity_smooth'].fillna(bat_df['capacity'])
        
        bat_df['soh'] = bat_df['capacity_smooth'] / init_cap
        
        # Find EOL
        eol_indices = bat_df.index[bat_df['soh'] < self.config.soh_threshold].tolist()
        
        if eol_indices:
            event = 1  # Uncensored
            eol_cycle = eol_indices[0]
        else:
            event = 0  # Censored
            eol_cycle = len(bat_df) - 1
        
        # Calculate derivatives and rolling statistics
        bat_df['soh_derivative'] = bat_df['soh'].diff().fillna(0)
        bat_df['capacity_derivative'] = bat_df['capacity'].diff().fillna(0)
        bat_df['capacity_rolling_std'] = bat_df['capacity'].rolling(window=10, min_periods=1).std()
        
        # NOTE: Removed cycle_normalized - it causes data leakage with RUL
        # NOTE: Removed degradation_rate - redundant with SOH (both use init_cap ratio)
        
        # Create samples up to EOL
        max_k = eol_cycle if event == 1 else len(bat_df)
        
        for k in range(max_k):
            row = bat_df.iloc[k]
            rul = eol_cycle - k  # Remaining Useful Life
            
            if rul <= 0:
                continue
            
            sample = {
                'battery_id': bat_id,
                'cycle_index': k,
                'capacity': row['capacity'],
                'capacity_smooth': row['capacity_smooth'],
                'soh': row['soh'],
                'soh_derivative': row['soh_derivative'],
                'capacity_derivative': row['capacity_derivative'],
                'capacity_rolling_std': row['capacity_rolling_std'],
                'time': rul,  # RUL as survival time
                'event': event,
            }
            samples.append(sample)
        
        return samples
    
    def prepare_survival_data(self, df: pd.DataFrame, 
                             test_size: float = 0.2) -> Dict:
        """Prepare train/test split at battery level."""
        battery_ids = df['battery_id'].unique()
        
        train_bats, test_bats = train_test_split(
            battery_ids, test_size=test_size, random_state=RANDOM_SEED
        )
        
        train_df = df[df['battery_id'].isin(train_bats)].copy()
        test_df = df[df['battery_id'].isin(test_bats)].copy()
        
        # Extract features and labels
        X_train = train_df[self.feature_names].values.astype(np.float32)
        T_train = train_df['time'].values.astype(np.float32)
        E_train = train_df['event'].values.astype(np.float32)
        bat_train = train_df['battery_id'].values
        
        X_test = test_df[self.feature_names].values.astype(np.float32)
        T_test = test_df['time'].values.astype(np.float32)
        E_test = test_df['event'].values.astype(np.float32)
        bat_test = test_df['battery_id'].values
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)
        
        print(f"\n Train/Test Split:")
        print(f"   Train batteries: {len(train_bats)}")
        print(f"   Test batteries: {len(test_bats)}")
        print(f"   Train samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")
        
        return {
            'X_train': X_train_scaled, 'T_train': T_train, 'E_train': E_train,
            'X_test': X_test_scaled, 'T_test': T_test, 'E_test': E_test,
            'train_bats': train_bats, 'test_bats': test_bats,
            'bat_train': bat_train, 'bat_test': bat_test,
            'train_df': train_df, 'test_df': test_df
        }


# ===================================================================================
# PART 3: DEEPSURV MODEL IMPLEMENTATION
# ===================================================================================

class DeepSurv(nn.Module):
    """
    DeepSurv: Deep Learning for Survival Analysis.
    
    Architecture: MLP that outputs log-hazard ratio.
    Loss: Negative log partial likelihood (Cox PH loss).
    """
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [64, 32],
                 dropout: float = 0.3, activation: str = 'relu'):
        super(DeepSurv, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'selu':
                layers.append(nn.SELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        # Output layer: single node for log-hazard ratio
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def cox_ph_loss(risk_scores: torch.Tensor, times: torch.Tensor, 
                events: torch.Tensor) -> torch.Tensor:
    """
    Cox Proportional Hazards Negative Log Partial Likelihood.
    
    L = -Σ_{i: E_i=1} [h_i - log(Σ_{j: T_j >= T_i} exp(h_j))]
    
    Args:
        risk_scores: Log-hazard ratios f(x), shape (N,)
        times: Survival times T, shape (N,)
        events: Event indicators E (1=event, 0=censored), shape (N,)
    
    Returns:
        Negative log partial likelihood (scalar)
    """
    # Sort by descending time for cumulative sum trick
    idx = torch.argsort(times, descending=True)
    risk_sorted = risk_scores[idx]
    events_sorted = events[idx]
    
    # Compute log cumulative sum of exp(risk) - numerically stable
    log_cumsum = torch.logcumsumexp(risk_sorted, dim=0)
    
    # Partial likelihood for uncensored observations
    event_mask = events_sorted == 1
    log_likelihood = risk_sorted[event_mask] - log_cumsum[event_mask]
    
    # Return negative mean log likelihood
    return -log_likelihood.mean() if event_mask.sum() > 0 else torch.tensor(0.0)


# ===================================================================================
# PART 4: EVALUATION METRICS
# ===================================================================================

class SurvivalMetrics:
    """Comprehensive survival analysis metrics for DeepSurv evaluation."""
    
    @staticmethod
    def concordance_index(risk_scores: np.ndarray, times: np.ndarray, 
                         events: np.ndarray) -> float:
        """
        Calculate Harrell's Concordance Index (C-index).
        
        C-index measures the model's ability to correctly rank pairs of subjects
        by their risk scores. C-index = 0.5 means random, 1.0 means perfect.
        
        Args:
            risk_scores: Predicted risk scores (higher = more risk)
            times: Survival times
            events: Event indicators (1=event occurred)
        
        Returns:
            C-index value between 0 and 1
        """
        n = len(times)
        concordant = 0
        permissible = 0
        
        for i in range(n):
            if events[i] == 1:  # Only consider uncensored subjects
                for j in range(n):
                    if times[i] < times[j]:  # i failed before j
                        permissible += 1
                        if risk_scores[i] > risk_scores[j]:
                            concordant += 1
                        elif risk_scores[i] == risk_scores[j]:
                            concordant += 0.5
        
        return concordant / permissible if permissible > 0 else 0.5
    
    @staticmethod
    def brier_score(survival_probs: np.ndarray, times: np.ndarray, 
                   events: np.ndarray, eval_time: float) -> float:
        """
        Calculate Brier Score at a specific time point.
        
        BS(t) = E[(S(t|X) - I(T > t))^2]
        
        Args:
            survival_probs: Predicted survival probabilities S(t|X)
            times: Observed survival times
            events: Event indicators
            eval_time: Time point to evaluate
        
        Returns:
            Brier score (lower is better, 0 is perfect)
        """
        n = len(times)
        
        # Binary outcome at eval_time
        outcome = (times > eval_time).astype(float)
        
        # Inverse probability censoring weights (simplified)
        # For censored observations beyond eval_time, they contribute
        weights = np.ones(n)
        for i in range(n):
            if events[i] == 0 and times[i] < eval_time:
                weights[i] = 0  # Exclude censored before eval_time
        
        # Brier score
        weighted_sq_diff = weights * (survival_probs - outcome) ** 2
        bs = np.sum(weighted_sq_diff) / np.sum(weights) if np.sum(weights) > 0 else 0
        
        return bs
    
    @staticmethod
    def integrated_brier_score(survival_func: callable, times: np.ndarray,
                               events: np.ndarray, 
                               time_grid: np.ndarray = None) -> float:
        """
        Calculate Integrated Brier Score (IBS) over time.
        
        IBS = (1/t_max) * ∫ BS(t) dt
        
        Args:
            survival_func: Function that returns S(t|X) for given time t
            times: Observed survival times
            events: Event indicators
            time_grid: Time points for integration
        
        Returns:
            IBS value (lower is better)
        """
        if time_grid is None:
            time_grid = np.linspace(times.min(), times.max(), 100)
        
        bs_values = []
        for t in time_grid:
            survival_probs = survival_func(t)
            bs = SurvivalMetrics.brier_score(survival_probs, times, events, t)
            bs_values.append(bs)
        
        # Trapezoidal integration
        ibs = np.trapz(bs_values, time_grid) / (time_grid.max() - time_grid.min())
        return ibs
    
    @staticmethod
    def time_dependent_auc(risk_scores: np.ndarray, times: np.ndarray,
                          events: np.ndarray, eval_time: float) -> float:
        """
        Calculate time-dependent AUC at a specific time point.
        
        For subjects who experienced an event before eval_time vs.
        those who survived past eval_time.
        """
        # Cases: event before eval_time
        cases = (times <= eval_time) & (events == 1)
        # Controls: survived past eval_time
        controls = times > eval_time
        
        if cases.sum() == 0 or controls.sum() == 0:
            return 0.5
        
        case_risks = risk_scores[cases]
        control_risks = risk_scores[controls]
        
        # Mann-Whitney U statistic (equivalent to AUC)
        concordant = 0
        total = 0
        
        for cr in case_risks:
            for cor in control_risks:
                total += 1
                if cr > cor:
                    concordant += 1
                elif cr == cor:
                    concordant += 0.5
        
        return concordant / total if total > 0 else 0.5


# ===================================================================================
# PART 5: TRAINING AND EVALUATION PIPELINE
# ===================================================================================

class DeepSurvTrainer:
    """Complete training pipeline for DeepSurv model."""
    
    def __init__(self, config: Config, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_history = {'train_loss': [], 'val_loss': [], 
                                'train_cindex': [], 'val_cindex': []}
        
    def train(self, train_data: Dict, val_data: Dict = None) -> Dict:
        """Train the DeepSurv model."""
        print("\n" + "=" * 60)
        print("PART 3: MODEL TRAINING")
        print("=" * 60)
        
        # Create data loaders
        train_dataset = SurvivalDataset(
            train_data['X'], train_data['T'], train_data['E']
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        
        if val_data:
            val_dataset = SurvivalDataset(
                val_data['X'], val_data['T'], val_data['E']
            )
            val_loader = DataLoader(
                val_dataset, batch_size=len(val_dataset), shuffle=False
            )
        
        # Initialize model
        input_dim = train_data['X'].shape[1]
        self.model = DeepSurv(
            input_dim=input_dim,
            hidden_layers=self.config.hidden_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        print(f"\n Model Configuration:")
        print(f"   Architecture: {self.config.hidden_layers}")
        print(f"   Dropout: {self.config.dropout}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Device: {self.device}")
        
        # Training loop
        best_val_cindex = 0
        best_model_state = None
        patience_counter = 0
        
        print(f"\n Training for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for X_batch, T_batch, E_batch in train_loader:
                X_batch = X_batch.to(self.device)
                T_batch = T_batch.to(self.device)
                E_batch = E_batch.to(self.device)
                
                optimizer.zero_grad()
                risk = self.model(X_batch)
                loss = cox_ph_loss(risk, T_batch, E_batch)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Compute training C-index
            self.model.eval()
            with torch.no_grad():
                train_X = torch.FloatTensor(train_data['X']).to(self.device)
                train_risk = self.model(train_X).cpu().numpy()
                train_cindex = SurvivalMetrics.concordance_index(
                    train_risk, train_data['T'], train_data['E']
                )
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_cindex'].append(train_cindex)
            
            # Validation phase
            if val_data:
                self.model.eval()
                with torch.no_grad():
                    val_X = torch.FloatTensor(val_data['X']).to(self.device)
                    val_T = torch.FloatTensor(val_data['T']).to(self.device)
                    val_E = torch.FloatTensor(val_data['E']).to(self.device)
                    
                    val_risk = self.model(val_X)
                    val_loss = cox_ph_loss(val_risk, val_T, val_E).item()
                    
                    val_risk_np = val_risk.cpu().numpy()
                    val_cindex = SurvivalMetrics.concordance_index(
                        val_risk_np, val_data['T'], val_data['E']
                    )
                
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_cindex'].append(val_cindex)
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_cindex > best_val_cindex:
                    best_val_cindex = val_cindex
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.patience:
                    print(f"\n   Early stopping at epoch {epoch + 1}")
                    break
            
            # Print progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                msg = f"   Epoch {epoch + 1:3d}: Train Loss={avg_train_loss:.4f}, Train C-index={train_cindex:.4f}"
                if val_data:
                    msg += f", Val Loss={val_loss:.4f}, Val C-index={val_cindex:.4f}"
                print(msg)
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n Training complete! Best Val C-index: {best_val_cindex:.4f}")
        
        return self.training_history
    
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores for new data."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            risk = self.model(X_tensor).cpu().numpy()
        return risk
    
    def compute_baseline_hazard(self, risk: np.ndarray, times: np.ndarray,
                               events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Breslow estimator for baseline cumulative hazard."""
        order = np.argsort(times)
        T_sorted = times[order]
        E_sorted = events[order]
        risk_sorted = risk[order]
        exp_risk = np.exp(risk_sorted)
        
        unique_times = np.unique(T_sorted[E_sorted == 1])
        
        # Handle case where no events occurred
        if len(unique_times) == 0:
            # Return constant hazard if no events
            unique_times = np.array([times.max()])
            H0 = np.array([1e-6])
            return unique_times, H0
        
        H0 = []
        
        for t in unique_times:
            d_k = ((T_sorted == t) & (E_sorted == 1)).sum()
            risk_set = T_sorted >= t
            denom = exp_risk[risk_set].sum()
            H0.append(d_k / max(denom, 1e-12))
        
        H0 = np.cumsum(H0)
        return unique_times, H0
    
    def predict_survival(self, X: np.ndarray, times_grid: np.ndarray,
                        baseline_times: np.ndarray, 
                        baseline_H0: np.ndarray) -> np.ndarray:
        """Predict survival probabilities S(t|X)."""
        risk = self.predict_risk(X)
        exp_risk = np.exp(risk)
        
        # Interpolate baseline hazard
        H0_interp = interp1d(baseline_times, baseline_H0, kind='previous',
                            bounds_error=False, fill_value=(0, baseline_H0[-1]))
        
        # S(t|X) = exp(-H0(t) * exp(f(X)))
        survival_probs = np.zeros((len(X), len(times_grid)))
        for i, t in enumerate(times_grid):
            H0_t = H0_interp(t)
            survival_probs[:, i] = np.exp(-H0_t * exp_risk)
        
        return survival_probs


# ===================================================================================
# PART 6: VISUALIZATION FOR PUBLICATION
# ===================================================================================

class SurvivalVisualization:
    """Publication-quality visualizations for survival analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_training_curves(self, history: Dict, save: bool = True) -> plt.Figure:
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1 = axes[0]
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
        if history['val_loss']:
            ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cox Partial Likelihood Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # C-index curves
        ax2 = axes[1]
        ax2.plot(epochs, history['train_cindex'], 'b-', linewidth=2, label='Train')
        if history['val_cindex']:
            ax2.plot(epochs, history['val_cindex'], 'r-', linewidth=2, label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('C-index')
        ax2.set_title('Concordance Index During Training')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.4, 1.0)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
            print(f"   Saved: training_curves.png")
        
        return fig
    
    def plot_risk_distribution(self, risk_scores: np.ndarray, events: np.ndarray,
                              save: bool = True) -> plt.Figure:
        """Plot risk score distribution by event status."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        risk_uncensored = risk_scores[events == 1]
        risk_censored = risk_scores[events == 0]
        
        # Only plot uncensored data
        ax.hist(risk_uncensored, bins=30, alpha=0.6, color='red', 
                label=f'Uncensored (n={len(risk_uncensored)})', density=True)
        
        # Only plot censored if there are any censored events
        if len(risk_censored) > 0:
            ax.hist(risk_censored, bins=30, alpha=0.6, color='blue',
                    label=f'Censored (n={len(risk_censored)})', density=True)
            title = 'Risk Score Distribution by Event Status'
        else:
            title = 'Risk Score Distribution (Uncensored Events)'
        
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'risk_distribution.png'))
            print(f"   Saved: risk_distribution.png")
        
        return fig
    
    def plot_survival_curves(self, survival_probs: np.ndarray, times_grid: np.ndarray,
                            risk_scores: np.ndarray, n_groups: int = 4,
                            save: bool = True) -> plt.Figure:
        """Plot Kaplan-Meier style survival curves by risk group."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Divide into risk groups
        risk_quantiles = np.percentile(risk_scores, np.linspace(0, 100, n_groups + 1))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_groups))
        
        for i in range(n_groups):
            mask = (risk_scores >= risk_quantiles[i]) & (risk_scores < risk_quantiles[i + 1])
            if i == n_groups - 1:
                mask = risk_scores >= risk_quantiles[i]
            
            if mask.sum() > 0:
                mean_survival = survival_probs[mask].mean(axis=0)
                label = f'Risk Q{i+1} (n={mask.sum()})'
                ax.plot(times_grid, mean_survival, color=colors[i], linewidth=2, label=label)
        
        ax.set_xlabel('Time (Cycles)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Predicted Survival Curves by Risk Quartile')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(times_grid.min(), times_grid.max())
        ax.set_ylim(0, 1)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'survival_curves.png'))
            print(f"   Saved: survival_curves.png")
        
        return fig
    
    def plot_calibration(self, predicted_survival: np.ndarray, actual_times: np.ndarray,
                        events: np.ndarray, eval_time: float,
                        save: bool = True) -> plt.Figure:
        """Plot calibration curve at a specific time point."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Bin predictions
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        
        actual_survival = (actual_times > eval_time).astype(float)
        
        bin_centers = []
        observed_probs = []
        
        for i in range(n_bins):
            mask = (predicted_survival >= bins[i]) & (predicted_survival < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append(predicted_survival[mask].mean())
                observed_probs.append(actual_survival[mask].mean())
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        ax.scatter(bin_centers, observed_probs, s=100, c='steelblue', edgecolors='black', 
                  label='Observed')
        ax.plot(bin_centers, observed_probs, 'b-', alpha=0.5)
        
        ax.set_xlabel(f'Predicted Survival Probability at t={eval_time:.0f}')
        ax.set_ylabel('Observed Proportion Surviving')
        ax.set_title('Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'calibration_plot.png'))
            print(f"   Saved: calibration_plot.png")
        
        return fig
    
    def plot_predicted_vs_actual(self, predicted_rul: np.ndarray, actual_rul: np.ndarray,
                                save: bool = True) -> plt.Figure:
        """Plot predicted vs actual RUL."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(actual_rul, predicted_rul, alpha=0.3, c='steelblue', s=20)
        
        # Perfect prediction line
        lims = [min(actual_rul.min(), predicted_rul.min()),
                max(actual_rul.max(), predicted_rul.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        
        # Compute metrics
        mae = np.mean(np.abs(actual_rul - predicted_rul))
        rmse = np.sqrt(np.mean((actual_rul - predicted_rul) ** 2))
        corr = np.corrcoef(actual_rul, predicted_rul)[0, 1]
        
        ax.set_xlabel('Actual RUL (Cycles)')
        ax.set_ylabel('Predicted RUL (Cycles)')
        ax.set_title('Predicted vs Actual Remaining Useful Life')
        
        textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nCorr: {corr:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'predicted_vs_actual.png'))
            print(f"   Saved: predicted_vs_actual.png")
        
        return fig
    
    def plot_comprehensive_metrics(self, metrics: Dict, save: bool = True) -> plt.Figure:
        """Create a comprehensive metrics summary figure."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. C-index comparison (if cross-validation)
        ax1 = axes[0, 0]
        if 'cv_cindices' in metrics:
            folds = range(1, len(metrics['cv_cindices']) + 1)
            bars = ax1.bar(folds, metrics['cv_cindices'], color='steelblue', edgecolor='black')
            ax1.axhline(y=np.mean(metrics['cv_cindices']), color='red', linestyle='--',
                       linewidth=2, label=f"Mean: {np.mean(metrics['cv_cindices']):.4f}")
            ax1.set_xlabel('Fold')
            ax1.set_ylabel('C-index')
            ax1.set_title('Cross-Validation C-index by Fold')
            ax1.legend()
            ax1.set_ylim(0.5, 1.0)
            ax1.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Time-dependent AUC
        ax2 = axes[0, 1]
        if 'td_auc' in metrics:
            times = list(metrics['td_auc'].keys())
            aucs = list(metrics['td_auc'].values())
            ax2.plot(times, aucs, 'b-o', linewidth=2, markersize=8)
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Time (Cycles)')
            ax2.set_ylabel('AUC')
            ax2.set_title('Time-dependent AUC')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0.4, 1.0)
        
        # 3. Brier Score over time
        ax3 = axes[1, 0]
        if 'brier_scores' in metrics:
            times = list(metrics['brier_scores'].keys())
            bs = list(metrics['brier_scores'].values())
            ax3.plot(times, bs, 'g-o', linewidth=2, markersize=8)
            ax3.set_xlabel('Time (Cycles)')
            ax3.set_ylabel('Brier Score')
            ax3.set_title('Brier Score over Time (Lower is Better)')
            ax3.grid(True, alpha=0.3)
        
        # 4. Summary metrics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['C-index (Test)', f"{metrics.get('test_cindex', 'N/A'):.4f}"],
            ['IBS', f"{metrics.get('ibs', 'N/A'):.4f}"],
            ['Mean Brier Score', f"{metrics.get('mean_brier', 'N/A'):.4f}"],
            ['MAE (RUL)', f"{metrics.get('mae', 'N/A'):.2f} cycles"],
            ['RMSE (RUL)', f"{metrics.get('rmse', 'N/A'):.2f} cycles"],
        ]
        
        table = ax4.table(cellText=summary_data, loc='center', cellLoc='center',
                         colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4a90d9')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('DeepSurv Model Performance Metrics', fontsize=18, 
                    fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, 'comprehensive_metrics.png'))
            print(f"   Saved: comprehensive_metrics.png")
        
        return fig


# ===================================================================================
# PART 7: MAIN EXPERIMENT PIPELINE
# ===================================================================================

def run_experiment(config: Config):
    """Run the complete DeepSurv experiment pipeline."""
    
    print("\n" + "=" * 70)
    print("  DEEPSURV FOR BATTERY RUL PREDICTION - JOURNAL EXPERIMENT  ")
    print("=" * 70)
    
    # 1. EDA (only for first run, skip for ablation sub-experiments)
    # Skip EDA if this is an ablation sub-experiment (output_dir contains 'ablation_')
    is_ablation_subexp = 'ablation_' in config.output_dir and config.output_dir != config.base_dir + '/results'
    
    if not is_ablation_subexp:
        eda = BatteryEDA(config)
        battery_stats = eda.run_full_eda()
    else:
        # For ablation sub-experiments, only load data without generating plots
        eda = BatteryEDA(config)
        eda.load_data()  # Just load data, don't generate all EDA plots
        battery_stats = eda.battery_stats if hasattr(eda, 'battery_stats') else None
    
    # 2. Data Processing
    processor = BatteryDataProcessor(config)
    df, feature_names = processor.load_and_process()
    
    # 3. Prepare train/test split
    data = processor.prepare_survival_data(df, test_size=0.2)
    
    # 3.5. Plot individual battery curves with train/test labels
    print("\n" + "=" * 60)
    print("PART 1.5: INDIVIDUAL BATTERY DEGRADATION CURVES")
    print("=" * 60)
    
    train_bats = set(data['train_bats'])
    test_bats = set(data['test_bats'])
    
    print(f"\n Train Batteries ({len(train_bats)}): {sorted(train_bats)}")
    print(f" Test Batteries ({len(test_bats)}): {sorted(test_bats)}")
    
    # Update EDA plots with train/test information
    eda.plot_capacity_degradation(train_batteries=train_bats, test_batteries=test_bats)
    
    # Generate individual plots for EACH battery ID
    eda.plot_individual_battery_curves(train_batteries=train_bats, test_batteries=test_bats)
    
    # 4. Train DeepSurv
    train_data = {'X': data['X_train'], 'T': data['T_train'], 'E': data['E_train']}
    val_data = {'X': data['X_test'], 'T': data['T_test'], 'E': data['E_test']}
    
    trainer = DeepSurvTrainer(config)
    history = trainer.train(train_data, val_data)
    
    # 5. Evaluate on test set
    print("\n" + "=" * 60)
    print("PART 4: EVALUATION METRICS")
    print("=" * 60)
    
    test_risk = trainer.predict_risk(data['X_test'])
    
    # C-index
    test_cindex = SurvivalMetrics.concordance_index(
        test_risk, data['T_test'], data['E_test']
    )
    print(f"\n Test Set C-index: {test_cindex:.4f}")
    
    # Compute baseline hazard on training data
    train_risk = trainer.predict_risk(data['X_train'])
    baseline_times, baseline_H0 = trainer.compute_baseline_hazard(
        train_risk, data['T_train'], data['E_train']
    )
    
    # Survival predictions
    times_grid = np.linspace(data['T_test'].min(), data['T_test'].max(), 100)
    survival_probs = trainer.predict_survival(
        data['X_test'], times_grid, baseline_times, baseline_H0
    )
    
    # Expected RUL (area under survival curve)
    expected_rul = np.trapz(survival_probs, times_grid, axis=1)
    actual_rul = data['T_test']
    
    mae = np.mean(np.abs(actual_rul - expected_rul))
    rmse = np.sqrt(np.mean((actual_rul - expected_rul) ** 2))
    print(f"   MAE (RUL): {mae:.2f} cycles")
    print(f"   RMSE (RUL): {rmse:.2f} cycles")
    
    # Time-dependent AUC
    td_auc = {}
    eval_times = np.percentile(data['T_test'][data['E_test'] == 1], [25, 50, 75])
    for t in eval_times:
        auc = SurvivalMetrics.time_dependent_auc(test_risk, data['T_test'], data['E_test'], t)
        td_auc[int(t)] = auc
        print(f"   AUC at t={int(t)}: {auc:.4f}")
    
    # Brier Score
    brier_scores = {}
    for t in eval_times:
        # Get survival probability at time t
        t_idx = np.argmin(np.abs(times_grid - t))
        surv_at_t = survival_probs[:, t_idx]
        bs = SurvivalMetrics.brier_score(surv_at_t, data['T_test'], data['E_test'], t)
        brier_scores[int(t)] = bs
        print(f"   Brier Score at t={int(t)}: {bs:.4f}")
    
    mean_brier = np.mean(list(brier_scores.values()))
    print(f"   Mean Brier Score: {mean_brier:.4f}")
    
    # 6. Visualizations
    print("\n" + "=" * 60)
    print("PART 5: GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    viz = SurvivalVisualization(config.output_dir)
    viz.plot_training_curves(history)
    viz.plot_risk_distribution(test_risk, data['E_test'])
    viz.plot_survival_curves(survival_probs, times_grid, test_risk)
    viz.plot_predicted_vs_actual(expected_rul, actual_rul)
    
    # Calibration at median time
    median_time = np.median(eval_times)
    t_idx = np.argmin(np.abs(times_grid - median_time))
    viz.plot_calibration(survival_probs[:, t_idx], data['T_test'], 
                        data['E_test'], median_time)
    
    # Comprehensive metrics plot
    metrics = {
        'test_cindex': test_cindex,
        'td_auc': td_auc,
        'brier_scores': brier_scores,
        'mean_brier': mean_brier,
        'mae': mae,
        'rmse': rmse,
        'ibs': mean_brier  # Simplified IBS approximation
    }
    viz.plot_comprehensive_metrics(metrics)
    
    # 7. Save results
    results = {
        'test_cindex': float(test_cindex),
        'mae': float(mae),
        'rmse': float(rmse),
        'mean_brier_score': float(mean_brier),
        'time_dependent_auc': {str(k): float(v) for k, v in td_auc.items()},
        'brier_scores': {str(k): float(v) for k, v in brier_scores.items()},
        'config': {
            'hidden_layers': config.hidden_layers,
            'dropout': config.dropout,
            'learning_rate': config.learning_rate,
            'epochs': config.epochs,
            'soh_threshold': config.soh_threshold
        },
        'training_samples': len(data['X_train']),
        'test_samples': len(data['X_test']),
        'features': feature_names
    }
    
    with open(os.path.join(config.output_dir, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to: {config.output_dir}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'battery_id': data['bat_test'],
        'actual_rul': actual_rul,
        'predicted_rul': expected_rul,
        'risk_score': test_risk,
        'event': data['E_test']
    })
    predictions_df.to_csv(os.path.join(config.output_dir, 'test_predictions.csv'), index=False)
    
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE  ")
    print("=" * 70)
    
    return results, trainer, data


def run_cross_validation(config: Config) -> Dict:
    """Run K-Fold cross-validation for robust performance estimation."""
    
    print("\n" + "=" * 70)
    print("  K-FOLD CROSS-VALIDATION  ")
    print("=" * 70)
    
    # Load data
    processor = BatteryDataProcessor(config)
    df, feature_names = processor.load_and_process()
    
    # Get battery IDs
    battery_ids = df['battery_id'].unique()
    
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    cv_results = {
        'fold_cindices': [],
        'fold_maes': [],
        'fold_rmses': []
    }
    
    print(f"\n🔄 Running {config.n_folds}-Fold Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(battery_ids)):
        print(f"\n--- Fold {fold + 1}/{config.n_folds} ---")
        
        train_bats = battery_ids[train_idx]
        val_bats = battery_ids[val_idx]
        
        train_df = df[df['battery_id'].isin(train_bats)]
        val_df = df[df['battery_id'].isin(val_bats)]
        
        # Prepare data
        X_train = train_df[feature_names].values.astype(np.float32)
        T_train = train_df['time'].values.astype(np.float32)
        E_train = train_df['event'].values.astype(np.float32)
        
        X_val = val_df[feature_names].values.astype(np.float32)
        T_val = val_df['time'].values.astype(np.float32)
        E_val = val_df['event'].values.astype(np.float32)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        
        # Train
        train_data = {'X': X_train_scaled, 'T': T_train, 'E': E_train}
        val_data = {'X': X_val_scaled, 'T': T_val, 'E': E_val}
        
        trainer = DeepSurvTrainer(config)
        trainer.config.epochs = 100  # Reduced for CV
        history = trainer.train(train_data, val_data)
        
        # Evaluate
        val_risk = trainer.predict_risk(X_val_scaled)
        cindex = SurvivalMetrics.concordance_index(val_risk, T_val, E_val)
        
        # Compute baseline hazard and expected RUL
        train_risk = trainer.predict_risk(X_train_scaled)
        baseline_times, baseline_H0 = trainer.compute_baseline_hazard(
            train_risk, T_train, E_train
        )
        
        times_grid = np.linspace(T_val.min(), T_val.max(), 50)
        survival_probs = trainer.predict_survival(
            X_val_scaled, times_grid, baseline_times, baseline_H0
        )
        expected_rul = np.trapz(survival_probs, times_grid, axis=1)
        
        mae = np.mean(np.abs(T_val - expected_rul))
        rmse = np.sqrt(np.mean((T_val - expected_rul) ** 2))
        
        cv_results['fold_cindices'].append(cindex)
        cv_results['fold_maes'].append(mae)
        cv_results['fold_rmses'].append(rmse)
        
        print(f"   C-index: {cindex:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 50)
    print(f"C-index:  {np.mean(cv_results['fold_cindices']):.4f} ± {np.std(cv_results['fold_cindices']):.4f}")
    print(f"MAE:      {np.mean(cv_results['fold_maes']):.2f} ± {np.std(cv_results['fold_maes']):.2f}")
    print(f"RMSE:     {np.mean(cv_results['fold_rmses']):.2f} ± {np.std(cv_results['fold_rmses']):.2f}")
    
    # Save CV results
    cv_results['mean_cindex'] = float(np.mean(cv_results['fold_cindices']))
    cv_results['std_cindex'] = float(np.std(cv_results['fold_cindices']))
    cv_results['mean_mae'] = float(np.mean(cv_results['fold_maes']))
    cv_results['mean_rmse'] = float(np.mean(cv_results['fold_rmses']))
    
    with open(os.path.join(config.output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # Plot CV results
    viz = SurvivalVisualization(config.output_dir)
    metrics = {'cv_cindices': cv_results['fold_cindices']}
    viz.plot_comprehensive_metrics(metrics)
    
    return cv_results


def run_ablation_study(base_config: Config) -> Dict:
    """
    Run ablation study by comparing different feature configurations.
    
    This implements the feedback from advisor: comparing DeepSurv performance
    with different input feature sets (basic, dynamic, full).
    
    Args:
        base_config: Base configuration object
        
    Returns:
        Dictionary containing results for all feature sets
    """
    print("\n" + "=" * 70)
    print("  ABLATION STUDY: INPUT FEATURE CONFIGURATIONS  ")
    print("=" * 70)
    print("\n본 연구에서는 DeepSurv를 대표 모델로 설정하고,")
    print("입력 특징 구성에 따른 성능 및 예측 안정성 변화를 분석합니다.")
    print("\nFeature Set Configurations:")
    print("  - Case 1 (basic): capacity, soh (최소 정보 기반)")
    print("  - Case 2 (dynamic): basic + derivatives (열화 속도 반영)")
    print("  - Case 3 (full): dynamic + rolling statistics (변동성 반영)")
    print("=" * 70)
    
    feature_sets = ['basic', 'dynamic', 'full']
    results = {}
    
    for feature_set in feature_sets:
        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENTING WITH: {feature_set.upper()} FEATURE SET  ")
        print(f"{'=' * 70}")
        
        # Create config with specific feature set
        # Use separate output directory for each feature set
        feature_output_dir = os.path.join(base_config.output_dir, f'ablation_{feature_set}')
        os.makedirs(feature_output_dir, exist_ok=True)
        
        config = Config(
            base_dir=base_config.base_dir,
            min_init_capacity=base_config.min_init_capacity,
            min_cycles=base_config.min_cycles,
            soh_threshold=base_config.soh_threshold,
            hidden_layers=base_config.hidden_layers,
            dropout=base_config.dropout,
            learning_rate=base_config.learning_rate,
            weight_decay=base_config.weight_decay,
            batch_size=base_config.batch_size,
            epochs=base_config.epochs,
            patience=base_config.patience,
            n_folds=base_config.n_folds,
            feature_set_name=feature_set
        )
        # Override output directory
        config.output_dir = feature_output_dir
        
        # IMPORTANT: Generate preprocessing plots in main results folder (not in ablation sub-folders)
        # This ensures preprocessing comparison is available for the paper
        if feature_set == 'full':  # Only generate once, using full feature set
            main_eda = BatteryEDA(base_config)
            main_eda.load_data()
            main_eda.plot_before_after_preprocessing(save=True)
            print(f"\nPreprocessing before/after plots saved to: {base_config.output_dir}/")
        
        # Run experiment with this feature set
        try:
            exp_results, trainer, data = run_experiment(config)
            
            # Store results with feature set name
            feature_names = exp_results.get('features', exp_results.get('feature_names', []))
            results[feature_set] = {
                'config': feature_set,
                'feature_names': feature_names,
                'test_cindex': exp_results['test_cindex'],
                'test_mae': exp_results.get('mae', None),
                'test_rmse': exp_results.get('rmse', None),
                'train_cindex': exp_results.get('train_cindex', None),
                'n_features': len(feature_names),
                'n_train_samples': len(data['X_train']),
                'n_test_samples': len(data['X_test'])
            }
            
            print(f"\n{feature_set.upper()} Results:")
            feature_names = exp_results.get('features', exp_results.get('feature_names', []))
            print(f"  Features: {feature_names}")
            print(f"  Test C-index: {exp_results['test_cindex']:.4f}")
            
        except Exception as e:
            print(f"\nError in {feature_set} experiment: {str(e)}")
            import traceback
            traceback.print_exc()
            results[feature_set] = {'error': str(e)}
    
    # Create comparison summary
    print("\n" + "=" * 70)
    print("  ABLATION STUDY SUMMARY  ")
    print("=" * 70)
    
    comparison_data = []
    for feature_set in feature_sets:
        if feature_set in results and 'error' not in results[feature_set]:
            r = results[feature_set]
            comparison_data.append({
                'Feature Set': feature_set,
                'N Features': r['n_features'],
                'Test C-index': f"{r['test_cindex']:.4f}",
                'Test MAE': f"{r['test_mae']:.2f}" if r['test_mae'] else "N/A",
                'Test RMSE': f"{r['test_rmse']:.2f}" if r['test_rmse'] else "N/A"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_path = os.path.join(base_config.output_dir, 'ablation_study_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nComparison table saved to: {comparison_path}")
    
    # Create visualization
    _plot_ablation_results(results, base_config.output_dir)
    
    # Save full results as JSON
    results_path = os.path.join(base_config.output_dir, 'ablation_study_results.json')
    
    # Convert to JSON-serializable format
    json_results = {}
    for key, val in results.items():
        if 'error' not in val:
            json_results[key] = {
                'config': val['config'],
                'feature_names': val['feature_names'],
                'test_cindex': float(val['test_cindex']),
                'test_mae': float(val['test_mae']) if val['test_mae'] else None,
                'test_rmse': float(val['test_rmse']) if val['test_rmse'] else None,
                'n_features': val['n_features']
            }
        else:
            json_results[key] = val
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Full results saved to: {results_path}")
    
    return results


def _plot_ablation_results(results: Dict, output_dir: str):
    """Create visualization comparing ablation study results."""
    feature_sets = ['basic', 'dynamic', 'full']
    valid_results = {k: v for k, v in results.items() 
                     if k in feature_sets and 'error' not in v}
    
    if len(valid_results) == 0:
        print("No valid results to plot.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Extract data
    set_names = list(valid_results.keys())
    cindices = [valid_results[s]['test_cindex'] for s in set_names]
    maes = [valid_results[s]['test_mae'] for s in set_names if valid_results[s]['test_mae']]
    rmses = [valid_results[s]['test_rmse'] for s in set_names if valid_results[s]['test_rmse']]
    n_features = [valid_results[s]['n_features'] for s in set_names]
    
    # Plot 1: C-index comparison
    ax1 = axes[0]
    bars1 = ax1.bar(set_names, cindices, color=['#3498db', '#2ecc71', '#e74c3c'], 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Test C-index', fontsize=12, fontweight='bold')
    ax1.set_title('C-index by Feature Set', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, cindices):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: MAE comparison
    if maes:
        ax2 = axes[1]
        bars2 = ax2.bar(set_names[:len(maes)], maes, color=['#3498db', '#2ecc71', '#e74c3c'], 
                        alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Test MAE (cycles)', fontsize=12, fontweight='bold')
        ax2.set_title('Mean Absolute Error by Feature Set', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, maes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Number of features
    ax3 = axes[2]
    bars3 = ax3.bar(set_names, n_features, color=['#3498db', '#2ecc71', '#e74c3c'], 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax3.set_title('Feature Set Complexity', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars3, n_features):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Ablation Study: Input Feature Configuration Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'ablation_study_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ablation comparison plot saved to: {save_path}")


# ===================================================================================
# ENTRY POINT
# ===================================================================================

if __name__ == "__main__":
    # Create configuration
    config = Config()
    
    # Option 1: Run ablation study (recommended for journal paper)
    # This compares different input feature configurations
    ablation_results = run_ablation_study(config)
    
    # Option 2: Run single experiment with specific feature set
    # config.feature_set_name = 'full'  # Options: 'basic', 'dynamic', 'full'
    # results, trainer, data = run_experiment(config)
    
    # Option 3: Run cross-validation for more robust results
    # cv_results = run_cross_validation(config)
    
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE  ")
    print("=" * 70)
    print("Check the 'results' folder for outputs.")
    print("  - ablation_study_comparison.csv: Comparison table")
    print("  - ablation_study_comparison.png: Visualization")
    print("  - ablation_study_results.json: Full results")

