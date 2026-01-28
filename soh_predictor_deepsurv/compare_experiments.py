import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 현재 디렉토리 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model import DeepSurv, cox_ph_loss
from data_loader import BatteryEISDataset, DataLoader

def train_and_evaluate(dataset, name, epochs=100):
    print(f"\n--- Training on {name} Dataset ---")
    
    if len(dataset) < 2:
        print("Not enough data to train.")
        return 0.0

    # Train/Test Split logic handled inside dataset or just use full for small data comparison
    # For fair comparison with small data, let's use full dataset for both training and evaluation (Self-consistency)
    # or simple random split. Given small size, self-consistency or LOOCV is better but let's stick to simple train/eval on same for now to see capacity.
    
    # MinMaxScaler
    X = dataset.X
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-7
    X_scaled = (X - mean) / std
    
    dataset.X = X_scaled
    
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSurv(in_features=dataset.X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            x = batch['x'].to(device)
            t = batch['t'].to(device).float()
            e = batch['e'].to(device).float()
            
            if len(x) < 2: continue # Batch Norm fails
            
            sorted_idx = torch.argsort(t, descending=True)
            x = x[sorted_idx]
            t = t[sorted_idx]
            e = e[sorted_idx]
            
            optimizer.zero_grad()
            log_h = model(x)
            loss = cox_ph_loss(log_h, e)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    all_risk = []
    all_t = []
    
    with torch.no_grad():
        # Full batch
        x = torch.tensor(dataset.X).to(device)
        t = torch.tensor(dataset.SOH).float()
        
        log_h = model(x).cpu().numpy().flatten()
        all_risk = log_h
        all_t = t.numpy()
        
    # Analysis
    # SOH Prediction (Linear Fit)
    A = np.vstack([all_risk, np.ones(len(all_risk))]).T
    m, c = np.linalg.lstsq(A, all_t, rcond=None)[0]
    pred_soh = m * all_risk + c
    
    r2 = 1 - (np.sum((all_t - pred_soh)**2) / np.sum((all_t - np.mean(all_t))**2))
    mse = np.mean((all_t - pred_soh)**2)
    
    print(f"{name} Result -> R2: {r2:.4f}, MSE: {mse:.4f}")
    return r2

def main():
    base_path = Path(__file__).resolve().parent.parent / "datasets"
    
    # 1. Spec Only
    ds_spec = BatteryEISDataset(base_path, mode='train', augment=True) # Augment for Spec
    # Force load ONLY spec (hacky way or modifying class? simpler to just clear filtering in class but logic is coupled)
    # Actually, the class loads EVERYTHING by default based on _load calls.
    # Let's instantiate and filter manually by 'Source' which we added in data_loader.py records?
    # Wait, we added 'Source' key in record dicts in previous turn!
    
    # Re-verify data_loader.py structure for filtering
    # We will just reload and filter df.
    
    print("Loading all data...")
    full_ds = BatteryEISDataset(base_path, mode='train', augment=False) # No augment initially
    
    # Filter for Spec
    df_spec = full_ds.df[full_ds.df['Source'] == 'spectroscopy'].copy()
    ds_spec = BatteryEISDataset(base_path, mode='train', augment=False) # Dummy init
    ds_spec.df = df_spec
    ds_spec.X = df_spec[full_ds.feature_cols].values.astype(np.float32)
    ds_spec.SOH = df_spec['SOH'].values.astype(np.float32)
    ds_spec.Event = np.ones_like(ds_spec.SOH)
    # Apply Augmentation manually for Spec
    ds_spec._apply_augmentation(full_ds.feature_cols, factor=10) # Augment Spec
    ds_spec.X = ds_spec.df[full_ds.feature_cols].values.astype(np.float32)
    ds_spec.SOH = ds_spec.df['SOH'].values.astype(np.float32)
    ds_spec.Event = np.ones_like(ds_spec.SOH)
    
    
    # Filter for Company
    df_comp = full_ds.df[full_ds.df['Source'] == 'company_battery'].copy()
    ds_comp = BatteryEISDataset(base_path, mode='train', augment=False)
    ds_comp.df = df_comp
    ds_comp.X = df_comp[full_ds.feature_cols].values.astype(np.float32)
    ds_comp.SOH = df_comp['SOH'].values.astype(np.float32)
    ds_comp.Event = np.ones_like(ds_comp.SOH)
    # company data might not need augmentation or maybe yes? Let's keep it raw for request comparison.
    
    r2_spec = train_and_evaluate(ds_spec, "Spectroscopy (Augmented)")
    r2_comp = train_and_evaluate(ds_comp, "Company Battery") # Raw
    
    print("\n=== Summary ===")
    print(f"Spectroscopy Only (Augmented): R2 = {r2_spec:.4f}")
    print(f"Company Battery Only: R2 = {r2_comp:.4f}")

if __name__ == "__main__":
    main()
