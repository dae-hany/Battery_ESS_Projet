import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib

# 현재 디렉토리 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model import DeepSurv, cox_ph_loss
from data_loader import BatteryEISDataset, DataLoader

def train_and_save_final_model():
    print(f"\n=== Training Final Model on ALL Spectroscopy Data ===")
    
    base_path = Path(__file__).resolve().parent.parent / "datasets"
    save_dir = Path(__file__).parent / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load ALL Data (Spectroscopy Only)
    # We use augment=False initially to get raw data for calibration, 
    # but we will perform heavy augmentation for training.
    
    # Manually load and augment to ensure full control
    temp_ds = BatteryEISDataset(base_path, mode='train', augment=False)
    spectroscopy_records = [r for r in temp_ds.records if r.get('Source') == 'spectroscopy']
    
    if not spectroscopy_records:
        print("Error: No Spectroscopy records found.")
        return
    
    df_all = pd.DataFrame(spectroscopy_records)
    cols = [c for c in df_all.columns if c.startswith('R_') or c.startswith('X_')]
    df_all[cols] = df_all[cols].fillna(0)
    
    print(f"Total Raw Samples: {len(df_all)}")
    
    # Augmentation
    feature_cols = cols
    X_raw = df_all[feature_cols].values.astype(np.float32)
    SOH_raw = df_all['SOH'].values.astype(np.float32)
    
    factor = 20
    noise_std = 0.015
    
    aug_X = [X_raw]
    aug_SOH = [SOH_raw]
    
    print(f"Applying Data Augmentation (Factor: {factor})...")
    np.random.seed(42) # Fixed seed for reproducibility
    for _ in range(factor):
        noise = np.random.normal(0, noise_std, size=X_raw.shape)
        X_new = X_raw + (X_raw * noise)
        aug_X.append(X_new)
        aug_SOH.append(SOH_raw)
        
    X_train = np.vstack(aug_X).astype(np.float32)
    SOH_train = np.hstack(aug_SOH).astype(np.float32)
    E_train = np.ones_like(SOH_train)
    
    print(f"Total Training Samples: {len(X_train)}")
    
    # Scaling
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-7
    
    X_train_scaled = (X_train - mean) / std
    
    # Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSurv(in_features=len(feature_cols), hidden_layers=[64, 32], dropout=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    
    # Training Loop
    model.train()
    batch_size = 32
    n_batches = int(np.ceil(len(X_train_scaled) / batch_size))
    
    for epoch in range(200): # More epochs for final model
        # Shuffle
        perm = np.random.permutation(len(X_train_scaled))
        X_curr = X_train_scaled[perm]
        SOH_curr = SOH_train[perm]
        E_curr = E_train[perm]
        
        epoch_loss = 0
        for i in range(n_batches):
            s = i * batch_size
            e = min(s + batch_size, len(X_curr))
            if s >= e: break
            
            bx = torch.tensor(X_curr[s:e]).to(device)
            bt = torch.tensor(SOH_curr[s:e]).to(device)
            be = torch.tensor(E_curr[s:e]).to(device)
            
            if len(bx) < 2: continue
            
            sort_idx = torch.argsort(bt, descending=True)
            bx = bx[sort_idx]
            bt = bt[sort_idx]
            be = be[sort_idx]
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = cox_ph_loss(pred, be)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/200 - Loss: {epoch_loss/n_batches:.4f}")
            
    # Save Model Weights
    model_path = save_dir / "deepsurv_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel weights saved to: {model_path}")
    
    # Calibration (Linear Mapping)
    # Use RAW data (or lightly augmented) for clean calibration mapping
    # Let's use the predictions on the TRAINING SET to build the mapper.
    model.eval()
    with torch.no_grad():
        # Using full augmented set for robust regression
        train_risk = model(torch.tensor(X_train_scaled).to(device)).cpu().numpy().flatten()
        
    A = np.vstack([train_risk, np.ones(len(train_risk))]).T
    m, c = np.linalg.lstsq(A, SOH_train, rcond=None)[0]
    
    print(f"Calibration Parameters -> Slope: {m:.4f}, Intercept: {c:.4f}")
    
    # Save Metadata (Scaler, Calibration, Config)
    metadata = {
        'feature_cols': feature_cols,
        'scaler_mean': mean,
        'scaler_std': std,
        'calibration_slope': m,
        'calibration_intercept': c,
        'description': "Final DeepSurv model trained on all B2/B3/B5 Spectroscopy data with augmentation."
    }
    
    meta_path = save_dir / "model_metadata.pkl"
    joblib.dump(metadata, meta_path)
    print(f"Model metadata saved to: {meta_path}")
    print("Training Complete.")

if __name__ == "__main__":
    train_and_save_final_model()
