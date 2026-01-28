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

# 폰트 설정 (한글 깨짐 방지 및 스타일)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def c_index(risk_scores, t, e):
    """
    Compute Concordance Index (C-Index)
    O(N^2) simple implementation
    """
    n = len(t)
    count = 0
    correct = 0
    
    for i in range(n):
        for j in range(i+1, n):
            if e[i] == 0 and e[j] == 0: continue
            
            t_i, t_j = t[i], t[j]
            r_i, r_j = risk_scores[i], risk_scores[j]
            
            if t_i < t_j and e[i] == 1:
                count += 1
                if r_i > r_j: correct += 1
                elif r_i == r_j: correct += 0.5
            elif t_j < t_i and e[j] == 1:
                count += 1
                if r_j > r_i: correct += 1
                elif r_j == r_i: correct += 0.5
                
    return correct / count if count > 0 else 0.0

def train_one_fold(train_ids, test_ids, fold_idx):
    print(f"\n=== Fold {fold_idx}: Train {train_ids} -> Test {test_ids} ===")
    
    base_path = Path(__file__).resolve().parent.parent / "datasets"
    
    # Dataset Load (Augmentation은 Train에만 적용되도록 data_loader 수정 필요, 
    # 하지만 현재 구조상 생성자에서 split하므로, 수동으로 구성)
    
    # 1. Load ALL Data first (No Augmentation initially)
    # We need to access the raw records and split them manually to be safe.
    # Instantiate dataset just to load records
    temp_ds = BatteryEISDataset(base_path, mode='train', augment=False)
    # Filter only Spectroscopy
    spectroscopy_records = [r for r in temp_ds.records if r.get('Source') == 'spectroscopy']
    
    if not spectroscopy_records:
        print("No Spectroscopy records found.")
        return None
    
    df_all = pd.DataFrame(spectroscopy_records)
    # 정규화 (Missing 0)
    cols = [c for c in df_all.columns if c.startswith('R_') or c.startswith('X_')]
    df_all[cols] = df_all[cols].fillna(0)
    
    # Split
    # battery_id extraction assumes 'B2', 'B3' format
    # data_loader.py: extract_battery_id_simple returns 'B2', 'B3'...
    
    train_df = df_all[df_all['BATTERY_ID'].isin(train_ids)].copy()
    test_df = df_all[df_all['BATTERY_ID'].isin(test_ids)].copy()
    
    print(f"Train Samples (Raw): {len(train_df)}, Test Samples: {len(test_df)}")
    
    # Augmentation for Train (Manual)
    feature_cols = cols
    X_train_raw = train_df[feature_cols].values
    SOH_train_raw = train_df['SOH'].values
    
    aug_X = [X_train_raw]
    aug_SOH = [SOH_train_raw]
    
    # Augmentation Loop (Factor 10 ~ 20 to suffice)
    factor = 20
    noise_std = 0.015
    
    np.random.seed(42 + fold_idx) # Vary seed slightly
    for _ in range(factor):
        noise = np.random.normal(0, noise_std, size=X_train_raw.shape)
        X_new = X_train_raw + (X_train_raw * noise)
        aug_X.append(X_new)
        aug_SOH.append(SOH_train_raw)
        
    X_train = np.vstack(aug_X).astype(np.float32)
    SOH_train = np.hstack(aug_SOH).astype(np.float32)
    E_train = np.ones_like(SOH_train)
    
    print(f"Train Samples (Augmented): {len(X_train)}")
    
    # Test Data Prep
    X_test = test_df[feature_cols].values.astype(np.float32)
    SOH_test = test_df['SOH'].values.astype(np.float32)
    E_test = np.ones_like(SOH_test)
    
    # Scaler (Train Statistics)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-7
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSurv(in_features=len(feature_cols), hidden_layers=[64, 32], dropout=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3) # Adjusted for stability
    
    # Training
    model.train()
    batch_size = 32
    n_batches = int(np.ceil(len(X_train) / batch_size))
    
    for epoch in range(150):
        # Shuffle
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        SOH_train = SOH_train[perm]
        E_train = E_train[perm]
        
        epoch_loss = 0
        for i in range(n_batches):
            s = i * batch_size
            e = min(s + batch_size, len(X_train))
            if s >= e: break
            
            bx = torch.tensor(X_train[s:e]).to(device)
            bt = torch.tensor(SOH_train[s:e]).to(device)
            be = torch.tensor(E_train[s:e]).to(device)
            
            if len(bx) < 2: continue
            
            # Sort
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
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_test).to(device)
        risk_pred = model(x_t).cpu().numpy().flatten()
        
    # Result Visual & Metric
    # Risk Score vs SOH
    # Correlation
    if len(risk_pred) > 1:
        corr = np.corrcoef(risk_pred, SOH_test)[0, 1]
    else:
        corr = 0.0
        
    # Calculate C-Index on Test Set
    ci = c_index(risk_pred, SOH_test, E_test)
        
    # Fit SOH (Linear Mapping)
    # Train a simple linear regressor on Train predictions to map Risk -> SOH
    # This simulates "Calibration"
    with torch.no_grad():
        train_risk = model(torch.tensor(X_train).to(device)).cpu().numpy().flatten()
    
    A = np.vstack([train_risk, np.ones(len(train_risk))]).T
    m, c = np.linalg.lstsq(A, SOH_train, rcond=None)[0]
    
    soh_pred_test = m * risk_pred + c
    
    if len(SOH_test) > 0:
        mse = np.mean((SOH_test - soh_pred_test)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(SOH_test - soh_pred_test))
        r2 = 1 - (np.sum((SOH_test - soh_pred_test)**2) / np.sum((SOH_test - np.mean(SOH_test))**2)) if len(SOH_test) > 1 else 0.0
    else:
        mse, rmse, mae, r2 = 0.0, 0.0, 0.0, 0.0
        
    print(f"Test Result -> C-Index: {ci:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    return {
        'fold': fold_idx,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'true_soh': SOH_test,
        'pred_soh': soh_pred_test,
        'risk_score': risk_pred,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'ci': ci
    }

def visualize_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    avg_r2 = 0
    avg_mae = 0
    avg_rmse = 0
    avg_ci = 0
    
    for i, res in enumerate(results):
        ax = axes[i]
        true = res['true_soh']
        pred = res['pred_soh']
        test_ids = res['test_ids']
        
        r2 = res['r2']
        mae = res['mae']
        rmse = res['rmse']
        ci = res['ci']
        
        avg_r2 += r2
        avg_mae += mae
        avg_rmse += rmse
        avg_ci += ci
        
        ax.scatter(true, pred, label='Predicted', color='blue', alpha=0.7)
        # Identity Line
        min_val = min(min(true), min(pred)) - 2
        max_val = max(max(true), max(pred)) + 2
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
        
        ax.set_title(f"Test on {test_ids}\nCI: {ci:.3f}, R²: {r2:.2f}, MAE: {mae:.2f}")
        ax.set_xlabel("True SOH (%)")
        ax.set_ylabel("Predicted SOH (%)")
        ax.grid(True)
        ax.legend()
        
    n = len(results)
    avg_r2 /= n
    avg_mae /= n
    avg_rmse /= n
    avg_ci /= n
    
    plt.suptitle(f"LOOCV Results - Avg CI: {avg_ci:.3f}, Avg R²: {avg_r2:.2f}, Avg RMSE: {avg_rmse:.2f}", fontsize=16)
    plt.tight_layout()
    
    save_path = current_dir / "soh_loocv_results.png"
    plt.savefig(save_path)
    print(f"\nPlots saved to {save_path}")

def main():
    # 3-Fold LOOCV
    # IDs: B2, B3, B5
    # Be careful with ID strings: extract_battery_id_simple returns 'B2', 'B3'..
    
    scenarios = [
        (['B2', 'B3'], ['B5']),
        (['B2', 'B5'], ['B3']),
        (['B3', 'B5'], ['B2'])
    ]
    
    results = []
    for i, (train_ids, test_ids) in enumerate(scenarios):
        res = train_one_fold(train_ids, test_ids, i+1)
        if res:
            results.append(res)
            
    if results:
        visualize_results(results)

if __name__ == "__main__":
    main()
