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
from data_loader import get_dataloader

def c_index(risk_scores, t, e):
    """
    Compute Concordance Index (C-Index)
    O(N^2) simple implementation
    """
    n = len(t)
    count = 0
    correct = 0
    
    # t: SOH (Time), e: Event
    # risk_scores: Predicted Hazard
    # Condition: If T_i < T_j (i dies before j), then Risk_i should be > Risk_j
    
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

def train():
    # 경로 설정
    base_path = Path(__file__).resolve().parent.parent # 건민님 코드
    data_path = base_path / "datasets"
    
    print(f"Loading data from {data_path}...")
    train_loader, test_loader, feature_cols = get_dataloader(data_path, batch_size=32)
    
    if len(feature_cols) == 0:
        print("Error: No data loaded. Please check dataset path.")
        return

    print(f"Features: {len(feature_cols)}, Train Batches: {len(train_loader)}, Test Batches: {len(test_loader)}")
    
    # 모델 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSurv(in_features=len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) # L2 reg is important for Cox
    
    epochs = 100
    train_hist = []
    
    print("Start Training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            x = batch['x'].to(device)
            t = batch['t'].to(device).float()
            e = batch['e'].to(device).float()
            
            # Sort by Time (SOH) Descending for Cox Loss
            sorted_idx = torch.argsort(t, descending=True)
            x = x[sorted_idx]
            t = t[sorted_idx]
            e = e[sorted_idx]
            
            optimizer.zero_grad()
            log_h = model(x)
            
            loss = cox_ph_loss(log_h, e)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
            train_hist.append(total_loss / len(train_loader))

    # 평가 및 분석
    model.eval()
    all_risk = []
    all_t = []
    all_e = []
    
    loader_to_eval = test_loader if len(test_loader) > 0 else train_loader
    if len(test_loader) == 0:
        print("Warning: Test set is empty (probably too few Battery IDs). Evaluating on Training set instead.")
    
    with torch.no_grad():
        for batch in loader_to_eval:
            x = batch['x'].to(device)
            t = batch['t']
            e = batch['e']
            
            log_h = model(x).cpu().numpy()
            all_risk.extend(log_h.flatten())
            all_t.extend(t.numpy())
            all_e.extend(e.numpy())
            
    all_risk = np.array(all_risk)
    all_t = np.array(all_t)
    all_e = np.array(all_e)
    
    ci = c_index(all_risk, all_t, all_e)
    print(f"\nEvaluation Result (Test Set):")
    print(f"C-Index: {ci:.4f}")
    
    # SOH Prediction Analysis
    # Risk Score와 SOH의 관계 (상관관계)
    corr = np.corrcoef(all_risk, all_t)[0, 1]
    print(f"Correlation between Risk Score and SOH: {corr:.4f} (Expected Negative)")
    
    # 시각화 (Risk vs SOH)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(all_t, all_risk, alpha=0.6)
    plt.xlabel('True SOH (%)')
    plt.ylabel('Predicted Risk Score (Log Hazard)')
    plt.title(f'Risk Score vs SOH (Corr: {corr:.2f})')
    plt.grid(True)
    
    # SOH 예측 변환 (Linear Regression: Risk -> SOH)
    # Simple fitting: SOH = a * Risk + b
    A = np.vstack([all_risk, np.ones(len(all_risk))]).T
    m, c = np.linalg.lstsq(A, all_t, rcond=None)[0]
    
    pred_soh = m * all_risk + c
    
    # MSE, R2
    mse = np.mean((all_t - pred_soh)**2)
    r2 = 1 - (np.sum((all_t - pred_soh)**2) / np.sum((all_t - np.mean(all_t))**2))
    
    print(f"Derived SOH Prediction R^2: {r2:.4f}")
    print(f"Derived SOH Prediction MSE: {mse:.4f}")
    
    plt.subplot(1, 2, 2)
    plt.scatter(all_t, pred_soh, alpha=0.6, color='orange')
    plt.plot([min(all_t), max(all_t)], [min(all_t), max(all_t)], 'r--')
    plt.xlabel('True SOH (%)')
    plt.ylabel('Predicted SOH (from Risk)')
    plt.title(f'SOH Prediction (R2: {r2:.2f})')
    plt.grid(True)
    
    save_path = current_dir / "soh_deepsurv_result.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    train()
