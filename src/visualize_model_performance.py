import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import os

# src 폴더 내의 모듈을 임포트하기 위해 경로 설정이 필요할 수 있음
# 하지만 같은 폴더에 있으므로 직접 임포트 가능
from DeepSurv_Pytorch_v2 import BatteryDataset, DeepSurv, cox_ph_loss, c_index, METADATA_PATH, DATA_DIR

# ==========================================
# 설정 (Configuration)
# ==========================================
BASE_DIR = r"c:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def run_analysis_and_plot():
    # 데이터셋 로드
    print("데이터셋 로딩 중...")
    dataset = BatteryDataset(METADATA_PATH, DATA_DIR)
    battery_ids = dataset.get_battery_ids()
    
    # K-Fold 설정
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_c_indices = []
    fold_results = []
    
    print(f"시각화를 위한 {k}-Fold 교차 검증 실행 중...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(battery_ids)):
        train_bats = [battery_ids[i] for i in train_idx]
        val_bats = [battery_ids[i] for i in val_idx]
        
        train_indices = dataset.get_indices_for_batteries(train_bats)
        val_indices = dataset.get_indices_for_batteries(val_bats)
        
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)
        
        # 모델 학습
        model = DeepSurv(input_size=4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 50
        
        best_ci = 0.0
        best_risk = None
        best_t = None
        best_e = None
        
        for epoch in range(epochs):
            model.train()
            for x, t, e in train_loader:
                optimizer.zero_grad()
                out = model(x)
                loss = cox_ph_loss(out, t, e)
                loss.backward()
                optimizer.step()
        
            # 검증 (Validation)
            model.eval()
            val_risk = []
            val_t = []
            val_e = []
            with torch.no_grad():
                for x, t, e in val_loader:
                    out = model(x)
                    val_risk.append(out.numpy())
                    val_t.append(t.numpy())
                    val_e.append(e.numpy())
            
            val_risk = np.concatenate(val_risk)
            val_t = np.concatenate(val_t)
            val_e = np.concatenate(val_e)
            
            ci = c_index(val_risk, val_t, val_e)
            if ci > best_ci:
                best_ci = ci
                best_risk = val_risk
                best_t = val_t
                best_e = val_e
        
        print(f"Fold {fold+1} C-Index: {best_ci:.4f}")
        fold_c_indices.append(best_ci)
        
        # 최고 성능 Epoch의 예측값 저장
        fold_results.append({
            'risk': best_risk.flatten(),
            'time': best_t.flatten(),
            'event': best_e.flatten()
        })

    # 시각화 (Plotting) - 2x3 Grid (1 Bar + 5 Scatter)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 모든 Fold의 데이터 범위 계산 (축 통일용)
    all_risks = np.concatenate([res['risk'] for res in fold_results])
    all_times = np.concatenate([res['time'] for res in fold_results])
    
    x_min, x_max = all_risks.min(), all_risks.max()
    y_min, y_max = all_times.min(), all_times.max()
    
    # 여백 추가
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    x_lim = (x_min - x_margin, x_max + x_margin)
    y_lim = (y_min - y_margin, y_max + y_margin)

    
    # Plot 1: Fold별 C-Index
    ax1 = axes[0]
    folds = range(1, k+1)
    bars = ax1.bar(folds, fold_c_indices, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Fold Number', fontsize=12)
    ax1.set_ylabel('C-Index', fontsize=12)
    ax1.set_title('Model Performance By Fold', fontsize=14, weight='bold')
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
                
    # Plot 2~6: 각 Fold별 Scatter Plot
    for i in range(k):
        ax = axes[i+1]
        result = fold_results[i]
        
        # 이벤트가 발생한 데이터만 필터링
        events_mask = result['event'] == 1
        risks = result['risk'][events_mask]
        times = result['time'][events_mask]
        
        if len(risks) > 0:
            scatter = ax.scatter(risks, times, c=times, cmap='viridis_r', alpha=0.6)
            
            if len(risks) > 1:
                z = np.polyfit(risks, times, 1)
                p = np.poly1d(z)
                ax.plot(risks, p(risks), "r--", linewidth=2, label='Trend')
                
                corr = np.corrcoef(risks, times)[0, 1]
                ax.text(0.05, 0.95, f'Corr: {corr:.2f}', transform=ax.transAxes, 
                        fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Predicted Risk', fontsize=10)
        ax.set_ylabel('Actual Cycles', fontsize=10)
        ax.set_title(f'Fold {i+1} : Risk vs Time', fontsize=12, weight='bold')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True, alpha=0.5)

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, "model_performance_metrics.png")
    plt.savefig(output_path)
    print(f"성능 그래프 저장 완료: {output_path}")

if __name__ == "__main__":
    run_analysis_and_plot()
