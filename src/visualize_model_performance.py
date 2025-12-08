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
    all_val_risks = []
    all_val_times = []
    all_val_events = []
    
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
        
        # 최고 성능 Epoch의 예측값 수집
        all_val_risks.extend(best_risk.flatten())
        all_val_times.extend(best_t.flatten())
        all_val_events.extend(best_e.flatten())

    # 시각화 (Plotting)
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Fold별 C-Index
    ax1 = plt.subplot(1, 2, 1)
    folds = range(1, k+1)
    bars = ax1.bar(folds, fold_c_indices, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Fold Number', fontsize=12)
    ax1.set_ylabel('C-Index', fontsize=12)
    ax1.set_title('Fold별 모델 성능 (Model Performance by Fold)', fontsize=14, weight='bold')
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 막대 위에 값 표시
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
                
    # Plot 2: 위험 점수 vs 실제 생존 시간 (예측력)
    ax2 = plt.subplot(1, 2, 2)
    
    # 이벤트가 발생한(Uncensored) 데이터만 필터링하여 상관관계 분석
    events_mask = np.array(all_val_events) == 1
    risks = np.array(all_val_risks)[events_mask]
    times = np.array(all_val_times)[events_mask]
    
    # 산점도 (Scatter Plot)
    scatter = ax2.scatter(risks, times, c=times, cmap='viridis_r', alpha=0.6)
    
    # 추세선 (Trend Line)
    z = np.polyfit(risks, times, 1)
    p = np.poly1d(z)
    ax2.plot(risks, p(risks), "r--", linewidth=2, label='Trend')
    
    ax2.set_xlabel('예측된 위험 점수 (Predicted Risk Score)', fontsize=12)
    ax2.set_ylabel('실제 남은 수명 (Actual Remaining Cycles)', fontsize=12)
    ax2.set_title('예측력: 위험도 vs 생존 시간 (Risk vs Time)', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.5)
    ax2.legend()
    
    # 상관계수 표시
    corr = np.corrcoef(risks, times)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax2.transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, "model_performance_metrics.png")
    plt.savefig(output_path)
    print(f"성능 그래프 저장 완료: {output_path}")

if __name__ == "__main__":
    run_analysis_and_plot()
