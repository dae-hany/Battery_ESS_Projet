import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

# ==========================================
# 설정 (Configuration)
# ==========================================
# 프로젝트 루트 경로 설정 (현재 스크립트의 상위 폴더를 기준으로 함)
# src 폴더 안에 이 스크립트가 위치하므로, 부모의 부모가 프로젝트 루트가 됨
# 또는 절대 경로를 그대로 사용
BASE_DIR = r"c:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
METADATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset", "metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "cleaned_dataset", "data")

# 이상치 제거 기준 (Outlier Rejection Criteria)
MIN_INIT_CAPACITY = 1.5  # 최소 초기 용량 (Ah)
MIN_CYCLES = 50          # 최소 사이클 수

# 수명 종료 임계값 (EOL Threshold, SOH)
SOH_LIMIT = 0.8          # SOH 80% 미만 시 고장으로 간주

# ==========================================
# 데이터셋 클래스 (Dataset Class)
# ==========================================
class BatteryDataset(Dataset):
    """
    배터리 데이터를 로드하고 전처리하여 PyTorch Dataset으로 변환하는 클래스
    """
    def __init__(self, metadata_path, data_dir):
        self.metadata_path = metadata_path
        self.data_dir = data_dir
        self.samples = [] # (X, T, E, bat_id) 튜플을 저장할 리스트
        self.processed_battery_ids = [] # 처리된 배터리 ID 목록
        
        self._load_and_process_data()
        
    def _load_and_process_data(self):
        print("메타데이터 로딩 중...")
        df = pd.read_csv(self.metadata_path)
        
        # 방전(discharge) 사이클만 필터링
        discharge_df = df[df['type'] == 'discharge'].copy()
        discharge_df['Capacity'] = pd.to_numeric(discharge_df['Capacity'], errors='coerce')
        discharge_df = discharge_df.dropna(subset=['Capacity'])
        
        unique_batteries = discharge_df['battery_id'].unique()
        print(f"총 {len(unique_batteries)}개의 배터리 식별됨.")
        
        valid_batteries = []
        
        for bat_id in unique_batteries:
            bat_df = discharge_df[discharge_df['battery_id'] == bat_id].sort_values('test_id')
            
            if len(bat_df) == 0:
                continue
                
            init_cap = bat_df['Capacity'].iloc[0]
            cycles = len(bat_df)
            
            # 이상치 제거 (Outlier Check)
            if init_cap < MIN_INIT_CAPACITY or cycles < MIN_CYCLES:
                continue
            
            # 용량 스무딩 (Smoothing Capacity)
            # 이동 평균을 사용하여 노이즈 제거
            bat_df['Capacity_Smooth'] = bat_df['Capacity'].rolling(window=5, min_periods=1).mean()
            bat_df['SOH'] = bat_df['Capacity_Smooth'] / init_cap
            
            # EOL(수명 종료) 지점 결정
            bat_df = bat_df.reset_index(drop=True)
            
            eol_indices = bat_df.index[bat_df['SOH'] < SOH_LIMIT].tolist()
            if eol_indices:
                event = 1        # 고장 발생 (Uncensored)
                eol_cycle = eol_indices[0]
            else:
                event = 0        # 중도 절단 (Censored)
                eol_cycle = len(bat_df) - 1
            
            # 피처 추출 (Feature Extraction)
            bat_features = []
            # print(f"{bat_id} 처리 중 ({len(bat_df)} cycles)...") 
            
            for i, row in bat_df.iterrows():
                filename = row['filename']
                file_path = os.path.join(self.data_dir, filename)
                
                try:
                    # 개별 사이클 파일 로드 (전압, 온도 등 추출)
                    # 속도를 위해 필요한 컬럼만 읽거나 최적화 가능
                    cycle_data = pd.read_csv(file_path)
                    
                    discharge_time = cycle_data['Time'].max() - cycle_data['Time'].min()
                    max_temp = cycle_data['Temperature_measured'].max()
                    min_voltage = cycle_data['Voltage_measured'].min()
                    
                    bat_features.append([discharge_time, max_temp, min_voltage, row['Capacity_Smooth']])
                except Exception as e:
                    print(f"{filename} 읽기 오류: {e}")
                    bat_features.append([0, 0, 0, row['Capacity_Smooth']]) # 오류 시 기본값 처리
            
            bat_features = np.array(bat_features)
            
            # 샘플 생성 (Create Samples)
            # X: 현재 사이클의 피처
            # T: 남은 수명 (RUL = eol_cycle - current_cycle)
            # E: 이벤트 발생 여부
            
            # 이벤트 발생 시점까지만 데이터 사용
            max_k = eol_cycle if event == 1 else len(bat_df)
            
            for k in range(max_k):
                X = bat_features[k] # Shape (4,)
                T = eol_cycle - k
                E = event
                
                self.samples.append((X, T, E, bat_id))
                
            valid_batteries.append(bat_id)
            
        self.processed_battery_ids = valid_batteries
        print(f"{len(valid_batteries)}개 배터리 처리 완료. 총 샘플 수: {len(self.samples)}")

    def get_battery_ids(self):
        """처리된 배터리 ID 목록 반환"""
        return self.processed_battery_ids

    def get_indices_for_batteries(self, battery_ids):
        """특정 배터리 ID들에 해당하는 샘플 인덱스 반환 (K-Fold용)"""
        indices = []
        for idx, (X, T, E, bat_id) in enumerate(self.samples):
            if bat_id in battery_ids:
                indices.append(idx)
        return indices

    def __getitem__(self, idx):
        X, T, E, bat_id = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(T, dtype=torch.float32), torch.tensor(E, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

# ==========================================
# DeepSurv 모델 (Model Architecture)
# ==========================================
class DeepSurv(nn.Module):
    def __init__(self, input_size, hidden_layers=[64, 32], dropout=0.1):
        super(DeepSurv, self).__init__()
        layers = []
        in_dim = input_size
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1)) # 출력: 로그 위험률 (Log Hazard Ratio)
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 손실 함수 (CoxPH Loss)
# ==========================================
def cox_ph_loss(log_h, t, e):
    """
    Cox Proportional Hazards Loss (Negative Log Partial Likelihood)
    """
    # 시간(t) 기준 내림차순 정렬
    idx = torch.argsort(t, descending=True)
    log_h = log_h[idx]
    t = t[idx]
    e = e[idx]
    
    # Risk Set 계산
    exp_log_h = torch.exp(log_h)
    # 누적 합 (Reverse Cumulative Sum)
    risk_sum = torch.flip(torch.cumsum(torch.flip(exp_log_h, [0]), dim=0), [0])
    
    # 로그 우도 계산
    log_risk = torch.log(risk_sum + 1e-8)
    
    loss = -torch.sum(e.view(-1, 1) * (log_h - log_risk)) / (torch.sum(e) + 1e-8)
    return loss

# ==========================================
# 평가 지표 (C-Index)
# ==========================================
def c_index(risk_scores, t, e):
    """
    Concordance Index 계산
    """
    n = len(t)
    concordant = 0
    permissible = 0
    
    for i in range(n):
        if e[i] == 1:
            for j in range(n):
                if t[i] < t[j]: # i가 j보다 먼저 고장남
                    permissible += 1
                    if risk_scores[i] > risk_scores[j]: # 모델도 i의 위험도가 더 높다고 예측함
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
                        
    return concordant / permissible if permissible > 0 else 0.0

# ==========================================
# 학습 함수 (Training Function)
# ==========================================
def train_model(train_loader, val_loader, input_size, epochs=50, lr=0.001):
    model = DeepSurv(input_size=input_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_c_index = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, t, e in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = cox_ph_loss(out, t, e)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
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
        if ci > best_c_index:
            best_c_index = ci
            
    return best_c_index

# ==========================================
# K-Fold 교차 검증 (Cross-Validation)
# ==========================================
def run_kfold_cv(dataset, k=5, epochs=50):
    from sklearn.model_selection import KFold
    
    battery_ids = dataset.get_battery_ids()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    
    print(f"\n{len(battery_ids)}개 배터리에 대해 {k}-Fold 교차 검증 시작...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(battery_ids)):
        train_bats = [battery_ids[i] for i in train_idx]
        val_bats = [battery_ids[i] for i in val_idx]
        
        train_indices = dataset.get_indices_for_batteries(train_bats)
        val_indices = dataset.get_indices_for_batteries(val_bats)
        
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)
        
        print(f"Fold {fold+1}: 학습 배터리 {len(train_bats)}개, 검증 배터리 {len(val_bats)}개")
        
        best_ci = train_model(train_loader, val_loader, input_size=4, epochs=epochs)
        print(f"Fold {fold+1} 최고 C-Index: {best_ci:.4f}")
        fold_results.append(best_ci)
        
    avg_ci = np.mean(fold_results)
    print(f"\n평균 C-Index: {avg_ci:.4f}")
    return avg_ci

if __name__ == "__main__":
    dataset = BatteryDataset(METADATA_PATH, DATA_DIR)
    run_kfold_cv(dataset, k=5, epochs=50)
