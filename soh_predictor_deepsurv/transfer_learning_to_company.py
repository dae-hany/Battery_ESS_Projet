import torch
import torch.optim as optim
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 현재 디렉토리 모듈 임포트 설정
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model import DeepSurv, cox_ph_loss
from data_loader import BatteryEISDataset

class CompanyTransferExperiment:
    def __init__(self):
        self.model_dir = current_dir / "saved_models"
        self.meta_path = self.model_dir / "model_metadata.pkl"
        self.weights_path = self.model_dir / "deepsurv_final.pth"
        
        # Load Metadata
        if not self.meta_path.exists():
            raise FileNotFoundError("Metadata not found.")
        self.meta = joblib.load(self.meta_path)
        self.feature_cols = self.meta['feature_cols']
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_company_data(self):
        """BatteryEISDataset을 사용하여 Company Battery 데이터만 로드"""
        base_path = current_dir.parent / "datasets"
        
        # augment=False로 로드 (전이 학습용 데이터는 순수해야 함)
        # 중요: BatteryEISDataset 내부 로직을 사용하여 'company_battery' 데이터만 필터링
        ds = BatteryEISDataset(base_path, mode='test', augment=False)
        
        company_records = [r for r in ds.records if r.get('Source') == 'company_battery']
        
        if not company_records:
            raise ValueError("No Company Battery records found.")
            
        print(f"Loaded {len(company_records)} samples from Company Battery dataset.")
        
        # DataFrame 변환 및 피처 추출
        df = pd.DataFrame(company_records)
        
        # 결측치 처리 (0으로 채움)
        cols = [c for c in df.columns if c.startswith('R_') or c.startswith('X_')]
        df[cols] = df[cols].fillna(0)
        
        # 피처 정렬 (모델 학습 순서와 동일하게)
        X_list = []
        for _, row in df.iterrows():
            row_feats = []
            for col in self.feature_cols:
                val = row.get(col, 0.0) # 없으면 0.0
                row_feats.append(val)
            X_list.append(row_feats)
            
        X = np.array(X_list).astype(np.float32)
        y = df['SOH'].values.astype(np.float32)
        
        return X, y

    def run_experiment(self, train_ratio=0.2, epochs=100, lr=0.001):
        print(f"\n=== Company Battery Transfer Learning (Train Ratio: {train_ratio*100}%) ===")
        
        # 1. Data Load
        X, y = self.load_company_data()
        
        # SOH가 너무 한쪽에 쏠려있으면 Split이 어려울 수 있음
        # Stratified Split (binning)
        y_bins = pd.cut(y, bins=3, labels=False)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, stratify=y_bins, random_state=42)
        except ValueError:
            # 샘플 수가 너무 적으면 Stratify 실패 가능 -> Random Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
            
        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        
        # 2. Re-normalization (Domain Adaptation)
        mean_ft = X_train.mean(axis=0)
        std_ft = X_train.std(axis=0) + 1e-7
        
        X_train_scaled = (X_train - mean_ft) / std_ft
        X_test_scaled = (X_test - mean_ft) / std_ft
        
        # 3. Load Pre-trained Model
        model = DeepSurv(in_features=len(self.feature_cols), hidden_layers=[64, 32], dropout=0.3)
        model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 4. Fine-tuning
        model.train()
        t_X_train = torch.tensor(X_train_scaled).to(self.device)
        t_y_train = torch.tensor(y_train).to(self.device)
        t_E_train = torch.ones_like(t_y_train)
        
        print("Fine-tuning...")
        for epoch in range(epochs):
            sort_idx = torch.argsort(t_y_train, descending=True)
            bx = t_X_train[sort_idx]
            by = t_y_train[sort_idx]
            be = t_E_train[sort_idx]
            
            optimizer.zero_grad()
            pred = model(bx)
            
            # 데이터가 너무 적을 때(ex: batch size < 2) cox loss 계산 불가할 수 있음
            if len(bx) > 1:
                loss = cox_ph_loss(pred, be)
                loss.backward()
                optimizer.step()
            
            if (epoch+1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {loss.item() if len(bx)>1 else 0:.4f}")
                
        # 5. Re-Calibration
        model.eval()
        with torch.no_grad():
            risk_train = model(t_X_train).cpu().numpy().flatten()
            
        A = np.vstack([risk_train, np.ones(len(risk_train))]).T
        m, c = np.linalg.lstsq(A, y_train, rcond=None)[0]
        print(f"New Calibration -> Slope: {m:.4f}, Intercept: {c:.4f}")
        
        # 6. Evaluation
        t_X_test = torch.tensor(X_test_scaled).to(self.device)
        with torch.no_grad():
            risk_test = model(t_X_test).cpu().numpy().flatten()
            
        pred_soh = m * risk_test + c
        
        # Metrics
        test_mse = np.mean((y_test - pred_soh)**2)
        test_mae = np.mean(np.abs(y_test - pred_soh))
        test_r2 = 1 - (np.sum((y_test - pred_soh)**2) / np.sum((y_test - y_test.mean())**2))
        
        print(f"\n>>> Test Results (Fine-tuned):")
        print(f"  MAE : {test_mae:.4f}")
        print(f"  RMSE: {np.sqrt(test_mse):.4f}")
        print(f"  R2  : {test_r2:.4f}")
        
        # Visualization
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, pred_soh, alpha=0.6, color='green', label='Test Data')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal')
        plt.xlabel("True SOH (%)")
        plt.ylabel("Predicted SOH (%)")
        plt.title(f"Company Battery Transfer Learning (20% Train)\nR2: {test_r2:.2f}, MAE: {test_mae:.2f}")
        plt.legend()
        plt.grid(True)
        
        save_path = current_dir / "company_transfer_result.png"
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    exp = CompanyTransferExperiment()
    # Company 데이터는 양이 매우 적으므로 Train Ratio를 조금 높게(20%) 설정
    exp.run_experiment(train_ratio=0.2, epochs=150, lr=0.001)
