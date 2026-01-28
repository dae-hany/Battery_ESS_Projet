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

class TransferLearningExperiment:
    def __init__(self):
        self.model_dir = current_dir / "saved_models"
        self.meta_path = self.model_dir / "model_metadata.pkl"
        self.weights_path = self.model_dir / "deepsurv_final.pth"
        
        # Load Metadata
        if not self.meta_path.exists():
            raise FileNotFoundError("Metadata not found.")
        
        self.meta = joblib.load(self.meta_path)
        self.feature_cols = self.meta['feature_cols']
        self.src_mean = self.meta['scaler_mean']
        self.src_std = self.meta['scaler_std']
        
        # Setup Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _parse_frequency(self, col_name):
        name = col_name.lower().replace('hz', '')
        num_match = re.search(r'[\d\.]+', name)
        if not num_match: return None
        num = float(num_match.group(0))
        if 'k' in name: num *= 1000
        elif 'm' in name: num /= 1000
        return num

    def preprocess_nasa_data(self, csv_path):
        """NASA 데이터를 로드하고 Source Model의 피처 순서에 맞춰 매핑"""
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples from NASA dataset.")
        
        # Mapping Logic (Same as inference_nasa.py)
        nasa_cols = df.columns.tolist()
        nasa_r_cols = [c for c in nasa_cols if c.startswith('R_')]
        nasa_x_cols = [c for c in nasa_cols if c.startswith('X_')]
        
        nasa_r_map = {c: self._parse_frequency(c) for c in nasa_r_cols}
        nasa_x_map = {c: self._parse_frequency(c) for c in nasa_x_cols}
        
        X_list = []
        for _, row in df.iterrows():
            row_feats = []
            for model_col in self.feature_cols:
                target_freq = self._parse_frequency(model_col)
                if target_freq is None:
                    row_feats.append(0.0)
                    continue
                
                pool = nasa_r_map if model_col.startswith('R_') else nasa_x_map
                
                best_col = None
                min_diff = float('inf')
                for col, freq in pool.items():
                    if freq is None: continue
                    diff = abs(freq - target_freq)
                    if diff < min_diff:
                        min_diff = diff
                        best_col = col
                
                if best_col:
                    row_feats.append(row[best_col])
                else:
                    row_feats.append(0.0)
            X_list.append(row_feats)
            
        X = np.array(X_list).astype(np.float32)
        y = df['SOH'].values.astype(np.float32)
        
        # 중요: 전이 학습 시에는 Target Domain(NASA)의 통계로 다시 정규화하는 것이 일반적
        # 하지만 여기서는 데이터가 적다고 가정하므로, 
        # 1. Source Stat 사용 (Domain Adaptation 없음) -> Baseline
        # 2. Target Train Stat 사용 (Re-normalization) -> Better
        
        return X, y

    def run_experiment(self, nasa_csv_path, train_ratio=0.1, epochs=100, lr=0.001):
        print(f"\n=== Transfer Learning Experiment (Train Ratio: {train_ratio*100}%) ===")
        
        # 1. Data Prep
        X, y = self.preprocess_nasa_data(nasa_csv_path)
        
        # Split (Stratified if possible, but simple random here)
        # 10% for Fine-tunning, 90% for Testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42, shuffle=True)
        
        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        
        # Re-Normalize using TARGET TRAIN statistics (Crucial for Domain Shift)
        # Only use Train set stats to avoid leakage
        mean_ft = X_train.mean(axis=0)
        std_ft = X_train.std(axis=0) + 1e-7
        
        X_train_scaled = (X_train - mean_ft) / std_ft
        X_test_scaled = (X_test - mean_ft) / std_ft
        
        # 2. Load Pre-trained Model
        model = DeepSurv(in_features=len(self.feature_cols), hidden_layers=[64, 32], dropout=0.3)
        model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        model.to(self.device)
        
        # Freeze or Not? Let's try fine-tuning ALL layers first with small LR
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 3. Fine-tuning Loop
        model.train()
        train_loss_history = []
        
        # Convert to Tensor
        t_X_train = torch.tensor(X_train_scaled).to(self.device)
        t_y_train = torch.tensor(y_train).to(self.device)
        t_E_train = torch.ones_like(t_y_train) # All observed
        
        print("Fine-tuning...")
        for epoch in range(epochs):
            # Sort for Cox Loss
            sort_idx = torch.argsort(t_y_train, descending=True)
            batch_X = t_X_train[sort_idx]
            batch_y = t_y_train[sort_idx]
            batch_E = t_E_train[sort_idx]
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = cox_ph_loss(pred, batch_E)
            loss.backward()
            optimizer.step()
            
            train_loss_history.append(loss.item())
            if (epoch+1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
                
        # 4. Calibration (Re-fitting the linear mapper) on Train Set
        model.eval()
        with torch.no_grad():
            risk_train = model(t_X_train).cpu().numpy().flatten()
            
        # SOH = m * Risk + c
        A = np.vstack([risk_train, np.ones(len(risk_train))]).T
        m, c = np.linalg.lstsq(A, y_train, rcond=None)[0]
        print(f"New Calibration -> Slope: {m:.4f}, Intercept: {c:.4f}")
        
        # 5. Evaluation on Test Set
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
        plt.scatter(y_test, pred_soh, alpha=0.5, color='blue', label='Test Data')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal')
        plt.xlabel("True SOH (%)")
        plt.ylabel("Predicted SOH (%)")
        plt.title(f"Transfer Learning Results (10% Train)\nR2: {test_r2:.2f}, MAE: {test_mae:.2f}")
        plt.legend()
        plt.grid(True)
        
        save_path = current_dir / "transfer_learning_result.png"
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    experiment = TransferLearningExperiment()
    nasa_csv = r"C:\Users\User\daehan_study\Battery_ESS_Project_ECM_ANTIGRAVITY\nasa_battery_eis_28features.csv"
    experiment.run_experiment(nasa_csv, train_ratio=0.1, epochs=150, lr=0.001)
