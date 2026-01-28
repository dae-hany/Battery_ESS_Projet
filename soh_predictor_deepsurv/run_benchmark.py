import torch
import torch.optim as optim
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 현재 디렉토리 모듈 임포트 설정
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model import DeepSurv, cox_ph_loss
from data_loader import BatteryEISDataset

class BenchmarkRunner:
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
        
        # Mapping Cache
        self.parsed_feature_cols = {col: self._parse_frequency(col) for col in self.feature_cols}

    def _parse_frequency(self, col_name):
        name = col_name.lower().replace('hz', '')
        num_match = re.search(r'[\d\.]+', name)
        if not num_match: return None
        num = float(num_match.group(0))
        if 'k' in name: num *= 1000
        elif 'm' in name: num /= 1000
        return num

    def _map_features(self, df):
        """Map dataset columns to model features based on nearest frequency"""
        dataset_cols = df.columns.tolist()
        r_cols = {c: self._parse_frequency(c) for c in dataset_cols if c.startswith('R') or 'real' in c.lower()}
        x_cols = {c: self._parse_frequency(c) for c in dataset_cols if c.startswith('X') or 'imag' in c.lower()}
        
        X_list = []
        for _, row in df.iterrows():
            row_feats = []
            for model_col, target_freq in self.parsed_feature_cols.items():
                if target_freq is None:
                    row_feats.append(0.0)
                    continue
                
                # Determine pool based on prefix
                pool = r_cols if model_col.startswith('R_') else x_cols
                
                best_col = None
                min_diff = float('inf')
                for col, freq in pool.items():
                    if freq is None: continue
                    diff = abs(freq - target_freq)
                    if diff < min_diff:
                        min_diff = diff
                        best_col = col
                
                # Threshold for matching? (e.g. within 10%) - Optional
                # For now, nearest neighbour
                if best_col is not None:
                    row_feats.append(row[best_col])
                else:
                    row_feats.append(0.0)
            X_list.append(row_feats)
            
        return np.array(X_list).astype(np.float32)

    def load_dataset(self, name):
        """Load specific dataset and return X, y"""
        print(f"\n[Data Load] Loading {name}...")
        
        if name == 'NASA':
            csv_path = r"C:\Users\User\daehan_study\Battery_ESS_Project_ECM_ANTIGRAVITY\nasa_battery_eis_28features.csv"
            df = pd.read_csv(csv_path)
            y = df['SOH'].values.astype(np.float32)
            X = self._map_features(df)
            
        elif name == 'Check_Company' or name == 'SoC_Estimation':
            target_source = 'company_battery' if name == 'Check_Company' else 'soc_estimation'
            base_path = current_dir.parent / "datasets"
            # Using data_loader with augment=False
            ds = BatteryEISDataset(base_path, mode='test', augment=False)
            
            records = [r for r in ds.records if r.get('Source') == target_source]
            if not records:
                print(f"Warning: No records found for {name}")
                return None, None
                
            df = pd.DataFrame(records)
            df = df.fillna(0) # 결측치 0으로 채움 (Sparse Data 대응)
            y = df['SOH'].values.astype(np.float32)
            
            # Company/SoC data is already processed by data_loader to have R_xxx format roughly
            # But let's re-map using robust logic just in case
            X = self._map_features(df)
            
        else:
            raise ValueError(f"Unknown dataset: {name}")
            
        # Remove NaNs
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        print(f"  > Loaded {len(X)} samples (after NaN removal).")
        return X, y

    def evaluate_zero_shot(self, X, y):
        """Evaluate using the original pre-trained model"""
        # Load Original Model
        model = DeepSurv(in_features=len(self.feature_cols), hidden_layers=[64, 32], dropout=0.0)
        model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        # Original Scaling Stats
        mean = self.meta['scaler_mean']
        std = self.meta['scaler_std']
        slope = self.meta['calibration_slope']
        intercept = self.meta['calibration_intercept']
        
        # Scaling
        X_scaled = (X - mean) / std
        
        # Inference
        with torch.no_grad():
            t_X = torch.tensor(X_scaled).to(self.device)
            risk = model(t_X).cpu().numpy().flatten()
            
        pred = slope * risk + intercept
        
        # Metrics
        return self._calc_metrics(y, pred)

    def evaluate_few_shot(self, X, y, train_ratio=0.2, epochs=100, lr=0.001):
        """Fine-tune on small subset and evaluate on the rest"""
        # Split
        if len(X) < 10: # Too small
             return self.evaluate_zero_shot(X, y) # Fallback
             
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)
        except:
            # If straity fails or something
            indices = np.random.permutation(len(X))
            split = int(len(X) * train_ratio)
            X_train, y_train = X[indices[:split]], y[indices[:split]]
            X_test, y_test = X[indices[split:]], y[indices[split:]]

        # Domain Adaptation Normalization (Train Stats)
        mean_ft = X_train.mean(axis=0)
        std_ft = X_train.std(axis=0) + 1e-7
        
        X_train_scaled = (X_train - mean_ft) / std_ft
        X_test_scaled = (X_test - mean_ft) / std_ft
        
        # Load Model
        model = DeepSurv(in_features=len(self.feature_cols), hidden_layers=[64, 32], dropout=0.3)
        model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Fine-tuning
        model.train()
        t_X_train = torch.tensor(X_train_scaled).to(self.device)
        t_y_train = torch.tensor(y_train).to(self.device)
        t_E_train = torch.ones_like(t_y_train)
        
        for _ in range(epochs):
            if len(t_y_train) < 2: break 
            
            sort_idx = torch.argsort(t_y_train, descending=True)
            bx, by, be = t_X_train[sort_idx], t_y_train[sort_idx], t_E_train[sort_idx]
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = cox_ph_loss(pred, be)
            loss.backward()
            optimizer.step()
            
        # Calibration
        model.eval()
        with torch.no_grad():
            risk_train = model(t_X_train).cpu().numpy().flatten()
            
        A = np.vstack([risk_train, np.ones(len(risk_train))]).T
        m, c = np.linalg.lstsq(A, y_train, rcond=None)[0]
        
        # Evaluation
        t_X_test = torch.tensor(X_test_scaled).to(self.device)
        with torch.no_grad():
            risk_test = model(t_X_test).cpu().numpy().flatten()
            
        pred = m * risk_test + c
        
        return self._calc_metrics(y_test, pred)

    def _calc_metrics(self, true, pred):
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'True': true, 'Pred': pred}

    def run_all(self):
        datasets = ['NASA']
        results = {}
        
        for name in datasets:
            X, y = self.load_dataset(name)
            if X is None or len(X) == 0: continue
            
            print(f"Evaluating {name}...")
            # Zero Shot
            res_zero = self.evaluate_zero_shot(X, y)
            print(f"  Zero-Shot -> R2: {res_zero['R2']:.4f}, MAE: {res_zero['MAE']:.4f}")
            
            # Few Shot
            res_few = self.evaluate_few_shot(X, y, train_ratio=0.2, epochs=100)
            print(f"  Few-Shot  -> R2: {res_few['R2']:.4f}, MAE: {res_few['MAE']:.4f}")
            
            results[name] = {'Zero': res_zero, 'Few': res_few}
            
        self.plot_summary(results)
            
    def plot_summary(self, results):
        # 1. Bar Chart (Metric별 분리)
        metrics = ['R2', 'MAE', 'RMSE']
        data = []
        
        for name, res in results.items():
            for m in metrics:
                data.append({'Dataset': name, 'Method': 'Zero-Shot', 'Metric': m, 'Value': res['Zero'][m]})
                data.append({'Dataset': name, 'Method': 'Few-Shot', 'Metric': m, 'Value': res['Few'][m]})
                
        df_res = pd.DataFrame(data)
        
        # Plot 3 Metrics side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.barplot(data=df_res[df_res['Metric'] == metric], x='Dataset', y='Value', hue='Method', ax=ax, ci=None)
            ax.set_title(f"Performance Comparison - {metric}")
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        plt.savefig(current_dir / 'benchmark_summary_bar.png')
        
        # 2. Scatter Plots (One per dataset)
        fig, axes = plt.subplots(len(results), 2, figsize=(12, 5 * len(results)))
        if len(results) == 1: axes = np.array([axes]) # handle single row
        
        for i, (name, res) in enumerate(results.items()):
            # Zero Shot
            ax0 = axes[i, 0]
            true, pred = res['Zero']['True'], res['Zero']['Pred']
            ax0.scatter(true, pred, alpha=0.5)
            ax0.plot([min(true), max(true)], [min(true), max(true)], 'r--')
            ax0.set_title(f"{name} - Zero Shot\nR2: {res['Zero']['R2']:.2f}")
            ax0.set_xlabel('True SOH')
            ax0.set_ylabel('Pred SOH')
            
            # Few Shot
            ax1 = axes[i, 1]
            true, pred = res['Few']['True'], res['Few']['Pred']
            ax1.scatter(true, pred, alpha=0.5, color='orange')
            ax1.plot([min(true), max(true)], [min(true), max(true)], 'r--')
            ax1.set_title(f"{name} - Few Shot (20%)\nR2: {res['Few']['R2']:.2f}")
            ax1.set_xlabel('True SOH')
            
        plt.tight_layout()
        plt.savefig(current_dir / 'benchmark_scatters.png')
        print("Comparison plots saved.")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_all()
