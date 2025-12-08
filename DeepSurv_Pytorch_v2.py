import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = r"c:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
METADATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset", "metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "cleaned_dataset", "data")

# Outlier Rejection Criteria
MIN_INIT_CAPACITY = 1.5
MIN_CYCLES = 50

# EOL Threshold (SOH)
SOH_LIMIT = 0.8

class BatteryDataset(Dataset):
    def __init__(self, metadata_path, data_dir, sequence_length=1):
        self.metadata_path = metadata_path
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.samples = [] # List of (X, T, E)
        
        self._load_and_process_data()
        
    def _load_and_process_data(self):
        print("Loading metadata...")
        df = pd.read_csv(self.metadata_path)
        
        # Filter for discharge cycles
        discharge_df = df[df['type'] == 'discharge'].copy()
        discharge_df['Capacity'] = pd.to_numeric(discharge_df['Capacity'], errors='coerce')
        discharge_df = discharge_df.dropna(subset=['Capacity'])
        
        unique_batteries = discharge_df['battery_id'].unique()
        print(f"Found {len(unique_batteries)} unique batteries.")
        
        valid_batteries = []
        
        for bat_id in unique_batteries:
            bat_df = discharge_df[discharge_df['battery_id'] == bat_id].sort_values('test_id')
            
            if len(bat_df) == 0:
                continue
                
            init_cap = bat_df['Capacity'].iloc[0]
            cycles = len(bat_df)
            
            # Outlier Check
            if init_cap < MIN_INIT_CAPACITY or cycles < MIN_CYCLES:
                continue
            
            # Smoothing Capacity
            bat_df['Capacity_Smooth'] = bat_df['Capacity'].rolling(window=5, min_periods=1).mean()
            bat_df['SOH'] = bat_df['Capacity_Smooth'] / init_cap
            
            # Determine EOL
            # Reset index first to ensure alignment
            bat_df = bat_df.reset_index(drop=True)
            
            eol_indices = bat_df.index[bat_df['SOH'] < SOH_LIMIT].tolist()
            if eol_indices:
                event = 1
                eol_cycle = eol_indices[0]
            else:
                event = 0
                eol_cycle = len(bat_df) - 1
            
            # Extract Features for each cycle
            # We'll use pre-extracted features from metadata if available, or load files
            # Metadata has: start_time, ambient_temperature, Capacity, Re, Rct
            # We need more dynamic features. Let's load files.
            # WARNING: Loading 3000 files is slow. We'll do it once and maybe save cache.
            
            bat_features = []
            print(f"Processing {bat_id} ({len(bat_df)} cycles)...")
            
            for i, row in bat_df.iterrows():
                # Simple features from metadata
                # We can add file loading here if needed, but for speed let's start with Capacity + Metadata
                # If we really need voltage/current features, we must load files.
                # Let's try to load files for a few batteries to see speed.
                
                filename = row['filename']
                file_path = os.path.join(self.data_dir, filename)
                
                try:
                    # Optimization: Only read needed columns or use a fast reader
                    # If too slow, we will skip file loading and use Capacity only for now
                    # But user asked for "Analyze covariates", so we should use them.
                    # Let's assume we load them.
                    cycle_data = pd.read_csv(file_path)
                    
                    discharge_time = cycle_data['Time'].max() - cycle_data['Time'].min()
                    max_temp = cycle_data['Temperature_measured'].max()
                    min_voltage = cycle_data['Voltage_measured'].min()
                    
                    bat_features.append([discharge_time, max_temp, min_voltage, row['Capacity_Smooth']])
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    bat_features.append([0, 0, 0, row['Capacity_Smooth']]) # Fallback
            
            bat_features = np.array(bat_features)
            
            # Normalize features (Simple MinMax or StandardScaler per battery? No, global or per train set)
            # For now, raw features.
            
            # Create Samples
            # X = features at cycle k
            # T = RUL = eol_cycle - k
            # E = event
            
            # Only take samples UP TO the event time
            max_k = eol_cycle if event == 1 else len(bat_df)
            
            for k in range(max_k):
                X = bat_features[k] # Shape (4,)
                T = eol_cycle - k
                E = event
                
                # If censored, T is just "time until end of observation"
                # But for CoxPH, T is "observed time".
                
                self.samples.append((X, T, E, bat_id))
                
            valid_batteries.append(bat_id)
            
        self.processed_battery_ids = valid_batteries
        print(f"Processed {len(valid_batteries)} batteries. Total samples: {len(self.samples)}")

    def get_battery_ids(self):
        # Return list of battery IDs that were processed
        # We need to store this during processing
        return self.processed_battery_ids

    def get_indices_for_batteries(self, battery_ids):
        indices = []
        for idx, (X, T, E, bat_id) in enumerate(self.samples):
            if bat_id in battery_ids:
                indices.append(idx)
        return indices

    def __getitem__(self, idx):
        X, T, E, bat_id = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(T, dtype=torch.float32), torch.tensor(E, dtype=torch.float32)

# DeepSurv Model
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
            
        layers.append(nn.Linear(in_dim, 1)) # Output: Log Hazard Ratio
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# CoxPH Loss
def cox_ph_loss(log_h, t, e):
    # log_h: (Batch, 1)
    # t: (Batch,)
    # e: (Batch,)
    
    # Sort by time (descending)
    idx = torch.argsort(t, descending=True)
    log_h = log_h[idx]
    t = t[idx]
    e = e[idx]
    
    # Risk set calculation
    # For each i, risk set is all j such that t_j >= t_i
    # Since sorted, risk set for i is j >= i
    
    exp_log_h = torch.exp(log_h)
    risk_sum = torch.cumsum(exp_log_h, dim=0) # This sums from 0 to i. We want sum from i to N.
    # So we need reverse cumsum
    risk_sum = torch.flip(torch.cumsum(torch.flip(exp_log_h, [0]), dim=0), [0])
    
    # Log likelihood
    # sum_i E_i * (log_h_i - log(sum_{j in R_i} exp(log_h_j)))
    
    log_risk = torch.log(risk_sum + 1e-8)
    
    loss = -torch.sum(e.view(-1, 1) * (log_h - log_risk)) / (torch.sum(e) + 1e-8)
    return loss

def c_index(risk_scores, t, e):
    # risk_scores: (N,)
    # t: (N,)
    # e: (N,)
    
    n = len(t)
    concordant = 0
    permissible = 0
    
    # Simple O(N^2) implementation
    for i in range(n):
        if e[i] == 1:
            for j in range(n):
                if t[i] < t[j]: # i died before j
                    permissible += 1
                    if risk_scores[i] > risk_scores[j]: # i had higher risk
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
                        
    return concordant / permissible if permissible > 0 else 0.0

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
            
        # Validation
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
            
        # print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f}, Val C-Index {ci:.4f}")
        
    return best_c_index

def run_kfold_cv(dataset, k=5, epochs=50):
    from sklearn.model_selection import KFold
    
    battery_ids = dataset.get_battery_ids()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    
    print(f"Starting {k}-Fold Cross-Validation on {len(battery_ids)} batteries...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(battery_ids)):
        train_bats = [battery_ids[i] for i in train_idx]
        val_bats = [battery_ids[i] for i in val_idx]
        
        train_indices = dataset.get_indices_for_batteries(train_bats)
        val_indices = dataset.get_indices_for_batteries(val_bats)
        
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=len(val_subset), shuffle=False)
        
        print(f"Fold {fold+1}: Train Bats {len(train_bats)}, Val Bats {len(val_bats)}")
        
        best_ci = train_model(train_loader, val_loader, input_size=4, epochs=epochs)
        print(f"Fold {fold+1} Best C-Index: {best_ci:.4f}")
        fold_results.append(best_ci)
        
    avg_ci = np.mean(fold_results)
    print(f"\nAverage C-Index: {avg_ci:.4f}")
    return avg_ci

if __name__ == "__main__":
    dataset = BatteryDataset(METADATA_PATH, DATA_DIR)
    run_kfold_cv(dataset, k=5, epochs=50)
