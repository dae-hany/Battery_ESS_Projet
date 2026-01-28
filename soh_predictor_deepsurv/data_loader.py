import pandas as pd
import numpy as np
from pathlib import Path
import re
import torch
from torch.utils.data import Dataset, DataLoader
import unicodedata

# --- Utils ---
def normalize_string(s: str) -> str:
    """유니코드 정규화"""
    return unicodedata.normalize('NFC', s)

def simplify_filename(filename: str) -> str:
    return normalize_string(filename).lower()

def extract_soh_from_filename(filename: str) -> float:
    """파일명에서 SOH 추출 (예: ..._SOH85.5.csv)"""
    match = re.search(r'SOH(\d+\.?\d*)', filename)
    if match:
        return float(match.group(1))
    return None

def extract_condition_soh(filename: str) -> float:
    """Company Battery 데이터의 조건 기반 SOH 매핑"""
    name = simplify_filename(filename)
    
    # Mapping Table
    if 'new' in name or '신품' in name:
        return 95.0
    if '21만' in name:
        return 75.0
    if 'bad' in name or '불량' in name:
        return 60.0
    
    return None

def extract_battery_id_simple(filename: str) -> str:
    """파일명에서 Battery ID 추출"""
    # ..._BATT1.csv or ...B01...
    match = re.search(r'BATT(\d+)', filename, re.IGNORECASE)
    if match:
        return f'B{match.group(1)}'
    
    # B01 format
    parts = filename.split('_')
    for p in parts:
        if re.match(r'^B\d+$', p):
            return p
    
    return 'unknown'

def normalize_frequency_column_name(freq_value: float) -> str:
    """주파수 값을 표준 컬럼명 접미사로 변환 (예: 1000.0 -> 1kHz)"""
    if freq_value >= 1000:
        return '1kHz' # Only supporting 1kHz based on existing convention? 
        # But if it's 2000Hz? let's stick to existing logic:
        # return f"{int(freq_value/1000)}kHz" # Better generalized but let's follow existing simple logic if possible.
        # Existing utils.py: return '1kHz' if >= 1000. It seems it normalizes everything >= 1k to 1k? Or just 1kHz is 1000.
        # Let's handle 1000 specifically.
        if abs(freq_value - 1000) < 10: return '1kHz'
        if abs(freq_value - 2000) < 10: return '2kHz'
    
    return f"{int(freq_value)}Hz"

def get_rx_columns(df: pd.DataFrame):
    cols = df.columns
    r_cols = [c for c in cols if c.startswith('R_')]
    x_cols = [c for c in cols if c.startswith('X_')]
    return r_cols, x_cols

# --- Dataset ---
class BatteryEISDataset(Dataset):
    def __init__(self, data_path, mode='train', test_split=0.2, augment=False):
        self.data_path = Path(data_path)
        self.records = []
        self.augment = augment and (mode == 'train')
        
        # 1. 데이터 로드 (Spectroscopy Only)
        print(f"Loading Spectroscopy Data...")
        self._load_spectroscopy()
        
        print(f"Loading Company Battery Data...")
        self._load_company_battery()
        
        print(f"Loading SoC Estimation Data (may take a while)...")
        self._load_soc_estimation()
        
        # 2. DataFrame 변환
        if not self.records:
            print("Error: No data loaded!")
            self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame(self.records)
            # 결측치 처리 (주파수가 비어있는 경우 0으로)
            # R_..., X_... 컬럼만 0으로 채우고 나머지는 유지
            cols = [c for c in self.df.columns if c.startswith('R_') or c.startswith('X_')]
            self.df[cols] = self.df[cols].fillna(0)
            
            # SOH가 없는 경우 제외 (DeepSurv 학습 불가)
            self.df = self.df.dropna(subset=['SOH'])
            
            print(f"Total Loaded Samples (Original): {len(self.df)}")
            
            # Augmentation 적용
            if self.augment:
                self._apply_augmentation(cols)
        
        # 3. Train/Test Split
        if not self.df.empty:
            battery_ids = self.df['BATTERY_ID'].unique()
            np.random.seed(42)
            
            # 최소한의 Battery ID가 있어야 split 가능
            if len(battery_ids) > 1:
                test_cnt = max(1, int(len(battery_ids) * test_split))
                test_ids = np.random.choice(battery_ids, size=test_cnt, replace=False)
            else:
                test_ids = []

            if mode == 'train':
                self.df = self.df[~self.df['BATTERY_ID'].isin(test_ids)]
            else:
                self.df = self.df[self.df['BATTERY_ID'].isin(test_ids)]
            
            # 피처 추출
            self.r_cols, self.x_cols = get_rx_columns(self.df)
            self.feature_cols = self.r_cols + self.x_cols
            
            if not self.feature_cols:
                print("Warning: No feature columns found!")
                self.X = np.array([])
                self.SOH = np.array([])
                self.Event = np.array([])
            else:
                self.X = self.df[self.feature_cols].values.astype(np.float32)
                self.SOH = self.df['SOH'].values.astype(np.float32)
                self.Event = np.ones_like(self.SOH) 


    def _apply_augmentation(self, feature_cols, factor=10, noise_std=0.01):
        """데이터 증강: Gaussian Noise 주입"""
        print(f"Applying Data Augmentation (Factor: {factor}, Noise: {noise_std})...")
        augmented_records = []
        
        for _, row in self.df.iterrows():
            # 원본 추가 (이미 records에 있지만 df를 새로 만드므로 여기서는 제외하거나 포함 정책 결정)
            # 현재 self.df는 이미 records로 만들어짐. 
            # 단순히 self.records에 추가하는 방식이 아니라 DataFrame을 확장해야 함.
            pass
            
        # 효율적인 처리를 위해 numpy 사용
        X_org = self.df[feature_cols].values
        y_org = self.df['SOH'].values
        ids_org = self.df['BATTERY_ID'].values
        
        new_dfs = [self.df] # 원본 포함
        
        for _ in range(factor):
            noise = np.random.normal(0, noise_std, size=X_org.shape)
            X_new = X_org + (X_org * noise) # Multiplicative Noise (Signal Dependent)
            
            df_new = pd.DataFrame(X_new, columns=feature_cols)
            df_new['SOH'] = y_org
            df_new['BATTERY_ID'] = ids_org
            
            new_dfs.append(df_new)
            
        self.df = pd.concat(new_dfs, ignore_index=True)
        print(f"Augmented Samples: {len(self.df)}")


    def _load_spectroscopy(self):
        spec_dir = self.data_path / "raw_data/Spectroscopy_Individual"
        if spec_dir.exists():
            for f in spec_dir.glob("*.csv"):
                rec = self._process_processed_file(f, source='spectroscopy')
                if rec: self.records.append(rec) 

    
    def _load_company_battery(self):
        comp_dir = self.data_path / "raw_data/company_battery_data"
        if comp_dir.exists():
            for f in comp_dir.glob("*.csv"):
                # SOH 추출 (Condition based)
                soh = extract_condition_soh(f.name)
                if soh is None: continue
                
                rec = self._process_processed_file(f, source='company_battery', soh_override=soh)
                if rec: self.records.append(rec)
                
    def _load_soc_estimation(self):
        soc_dir = self.data_path / "SoC Estimation on Li-ion Batteries A New EIS-based Dataset for data-driven applications"
        if not soc_dir.exists(): return
        
        # B01 ~ B...
        for batt_dir in soc_dir.iterdir():
            if not batt_dir.is_dir() or not batt_dir.name.startswith('B'): continue
            
            batt_id = batt_dir.name # e.g. B01
            
            # EIS measurements
            eis_dir = batt_dir / 'EIS measurements'
            if not eis_dir.exists(): continue
            
            # Test folders
            for test_dir in eis_dir.iterdir():
                if not test_dir.is_dir(): continue
                
                # Hioki
                hioki_dir = test_dir / 'Hioki'
                if not hioki_dir.exists(): continue
                
                # CSV Files
                for f in hioki_dir.glob("*.csv"):
                    rec = self._process_raw_file(f, batt_id)
                    if rec: self.records.append(rec)

    def _process_processed_file(self, file_path, source, soh_override=None):
        """이미 R_..., X_... 또는 유사한 형태로 된 파일 처리"""
        try:
            soh = soh_override if soh_override is not None else extract_soh_from_filename(file_path.name)
            if soh is None: return None
            
            df = pd.read_csv(file_path)
            
            # 컬럼 확인 및 정규화
            # 기존 코드는 get_rx_columns 로 그냥 가져감.
            # 데이터 컬럼이 R_1kHz 등으로 되어있다고 가정.
            # 만약 공백이 있으면 제거
            df.columns = df.columns.str.strip()
            
            r_c, x_c = get_rx_columns(df)
            cols = r_c + x_c
            
            if not cols: return None
            
            for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
            
            rec = df[cols].mean().to_dict()
            rec['SOH'] = soh
            rec['BATTERY_ID'] = extract_battery_id_simple(file_path.name)
            rec['Source'] = source
            return rec
        except:
            return None

    def _process_raw_file(self, file_path, batt_id):
        """Raw CSV (Frequency, R, X) 처리"""
        try:
            # SOH Label?
            # SoC Estimation 데이터셋은 Aging 정보가 없으면 SOH를 알 수 없음.
            # 여기서는 편의상 Pseudo Label (95.0) 부여.
            # 사용자가 DeepSurv로 학습하길 원하므로 라벨이 있어야 함.
            # 단, B01, B02 등은 Aging Test 결과일 수 있음 -> 하지만 개별 파일이 Cycle과 매핑되지 않으면 알 수 없음.
            # 기존 soh_predictor에서는 95.0으로 두고 추후 보정함.
            soh = 95.0 
            
            df = pd.read_csv(file_path)
            
            # 컬럼 찾기
            freq_col = None
            r_col = None
            x_col = None
            
            for c in df.columns:
                cl = c.lower()
                if 'freq' in cl: freq_col = c
                elif 'r(' in cl or c=='R' or 'real' in cl: r_col = c
                elif 'x(' in cl or c=='X' or 'imag' in cl: x_col = c
            
            if not (freq_col and r_col and x_col): return None
            
            # Pivot
            rec = {}
            for _, row in df.iterrows():
                f_val = row[freq_col]
                r_val = row[r_col]
                x_val = row[x_col]
                
                if pd.isna(f_val): continue
                
                f_str = normalize_frequency_column_name(f_val)
                rec[f'R_{f_str}'] = r_val
                rec[f'X_{f_str}'] = x_val
            
            if not rec: return None
            
            rec['SOH'] = soh
            rec['BATTERY_ID'] = batt_id
            rec['Source'] = 'soc_estimation'
            return rec
        except:
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X[idx]),
            't': torch.tensor(self.SOH[idx]), # Time = SOH
            'e': torch.tensor(self.Event[idx])
        }

def get_dataloader(data_path, batch_size=32, augment=True):
    train_ds = BatteryEISDataset(data_path, mode='train', augment=augment)
    test_ds = BatteryEISDataset(data_path, mode='test', augment=False)
    
    # MinMaxScaler (Train 기준)
    if len(train_ds) > 0:
        # 데이터가 없으면 feature_cols가 비어있을 수 있음
        if train_ds.X.size > 0:
            mean = train_ds.X.mean(axis=0)
            std = train_ds.X.std(axis=0) + 1e-7
            
            train_ds.X = (train_ds.X - mean) / std
            if len(test_ds) > 0 and test_ds.X.size > 0:
                test_ds.X = (test_ds.X - mean) / std
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_ds.feature_cols
