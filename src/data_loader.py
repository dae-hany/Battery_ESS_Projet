# -*- coding: utf-8 -*-
"""
데이터 로더 및 전처리 모듈 (Data Loader & Preprocessing Module)
===========================================================

이 모듈은 NASA 배터리 데이터셋을 로드하고, LSTM 모델 학습을 위한 
Sliding Window 데이터셋을 생성하는 기능을 제공합니다.

주요 기능:
1. 배터리 데이터 전처리 (이상치 제거, 스무딩) - BatteryDataProcessor
2. Sliding Window 시계열 데이터 생성
3. PyTorch Dataset 및 DataLoader 구현

Author: ESS DeepSurv Research Team
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

# ===================================================================================
# Configuration
# ===================================================================================

from src.features.path_signature import extract_path_signature
from src.features.delta_q import extract_delta_q, interpolate_qv

@dataclass
class DataConfig:
    """데이터 처리 관련 설정 클래스"""
    # 기본 경로 설정 (사용자 환경에 맞게 수정 필요 시 main에서 주입)
    base_dir: str = r"C:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
    metadata_path: str = None
    data_dir: str = None
    cache_dir: str = None # 캐시 디렉토리
    
    # 데이터 필터링 파라미터
    min_init_capacity: float = 1.5  # 최소 초기 용량 (Ah)
    min_cycles: int = 50            # 최소 사이클 수
    soh_threshold: float = 0.8      # EOL(수명 종료) 기준 SOH
    
    # 데이터셋 파라미터
    window_size: int = 10           # LSTM 입력 시퀀스 길이 (Sliding Window)
    test_size: float = 0.2          # 테스트 데이터 비율
    random_seed: int = 42           # 재현성을 위한 시드
    
    # 피처 설정
    feature_set_name: str = "advanced"  # 'basic', 'dynamic', 'full', 'advanced', 'delta_q', 'all' 
    
    def __post_init__(self):
        # ... (same) ...
        if self.metadata_path is None:
            self.metadata_path = os.path.join(self.base_dir, "cleaned_dataset", "metadata.csv")
        if self.data_dir is None:
            self.data_dir = os.path.join(self.base_dir, "cleaned_dataset", "data")
        if self.cache_dir is None:
            self.cache_dir = os.path.join(self.base_dir, "cache")
            os.makedirs(self.cache_dir, exist_ok=True)

# ===================================================================================
# PyTorch Dataset
# ===================================================================================

class BatteryDataset(Dataset):
    """
    LSTM-DeepSurv 모델을 위한 PyTorch Dataset.
    
    Args:
        features (torch.Tensor): 입력 특성 (Batch, Window, Features)
        times (torch.Tensor): 생존 시간/RUL (Batch,) + Window 시점 기준
        events (torch.Tensor): 이벤트 발생 여부 (Batch,)
    """
    def __init__(self, features: torch.Tensor, times: torch.Tensor, events: torch.Tensor):
        self.features = features
        self.times = times
        self.events = events
        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        # (Input, Time, Event) 반환
        return self.features[idx], self.times[idx], self.events[idx]

class BatteryDataProcessor:
    """배터리 데이터 전처리 및 Sliding Window 데이터셋 생성 클래스"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        self.sig_dim = 0
        self.delta_q_dim = 6 # var, min, mean, skew, kurt, range
        self.v_grid = np.linspace(2.5, 4.2, 100) # Common Voltage Grid for NMC/LCO
        
    def _get_feature_names(self, feature_set_name: str) -> List[str]:
        """피처 세트 이름 반환"""
        basic_features = ['capacity', 'soh']
        dynamic_features = ['soh_derivative', 'capacity_derivative']
        stat_features = ['capacity_rolling_std']
        
        # Signature features
        sig_features = [f'sig_{i}' for i in range(self.sig_dim)] if self.sig_dim > 0 else []
        
        # Delta Q features
        delta_q_features = ['dq_var', 'dq_min', 'dq_mean', 'dq_skew', 'dq_kurt', 'dq_range']

        if feature_set_name == 'basic':
            return basic_features
        elif feature_set_name == 'dynamic':
            return basic_features + dynamic_features
        elif feature_set_name == 'full':
            return basic_features + dynamic_features + stat_features
        elif feature_set_name == 'advanced':
            # Path Signature only
            return basic_features + dynamic_features + stat_features + sig_features
        elif feature_set_name == 'delta_q':
            # Delta Q only (on top of dynamic)
            return basic_features + dynamic_features + stat_features + delta_q_features
        elif feature_set_name == 'all':
            # Everything
            return basic_features + dynamic_features + stat_features + delta_q_features + sig_features
        else:
            raise ValueError(f"Unknown feature_set: {feature_set_name}")

    def _preprocess_capacity(self, capacities: np.ndarray, outlier_threshold: float = 0.5) -> np.ndarray:
        """
        용량 데이터 전처리: 이상치 제거 및 측정 오류 보정.
        """
        cleaned = capacities.copy()
        
        for i in range(1, len(cleaned)):
            prev_val = cleaned[i-1]
            curr_val = cleaned[i]
            
            if prev_val > 0:
                drop_ratio = (prev_val - curr_val) / prev_val
                if drop_ratio > outlier_threshold:
                    cleaned[i] = prev_val
                if curr_val < 0.1:
                    cleaned[i] = prev_val
                    
        # Median Filter
        if len(cleaned) >= 3:
            cleaned = medfilt(cleaned, kernel_size=3)
            
        return cleaned

    def _process_battery(self, bat_df: pd.DataFrame, bat_id: str, init_cap: float) -> pd.DataFrame:
        """
        개별 배터리에 대한 피처 엔지니어링 수행.
        """
        bat_df = bat_df.copy()
        raw_capacity = bat_df['capacity'].values.copy()
        
        # Preprocess
        cleaned_capacity = self._preprocess_capacity(raw_capacity)
        bat_df['capacity'] = cleaned_capacity
        
        # Smoothing
        bat_df['capacity_smooth'] = bat_df['capacity'].rolling(window=7, min_periods=1, center=True).mean()
        bat_df['capacity_smooth'] = bat_df['capacity_smooth'].fillna(bat_df['capacity'])
        
        # SOH
        bat_df['soh'] = bat_df['capacity_smooth'] / init_cap
        
        # Derivatives
        bat_df['soh_derivative'] = bat_df['soh'].diff().fillna(0)
        bat_df['capacity_derivative'] = bat_df['capacity'].diff().fillna(0)
        bat_df['capacity_rolling_std'] = bat_df['capacity'].rolling(window=10, min_periods=1).std().fillna(0)
        
        # EOL & Event
        eol_indices = bat_df.index[bat_df['soh'] < self.config.soh_threshold].tolist()
        
        if eol_indices:
            event = 1 
            eol_cycle = eol_indices[0]
        else:
            event = 0 
            eol_cycle = len(bat_df) - 1
            
        bat_df['event'] = event
        bat_df['eol_cycle'] = eol_cycle
        
        # RUL
        bat_df['time'] = (eol_cycle - bat_df.index).astype(float)
        
        return bat_df

    def _extract_cycle_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper to load V, I, T from file"""
        file_path = os.path.join(self.config.data_dir, filename)
        if not os.path.exists(file_path):
            return None, None, None
            
        try:
            df = pd.read_csv(file_path)
            cols = df.columns
            v_col = next((c for c in cols if 'voltage' in c.lower()), None)
            i_col = next((c for c in cols if 'current' in c.lower()), None)
            t_col = next((c for c in cols if 'time' in c.lower()), None)
            
            if not (v_col and i_col and t_col):
                return None, None, None
                
            return df[v_col].values, df[i_col].values, df[t_col].values
        except:
            return None, None, None

    def load_and_process_features(self, use_cache: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """메타데이터 로드 및 모든 배터리의 피처를 포함한 DataFrame 생성"""
        
        cache_path = os.path.join(self.config.cache_dir, f"features_{self.config.feature_set_name}.pkl")
        
        if use_cache and os.path.exists(cache_path):
            print(f"[Info] Loading features from cache: {cache_path}")
            final_df = pd.read_pickle(cache_path)
            
            # Restore dims
            sig_cols = [c for c in final_df.columns if c.startswith('sig_')]
            self.sig_dim = len(sig_cols)
            self.feature_names = self._get_feature_names(self.config.feature_set_name)
            return final_df, self.feature_names

        # ... (Loading metadata same as before) ...
        print("[Info] Loading metadata...")
        if not os.path.exists(self.config.metadata_path):
             raise FileNotFoundError(f"Metadata file not found: {self.config.metadata_path}")

        metadata = pd.read_csv(self.config.metadata_path)
        metadata.columns = [c.strip().lower() for c in metadata.columns]
        
        discharge_df = metadata[metadata['type'] == 'discharge'].copy()
        discharge_df['capacity'] = pd.to_numeric(discharge_df['capacity'], errors='coerce')
        discharge_df = discharge_df.dropna(subset=['capacity'])
        
        processed_data = []
        
        print("[Info] Processing batteries...")
        for bat_id in discharge_df['battery_id'].unique():
            bat_df = discharge_df[discharge_df['battery_id'] == bat_id].copy()
            bat_df = bat_df.sort_values('test_id').reset_index(drop=True)
            
            if len(bat_df) < self.config.min_cycles:
                continue
            init_cap = bat_df['capacity'].iloc[0]
            if init_cap < self.config.min_init_capacity:
                continue
                
            # Basic Processing
            df_processed = self._process_battery(bat_df, bat_id, init_cap)
            
            # --- Advanced Features ---
            need_sig = self.config.feature_set_name in ['advanced', 'all']
            need_delta_q = self.config.feature_set_name in ['delta_q', 'all']
            
            if need_sig or need_delta_q:
                print(f"  > Extracting advanced features for {bat_id}...", end='\r')
                
                signatures = []
                delta_qs = []
                
                # Reference Cycle for Delta Q (Cycle 1)
                ref_q_interp = None
                
                for idx, row in df_processed.iterrows():
                    filename = row['filename']
                    v, i, t = self._extract_cycle_data(filename)
                    
                    if v is None: # Load failed
                        if need_sig: signatures.append(np.zeros(12)) # Fallback assumption
                        if need_delta_q: delta_qs.append(np.zeros(6))
                        continue
                        
                    # 1. Path Signature
                    if need_sig:
                        sig = extract_path_signature(v, i, t, level=2)
                        signatures.append(sig)
                        
                    # 2. Delta Q(V)
                    if need_delta_q:
                        # Need capacity curve (Ah). Usually calculated by integrating Current over Time.
                        # Ah = cumtrapz(I, t) / 3600.
                        # Check units: I in Amps, t in Seconds -> Ah.
                        # Assuming I is discharge current (positive or negative). 
                        # Usually discharge current is positive in dataset or negative?
                        # Nasa dataset: Discharge current is usually positive load, but stored as negative?
                        # Let's compute Q(t) = Integ |I| dt.
                        
                        # Simple integration
                        # Use abs(I) to be safe for capacity throughput
                        q_t = np.cumsum(np.abs(i)) * (t[1] - t[0]) / 3600.0 if len(t) > 1 else np.zeros_like(v)
                        # Normalize Q to start at 0? Yes.
                        # But wait, Delta Q(V) is Q(V). 
                        # We use the raw Q values vs V.
                        
                        # Reference Setup
                        if idx == 0:
                            ref_q_interp = interpolate_qv(v, q_t, self.v_grid)
                            
                        # Extract Features
                        if ref_q_interp is not None:
                            dq_feats = extract_delta_q(v, q_t, ref_q_interp, self.v_grid)
                        else:
                            dq_feats = np.zeros(6)
                            
                        delta_qs.append(dq_feats)

                # Attach to DataFrame
                if need_sig and signatures:
                    sig_matrix = np.vstack(signatures)
                    self.sig_dim = sig_matrix.shape[1]
                    for k in range(self.sig_dim):
                        df_processed[f'sig_{k}'] = sig_matrix[:, k]
                        
                if need_delta_q and delta_qs:
                    dq_matrix = np.vstack(delta_qs)
                    km = ['dq_var', 'dq_min', 'dq_mean', 'dq_skew', 'dq_kurt', 'dq_range']
                    for k, name in enumerate(km):
                        df_processed[name] = dq_matrix[:, k]

            df_processed['battery_id'] = bat_id
            processed_data.append(df_processed)
            
        final_df = pd.concat(processed_data, ignore_index=True)
        self.feature_names = self._get_feature_names(self.config.feature_set_name)
        
        if use_cache:
            final_df.to_pickle(cache_path)
            print(f"[Info] Features cached to {cache_path}")
        
        print(f"[Info] Preprocessing complete. Total samples: {len(final_df)}")
        return final_df, self.feature_names

    def create_sliding_window_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        DataFrame을 (Batch, Window, Features) 형태의 Sliding Window 데이터로 변환.
        
        Args:
            df (pd.DataFrame): 전처리된 데이터프레임
            
        Returns:
            Dict: {'X': tensor, 'T': tensor, 'E': tensor, 'battery_ids': list}
        """
        print(f"[Info] Creating sliding windows (Size={self.config.window_size})...")
        
        X_list = []
        T_list = []
        E_list = []
        Bat_ids = []
        
        window_size = self.config.window_size
        feature_cols = self.feature_names
        
        # 배터리별로 그룹화하여 윈도우 생성
        for bat_id, group in df.groupby('battery_id'):
            group = group.sort_values('test_id').reset_index(drop=True) # 순서 보장
            
            features = group[feature_cols].values
            times = group['time'].values
            events = group['event'].values
            
            num_samples = len(group)
            
            # 윈도우를 만들 수 없는 경우 스킵
            if num_samples < window_size:
                continue
                
            # Sliding Window 생성
            # i: 윈도우의 시작 인덱스
            # 윈도우: [i : i + window_size]
            # 타겟 시점: i + window_size - 1 (윈도우의 마지막 시점)
            for i in range(num_samples - window_size + 1):
                window_end_idx = i + window_size
                
                # EOL 이후의 데이터는 제외 (RUL <= 0 인 경우)
                # 타겟 시점의 RUL 확인
                target_rul = times[window_end_idx - 1]
                if target_rul < 0:
                    continue
                
                # Window 데이터 추출
                x_window = features[i : window_end_idx]
                
                # 타겟 데이터 추출 (윈도우 마지막 시점 기준)
                t_target = target_rul
                e_target = events[window_end_idx - 1] # 해당 배터리의 이벤트 상태는 동일하거나 변할 수 있음 (여기선 고정값 사용되지만 row기반 추출이 안전)
                
                X_list.append(x_window)
                T_list.append(t_target)
                E_list.append(e_target)
                Bat_ids.append(bat_id)
                
        # Numpy Array 변환
        X = np.array(X_list, dtype=np.float32)
        T = np.array(T_list, dtype=np.float32)
        E = np.array(E_list, dtype=np.float32)
        
        print(f"[Info] Sliding window generation complete.")
        print(f"       Shape: X={X.shape}, T={T.shape}, E={E.shape}")
        
        return {
            'X': X,
            'T': T,
            'E': E,
            'battery_ids': np.array(Bat_ids)
        }

    def prepare_datasets(self, df: pd.DataFrame) -> Tuple['BatteryDataset', 'BatteryDataset']:
        """
        전체 파이프라인 실행: 열차/테스트 분할 -> 스케일링 -> 윈도우 생성 -> Dataset 반환
        """
        # 1. Train/Test Battery Split
        battery_ids = df['battery_id'].unique()
        train_bats, test_bats = train_test_split(
            battery_ids, test_size=self.config.test_size, random_state=self.config.random_seed
        )
        
        print(f"[Info] Splitting batteries: {len(train_bats)} Train, {len(test_bats)} Test")
        
        train_df = df[df['battery_id'].isin(train_bats)].copy()
        test_df = df[df['battery_id'].isin(test_bats)].copy()
        
        # 2. Scaling (Fit on Train, Transform on Test)
        # 스케일링을 위해 Flatten -> Scale -> Reshape 방식 대신, 
        # Feature별로 먼저 스케일링하고 윈도우를 자르는 것이 효율적임.
        
        # 모든 피처를 하나의 행렬로 만들어 스케일러 학습 (Train only)
        # 주의: Sliding Window가 아니므로 각 행이 독립적 데이터 포인트임
        X_train_raw = train_df[self.feature_names].values
        self.scaler.fit(X_train_raw)
        
        # Train 데이터 변환
        train_df.loc[:, self.feature_names] = self.scaler.transform(train_df[self.feature_names].values)
        
        # Test 데이터 변환
        test_df.loc[:, self.feature_names] = self.scaler.transform(test_df[self.feature_names].values)
        
        # 3. Sliding Window 적용
        train_data = self.create_sliding_window_data(train_df)
        test_data = self.create_sliding_window_data(test_df)
        
        # 4. Convert to PyTorch Tensors
        train_dataset = BatteryDataset(
            torch.from_numpy(train_data['X']),
            torch.from_numpy(train_data['T']),
            torch.from_numpy(train_data['E'])
        )
        
        test_dataset = BatteryDataset(
            torch.from_numpy(test_data['X']),
            torch.from_numpy(test_data['T']),
            torch.from_numpy(test_data['E'])
        )
        
        return train_dataset, test_dataset

def get_dataloaders(config: DataConfig, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    설정값으로 데이터를 로드하고 DataLoader를 반환하는 헬퍼 함수.
    """
    processor = BatteryDataProcessor(config)
    
    # 데이터 로드 및 피처 생성
    df, _ = processor.load_and_process_features()
    
    # 데이터셋 생성 (스플릿, 스케일링, 윈도우 포함)
    train_dataset, test_dataset = processor.prepare_datasets(df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ===================================================================================
# Main Execution (Verification)
# ===================================================================================

if __name__ == "__main__":
    # 데이터 로더 동작 검증 코드
    print("=== Battery Data Loader Verification ===")
    
    # 테스트 설정
    config = DataConfig(
        window_size=10,
        min_cycles=30, # 테스트용으로 낮춤
        feature_set_name='dynamic'
    )
    
    try:
        processor = BatteryDataProcessor(config)
        
        # 1. 파일 존재 여부 확인 (경로가 다를 수 있으므로 예외처리)
        if not os.path.exists(config.metadata_path):
            print(f"[Warning] Metadata not found at {config.metadata_path}.")
            print("Please adjust 'DataConfig.base_dir' in the script if needed.")
        else:
            # 2. 로드 및 프로세싱 테스트
            df, features = processor.load_and_process_features()
            print(f"Features: {features}")
            print(f"Sample Records:\n{df.head()}")
            
            # 3. 데이터셋 생성 테스트
            train_ds, test_ds = processor.prepare_datasets(df)
            
            # 4. 차원 확인
            sample_x, sample_t, sample_e = train_ds[0]
            print(f"\n[Verification Success]")
            print(f"Input Shape (Window): {sample_x.shape} (Expected: ({config.window_size}, {len(features)}))")
            print(f"Time Label: {sample_t:.2f}")
            print(f"Event Label: {sample_e}")
            
            # 5. DataLoader 테스트
            loader, _ = get_dataloaders(config, batch_size=4)
            batch_x, batch_t, batch_e = next(iter(loader))
            print(f"Batch Shape: {batch_x.shape}")
            
    except Exception as e:
        print(f"\n[Error] Verification failed: {e}")
        import traceback
        traceback.print_exc()
