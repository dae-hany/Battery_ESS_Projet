import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

# 현재 디렉토리 모듈 임포트 설정
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from model import DeepSurv

class DeepSurvNasaPredictor:
    """NASA 데이터셋 맞춤형 DeepSurv 추론 클래스"""
    
    def __init__(self, model_dir_path=None):
        if model_dir_path is None:
            model_dir_path = current_dir / "saved_models"
        
        self.model_dir = Path(model_dir_path)
        
        # 1. 메타데이터 로드
        meta_path = self.model_dir / "model_metadata.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            
        self.meta = joblib.load(meta_path)
        self.feature_cols = self.meta['feature_cols'] # 모델이 학습한 피처 (ex: R_1000Hz, X_100Hz...)
        self.mean = self.meta['scaler_mean']
        self.std = self.meta['scaler_std']
        self.slope = self.meta['calibration_slope']
        self.intercept = self.meta['calibration_intercept']
        
        # 2. 모델 초기화 및 가중치 로드
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepSurv(in_features=len(self.feature_cols), hidden_layers=[64, 32], dropout=0.0)
        
        weights_path = self.model_dir / "deepsurv_final.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
            
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully.")
        print(f"Target Features ({len(self.feature_cols)}): {self.feature_cols[:3]} ...")

    def _parse_frequency(self, col_name):
        """컬럼명에서 주파수(Hz) 추출 (ex: 'R_1kHz' -> 1000.0)"""
        # 단위 처리 (k/m 등)
        name = col_name.lower().replace('hz', '')
        # 숫자 추출
        num_match = re.search(r'[\d\.]+', name)
        if not num_match: return None
        
        num = float(num_match.group(0))
        if 'k' in name: num *= 1000
        elif 'm' in name: num /= 1000 # milli-Hertz if exists
        
        return num

    def _map_nasa_features(self, nasa_row):
        """
        NASA 데이터의 한 행을 모델 입력 포맷으로 변환
        * 모델이 요구하는 주파수와 가장 가까운 NASA 주파수 컬럼을 찾아 매핑
        """
        mapped_features = []
        
        # NASA 컬럼 분석
        nasa_cols = nasa_row.index.tolist()
        nasa_r_cols = [c for c in nasa_cols if c.startswith('R_')]
        nasa_x_cols = [c for c in nasa_cols if c.startswith('X_')]
        
        # 주파수 파싱 캐시
        nasa_r_freq_map = {c: self._parse_frequency(c) for c in nasa_r_cols}
        nasa_x_freq_map = {c: self._parse_frequency(c) for c in nasa_x_cols}
        
        for model_col in self.feature_cols:
            target_freq = self._parse_frequency(model_col)
            if target_freq is None:
                mapped_features.append(0.0)
                continue
                
            is_real = model_col.startswith('R_')
            search_pool = nasa_r_freq_map if is_real else nasa_x_freq_map
            
            # 가장 가까운 주파수 찾기
            best_col = None
            min_diff = float('inf')
            
            for nasa_col, freq in search_pool.items():
                if freq is None: continue
                diff = abs(freq - target_freq)
                if diff < min_diff:
                    min_diff = diff
                    best_col = nasa_col
            
            # 임계값: 주파수 차이가 너무 크면 매핑 안 함? (일단 강제 매핑하되 Warning)
            # 여기서는 Nearest Neighbor 사용
            if best_col is not None:
                val = nasa_row[best_col]
                mapped_features.append(val)
            else:
                mapped_features.append(0.0) # 해당 영역 데이터 없음
                
        return np.array(mapped_features).reshape(1, -1).astype(np.float32)

    def predict_nasa_csv(self, csv_path):
        """NASA 포맷 CSV 파일 전체 예측"""
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded NASA Data: {len(df)} rows")
            
            results = []
            
            for idx, row in df.iterrows():
                # 1. Feature Mapping
                X = self._map_nasa_features(row)
                
                # 2. Scaling
                X_scaled = (X - self.mean) / self.std
                
                # 3. Inference
                with torch.no_grad():
                    t_x = torch.tensor(X_scaled).to(self.device)
                    risk_score = self.model(t_x).item()
                    
                # 4. SOH Calculation
                pred_soh = self.slope * risk_score + self.intercept
                
                results.append({
                    'Battery_ID': row.get('Battery_ID', f'Row_{idx}'),
                    'True_SOH': row.get('SOH', None),
                    'Pred_SOH': pred_soh,
                    'Risk': risk_score
                })
                
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Error processing NASA CSV: {e}")
            return None

def main():
    nasa_csv_path = r"C:\Users\User\daehan_study\Battery_ESS_Project_ECM_ANTIGRAVITY\nasa_battery_eis_28features.csv"
    
    predictor = DeepSurvNasaPredictor()
    
    # Run Prediction
    print("\n--- Running Inference on NASA Dataset ---")
    res_df = predictor.predict_nasa_csv(nasa_csv_path)
    
    if res_df is not None:
        # 결과 분석
        print("\n=== Prediction Summary ===")
        print(res_df[['Battery_ID', 'True_SOH', 'Pred_SOH']].head(10))
        
        # Metric
        valid_df = res_df.dropna(subset=['True_SOH'])
        if not valid_df.empty:
            mse = np.mean((valid_df['True_SOH'] - valid_df['Pred_SOH'])**2)
            mae = np.mean(np.abs(valid_df['True_SOH'] - valid_df['Pred_SOH']))
            r2 = 1 - (np.sum((valid_df['True_SOH'] - valid_df['Pred_SOH'])**2) / 
                      np.sum((valid_df['True_SOH'] - valid_df['True_SOH'].mean())**2))
            
            print(f"\nOverall Performance on NASA Data:")
            print(f"  MAE : {mae:.4f}")
            print(f"  RMSE: {np.sqrt(mse):.4f}")
            print(f"  R2  : {r2:.4f}")
            
            # Save results
            save_path = current_dir / "nasa_prediction_results.csv"
            res_df.to_csv(save_path, index=False)
            print(f"\nDetailed results saved to {save_path}")

if __name__ == "__main__":
    main()
