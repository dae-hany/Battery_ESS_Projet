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

class DeepSurvPredictor:
    """학습된 DeepSurv 모델을 사용하여 SOH를 예측하는 클래스"""
    
    def __init__(self, model_dir_path=None):
        if model_dir_path is None:
            model_dir_path = current_dir / "saved_models"
        
        self.model_dir = Path(model_dir_path)
        
        # 1. 메타데이터 로드
        meta_path = self.model_dir / "model_metadata.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            
        self.meta = joblib.load(meta_path)
        self.feature_cols = self.meta['feature_cols']
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
        
        print(f"Model loaded successfully from {self.model_dir}")
        print(f"Calibration: SOH = {self.slope:.4f} * Risk + {self.intercept:.4f}")

    def predict(self, csv_file_path):
        """
        단일 CSV 파일에 대한 SOH 예측
        
        Returns:
            pred_soh (float): 예측된 SOH (%)
            risk_score (float): 예측된 위험도 점수
        """
        # 1. 데이터 파싱
        try:
            df = pd.read_csv(csv_file_path)
            # 공백 제거
            df.columns = df.columns.str.strip()
            
            # 피처 추출
            # 학습 때 사용한 feature_cols 순서대로 값을 뽑아야 함
            features = []
            
            # 데이터가 시계열(여러 행)이면 평균 사용
            means = df.mean(numeric_only=True)
            
            for col in self.feature_cols:
                if col in means:
                    features.append(means[col])
                else:
                    # 해당 주파수 데이터가 없으면 0 처리
                    features.append(0.0)
            
            X = np.array(features).reshape(1, -1).astype(np.float32)
            
            # 2. 정규화 (Standardization)
            X_scaled = (X - self.mean) / self.std
            
            # 3. 모델 추론
            with torch.no_grad():
                tensor_X = torch.tensor(X_scaled).to(self.device)
                risk_score = self.model(tensor_X).item()
                
            # 4. SOH 변환
            pred_soh = self.slope * risk_score + self.intercept
            
            return pred_soh, risk_score
            
        except Exception as e:
            print(f"Prediction Error for {csv_file_path}: {e}")
            return None, None

def main():
    # 사용 예시
    predictor = DeepSurvPredictor()
    
    # 테스트 파일 찾기 (Spectroscopy 데이터 중 하나)
    dataset_dir = current_dir.parent / "datasets/raw_data/Spectroscopy_Individual"
    test_files = list(dataset_dir.glob("*.csv"))
    
    if not test_files:
        print("No test files found.")
        return

    print("\n--- Running Inference on Random Samples ---")
    
    # 랜덤하게 3개 뽑아서 테스트
    samples = np.random.choice(test_files, min(len(test_files), 3), replace=False)
    
    for f in samples:
        print(f"\nTarget File: {f.name}")
        
        # 파일명에서 정답 추출 (비교용)
        match = re.search(r'SOH(\d+\.?\d*)', f.name)
        true_soh = float(match.group(1)) if match else "Unknown"
        
        # 예측 수행
        pred_soh, risk = predictor.predict(f)
        
        if pred_soh is not None:
            print(f"  > Predicted SOH : {pred_soh:.2f}%")
            print(f"  > True Label    : {true_soh}%")
            print(f"  > Network Risk  : {risk:.4f}")
            if isinstance(true_soh, float):
                print(f"  > Error         : {abs(pred_soh - true_soh):.2f}%p")

if __name__ == "__main__":
    main()
