import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

def extract_soh(filename):
    match = re.search(r'SOH(\d+\.?\d*)', filename)
    return float(match.group(1)) if match else None

def analyze_spectroscopy_labels():
    base_path = Path(__file__).resolve().parent.parent / "datasets/raw_data/Spectroscopy_Individual"
    files = list(base_path.glob("*.csv"))
    
    records = []
    for f in files:
        soh = extract_soh(f.name)
        if soh is None: continue
        
        try:
            df = pd.read_csv(f)
            # 주요 주파수 대역의 저항값 추출 (물리적 지표)
            # 배터리 노화 시 내부 저항(R)은 일반적으로 증가함.
            # 1kHz (Ohmic Resistance)와 저주파 대역 (Charge Transfer) 확인
            
            # 컬럼명 정규화 필요. 파일마다 다를 수 있음.
            # 일단 R_... 형태로 되어있다고 가정하거나 찾아야 함.
            r_cols = [c for c in df.columns if c.startswith('R_')]
            
            if not r_cols: continue
            
            # 평균 저항값 (전체 주파수 대역 평균)
            avg_r = df[r_cols].values.mean()
            
            # 특정 주파수 (예: R_1kHz 또는 가장 높은 주파수, 가장 낮은 주파수)
            # 여기선 간단히 전체 평균 R 사용
            
            records.append({
                'filename': f.name,
                'SOH': soh,
                'Avg_R': avg_r
            })
        except:
            pass
            
    if not records:
        print("No valid records found.")
        return

    df_res = pd.DataFrame(records)
    
    # 상관관계 분석
    corr = df_res['SOH'].corr(df_res['Avg_R'])
    print(f"Correlation between SOH and Average Resistance: {corr:.4f}")
    print("Expected: Negative Correlation (High SOH -> Low Resistance)")
    
    # 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(df_res['Avg_R'], df_res['SOH'], c='blue', alpha=0.7)
    
    # Trend Line
    z = np.polyfit(df_res['Avg_R'], df_res['SOH'], 1)
    p = np.poly1d(z)
    plt.plot(df_res['Avg_R'], p(df_res['Avg_R']), "r--", label=f'Trend (Corr: {corr:.2f})')
    
    plt.title("SOH Label Validation: Resistance vs SOH")
    plt.xlabel("Average Resistance (Ohm)")
    plt.ylabel("Labeled SOH (%)")
    plt.grid(True)
    plt.legend()
    
    save_path = Path(__file__).parent / "label_validation.png"
    plt.savefig(save_path)
    print(f"Validation plot saved to {save_path}")
    
    # 이상치 탐지 (Trend에서 많이 벗어난 데이터)
    # Residual Calculation
    df_res['Predicted_SOH'] = p(df_res['Avg_R'])
    df_res['Residual'] = np.abs(df_res['SOH'] - df_res['Predicted_SOH'])
    
    # 잔차가 큰 상위 3개 출력
    outliers = df_res.nlargest(3, 'Residual')
    print("\nPotential Label Errors (High Residuals):")
    print(outliers[['filename', 'SOH', 'Avg_R', 'Residual']])

if __name__ == "__main__":
    analyze_spectroscopy_labels()
