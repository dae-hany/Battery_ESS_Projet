import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 설정 (Configuration)
# ==========================================
BASE_DIR = r"c:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
METADATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset", "metadata.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# DeepSurv 학습을 위해 엄선된 10개의 배터리 목록
SELECTED_BATTERIES = [
    'B0005', 'B0006', 'B0007', 'B0018',  # NASA 표준 배터리 (Standard)
    'B0042', 'B0043', 'B0044',           # 추가 양품 배터리 (Additional Good)
    'B0046', 'B0047', 'B0048'            # 추가 양품 배터리 (Additional Good)
]

def plot_selected_batteries():
    print("메타데이터 로딩 중...")
    df = pd.read_csv(METADATA_PATH)
    
    # 방전 데이터 필터링 및 전처리
    discharge_df = df[df['type'] == 'discharge'].copy()
    discharge_df['Capacity'] = pd.to_numeric(discharge_df['Capacity'], errors='coerce')
    discharge_df = discharge_df.dropna(subset=['Capacity'])
    
    plt.figure(figsize=(12, 8))
    
    print(f"엄선된 {len(SELECTED_BATTERIES)}개 배터리 시각화 중...")
    
    for battery in SELECTED_BATTERIES:
        bat_df = discharge_df[discharge_df['battery_id'] == battery].sort_values('test_id')
        
        if len(bat_df) == 0:
            print(f"경고: {battery}에 대한 데이터가 없습니다.")
            continue
            
        cycles = range(len(bat_df))
        capacity = bat_df['Capacity'].values
        
        plt.plot(cycles, capacity, label=battery, linewidth=2)
        
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('Capacity (Ah)', fontsize=12)
    plt.title('Capacity Degradation of Selected 10 Batteries (엄선된 10개 배터리의 용량 감소)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, "capacity_trends_selected.png")
    plt.savefig(output_path)
    print(f"그래프 저장 완료: {output_path}")

if __name__ == "__main__":
    plot_selected_batteries()
