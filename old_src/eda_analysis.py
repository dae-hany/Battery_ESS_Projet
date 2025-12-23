import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ==========================================
# 설정 (Configuration)
# ==========================================
# 프로젝트 루트 경로 설정
BASE_DIR = r"c:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
METADATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset", "metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "cleaned_dataset", "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_and_plot_data():
    print("메타데이터 로딩 중...")
    df = pd.read_csv(METADATA_PATH)
    
    # 방전(discharge) 사이클만 필터링
    discharge_df = df[df['type'] == 'discharge'].copy()
    
    # Capacity를 숫자형으로 변환 (오류 발생 시 NaN 처리)
    discharge_df['Capacity'] = pd.to_numeric(discharge_df['Capacity'], errors='coerce')
    
    # NaN 값이 있는 행 제거
    discharge_df = discharge_df.dropna(subset=['Capacity'])
    
    # 고유 배터리 ID 추출
    batteries = discharge_df['battery_id'].unique()
    print(f"식별된 배터리 목록: {batteries}")
    
    # 축 범위 고정을 위한 전역 최소/최대값 계산
    all_cycles = []
    all_capacities = []
    
    battery_data = {}
    stats = []

    for battery in batteries:
        bat_df = discharge_df[discharge_df['battery_id'] == battery]
        bat_df = bat_df.sort_values('test_id')
        
        cycles = len(bat_df)
        if cycles > 0:
            init_cap = bat_df['Capacity'].iloc[0]
            final_cap = bat_df['Capacity'].iloc[-1]
            stats.append({
                'Battery': battery,
                'Cycles': cycles,
                'Init_Cap': init_cap,
                'Final_Cap': final_cap,
                'Degradation': (init_cap - final_cap) / init_cap
            })
            
            battery_data[battery] = bat_df['Capacity'].values
            all_cycles.append(cycles)
            all_capacities.extend(bat_df['Capacity'].values)

    max_cycles = max(all_cycles) if all_cycles else 0
    min_cap = min(all_capacities) if all_capacities else 0
    max_cap = max(all_capacities) if all_capacities else 2.0
    
    # 배터리를 4개 그룹으로 분할하여 시각화
    battery_groups = np.array_split(batteries, 4)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, group in enumerate(battery_groups):
        ax = axes[i]
        for battery in group:
            if battery in battery_data:
                data = battery_data[battery]
                ax.plot(range(len(data)), data, label=battery)
        
        ax.set_title(f'Group {i+1}')
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Capacity (Ah)')
        ax.set_xlim(0, max_cycles)
        ax.set_ylim(min_cap, max_cap)
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True)

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, "capacity_trends_grouped.png")
    plt.savefig(output_path)
    print(f"그룹별 용량 그래프 저장 완료: {output_path}")

    # 통계 데이터 저장
    stats_df = pd.DataFrame(stats)
    print("\n배터리 통계 요약:")
    print(stats_df.to_string())
    
    stats_path = os.path.join(RESULTS_DIR, "battery_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"통계 파일 저장 완료: {stats_path}")
    
    # 이상치 확인 (사이클 < 50 또는 초기 용량 < 1.0)
    print("\n잠재적 이상치 (Cycles < 50 or Init_Cap < 1.0):")
    outliers = stats_df[(stats_df['Cycles'] < 50) | (stats_df['Init_Cap'] < 1.0)]
    print(outliers.to_string())

if __name__ == "__main__":
    load_and_plot_data()
