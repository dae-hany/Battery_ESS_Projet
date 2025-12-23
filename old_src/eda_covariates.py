import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 설정 (Configuration)
# ==========================================
BASE_DIR = r"c:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
METADATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset", "metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "cleaned_dataset", "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def plot_covariates(battery_id):
    """
    특정 배터리의 전압, 전류, 온도 곡선을 초기/중기/말기 사이클로 나누어 시각화
    """
    print(f"{battery_id} 공변량(Covariates) 분석 중...")
    df = pd.read_csv(METADATA_PATH)
    
    # 해당 배터리의 방전 데이터 필터링
    bat_df = df[(df['battery_id'] == battery_id) & (df['type'] == 'discharge')].copy()
    bat_df = bat_df.sort_values('test_id')
    
    if len(bat_df) == 0:
        print(f"{battery_id}에 대한 방전 데이터가 없습니다.")
        return

    # 3개의 대표 사이클 선택: 처음(First), 중간(Middle), 마지막(Last)
    cycles_to_plot = [0, len(bat_df)//2, len(bat_df)-1]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    for i, idx in enumerate(cycles_to_plot):
        row = bat_df.iloc[idx]
        filename = row['filename']
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없음: {file_path}")
            continue
            
        try:
            # 사이클 데이터 로드
            cycle_data = pd.read_csv(file_path)
            
            # 전압 곡선 (Voltage)
            axes[0].plot(cycle_data['Time'], cycle_data['Voltage_measured'], label=f'Cycle {idx+1}')
            axes[0].set_ylabel('Voltage (V)')
            axes[0].set_title(f'{battery_id} - 전압 곡선 (Voltage Curves)')
            
            # 전류 곡선 (Current)
            axes[1].plot(cycle_data['Time'], cycle_data['Current_measured'], label=f'Cycle {idx+1}')
            axes[1].set_ylabel('Current (A)')
            axes[1].set_title(f'{battery_id} - 전류 곡선 (Current Curves)')
            
            # 온도 곡선 (Temperature)
            axes[2].plot(cycle_data['Time'], cycle_data['Temperature_measured'], label=f'Cycle {idx+1}')
            axes[2].set_ylabel('Temperature (C)')
            axes[2].set_title(f'{battery_id} - 온도 곡선 (Temperature Curves)')
            
        except Exception as e:
            print(f"{filename} 읽기 오류: {e}")

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[2].set_xlabel('Time (s)')
    
    output_path = os.path.join(PLOTS_DIR, f"covariates_{battery_id}.png")
    plt.savefig(output_path)
    print(f"공변량 그래프 저장 완료: {output_path}")

if __name__ == "__main__":
    # 대표적인 정상 배터리 분석
    plot_covariates('B0005')
    # 비교를 위한 다른 배터리 분석
    plot_covariates('B0047')
