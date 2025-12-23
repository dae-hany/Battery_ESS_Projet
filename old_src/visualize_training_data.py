import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ==========================================
# 설정 (Configuration)
# ==========================================
BASE_DIR = r"c:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
METADATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset", "metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "cleaned_dataset", "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

SAMPLE_BATTERY = 'B0005' # 시각화할 샘플 배터리
SOH_LIMIT = 0.8          # 수명 종료 임계값

def visualize_training_data():
    print(f"{SAMPLE_BATTERY} 데이터 로딩 중...")
    df = pd.read_csv(METADATA_PATH)
    
    # 방전 데이터 필터링
    bat_df = df[(df['battery_id'] == SAMPLE_BATTERY) & (df['type'] == 'discharge')].copy()
    bat_df = bat_df.sort_values('test_id')
    
    # Capacity 변환 및 결측치 제거
    bat_df['Capacity'] = pd.to_numeric(bat_df['Capacity'], errors='coerce')
    bat_df = bat_df.dropna(subset=['Capacity'])
    
    # SOH 및 스무딩된 용량 계산
    init_cap = bat_df['Capacity'].iloc[0]
    bat_df['Capacity_Smooth'] = bat_df['Capacity'].rolling(window=5, min_periods=1).mean()
    bat_df['SOH'] = bat_df['Capacity_Smooth'] / init_cap
    
    # 피처 추출 (Feature Extraction)
    features = {
        'Cycle': [],
        'Discharge Time (s)': [],
        'Max Temp (C)': [],
        'Min Voltage (V)': [],
        'Smoothed Capacity (Ah)': []
    }
    
    print("개별 사이클 파일에서 피처 추출 중...")
    for i, row in bat_df.iterrows():
        filename = row['filename']
        file_path = os.path.join(DATA_DIR, filename)
        
        try:
            cycle_data = pd.read_csv(file_path)
            
            discharge_time = cycle_data['Time'].max() - cycle_data['Time'].min()
            max_temp = cycle_data['Temperature_measured'].max()
            min_voltage = cycle_data['Voltage_measured'].min()
            
            features['Cycle'].append(len(features['Cycle']) + 1)
            features['Discharge Time (s)'].append(discharge_time)
            features['Max Temp (C)'].append(max_temp)
            features['Min Voltage (V)'].append(min_voltage)
            features['Smoothed Capacity (Ah)'].append(row['Capacity_Smooth'])
            
        except Exception as e:
            print(f"{filename} 읽기 오류: {e}")
            
    feat_df = pd.DataFrame(features)
    
    # 이벤트(Event) 및 시간(Time) 결정
    if (bat_df['SOH'] < SOH_LIMIT).any():
        eol_idx = np.where(bat_df['SOH'] < SOH_LIMIT)[0][0]
    else:
        eol_idx = len(feat_df) - 1
    
    # 시각화 (Plotting)
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f'학습 데이터 구조 시각화 (예시: {SAMPLE_BATTERY})', fontsize=16, weight='bold')
    
    # 그리드 레이아웃 설정
    gs = fig.add_gridspec(3, 2)
    
    # 1. 입력 피처 (Input Features) - 상단 2행
    
    # Feature 1: 방전 시간
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(feat_df['Cycle'], feat_df['Discharge Time (s)'], color='tab:blue', linewidth=2)
    ax1.set_title('입력 1: 방전 시간 (Discharge Time)', weight='bold')
    ax1.set_ylabel('Time (s)')
    ax1.grid(True, alpha=0.3)
    
    # Feature 2: 최고 온도
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(feat_df['Cycle'], feat_df['Max Temp (C)'], color='tab:red', linewidth=2)
    ax2.set_title('입력 2: 최고 온도 (Max Temperature)', weight='bold')
    ax2.set_ylabel('Temp (C)')
    ax2.grid(True, alpha=0.3)
    
    # Feature 3: 최저 전압
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(feat_df['Cycle'], feat_df['Min Voltage (V)'], color='tab:green', linewidth=2)
    ax3.set_title('입력 3: 최저 전압 (Min Voltage)', weight='bold')
    ax3.set_ylabel('Voltage (V)')
    ax3.grid(True, alpha=0.3)
    
    # Feature 4: 스무딩된 용량
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(feat_df['Cycle'], feat_df['Smoothed Capacity (Ah)'], color='tab:purple', linewidth=2)
    ax4.set_title('입력 4: 스무딩된 용량 (Smoothed Capacity)', weight='bold')
    ax4.set_ylabel('Capacity (Ah)')
    ax4.grid(True, alpha=0.3)
    
    # 2. 타겟 데이터 (Target Data) - 하단 1행
    ax_target = fig.add_subplot(gs[2, :])
    
    # SOH 곡선
    ax_target.plot(feat_df['Cycle'], bat_df['SOH'].values, color='black', linewidth=2, label='SOH')
    
    # 임계값 라인
    ax_target.axhline(y=SOH_LIMIT, color='red', linestyle='--', linewidth=2, label=f'Threshold ({SOH_LIMIT*100}%)')
    
    # 이벤트 마커
    if (bat_df['SOH'] < SOH_LIMIT).any():
        fail_cycle = feat_df['Cycle'].iloc[eol_idx]
        ax_target.scatter(fail_cycle, SOH_LIMIT, color='red', s=100, zorder=5, label='Event (Failure)')
        ax_target.annotate(f'Event (E=1)\nTime (T) = {fail_cycle}', 
                           xy=(fail_cycle, SOH_LIMIT), 
                           xytext=(fail_cycle+10, SOH_LIMIT+0.05),
                           arrowprops=dict(facecolor='black', shrink=0.05),
                           fontsize=12, weight='bold')
        
        # 고장 이후 영역 표시
        ax_target.axvspan(fail_cycle, feat_df['Cycle'].max(), color='red', alpha=0.1, label='Failure Region')
    
    ax_target.set_title('타겟 데이터 정의: 이벤트(E) & 시간(T)', weight='bold')
    ax_target.set_xlabel('Cycle Number')
    ax_target.set_ylabel('SOH')
    ax_target.legend(loc='upper right')
    ax_target.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(PLOTS_DIR, "training_data_visualization.png")
    plt.savefig(output_path)
    print(f"시각화 저장 완료: {output_path}")

if __name__ == "__main__":
    visualize_training_data()
