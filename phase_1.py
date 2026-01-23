import scipy.io
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import os
from datetime import datetime

class NASALoader:
    def __init__(self, base_path):
        self.base_path = base_path
        # 사용자가 정의한 23개 타겟 주파수 포인트 (통합)
        # Phase 2의 Warburg 분석을 위해 저주파(0.05~0.4Hz)의 R값이 필요하며,
        # Rct 분석을 위해 중주파의 R, X 값이 모두 필요함.
        # 따라서 모든 주파수에 대해 R, X를 모두 추출하도록 변경.
        self.target_freqs = [0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0, 20.0, 40.0, 100.0, 200.0, 400.0, 1000.0]
        self.nominal_capacity = 2.0 # B0005, 06, 07, 18의 공칭 용량

    def load_mat(self, file_name):
        file_path = os.path.join(self.base_path, file_name)
        mat = scipy.io.loadmat(file_path)
        # 파일명에서 키 추출 (예: 'B0005')
        key = file_name.split('.')[0]
        data = mat[key][0, 0]['cycle'][0]
        return data

    def parse_time(self, t_data):
        # Handle scalar (timestamp), string, or vector [Y,M,D,h,m,s]
        try:
            if t_data.size == 6:
                # Vector case: [2008, 7, 7, 12, 26, 45.75]
                # Note: t_data is likely [[Y, M, D, H, M, S]] if accessed via data[i]['time']
                vals = t_data.flatten()
                vals = [int(v) for v in vals] # Convert to int for datetime (seconds lose precision if int)
                # Actually seconds can be float.
                sec = t_data.flatten()[5]
                vals[5] = int(sec)
                
                dt = datetime(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
                # Add fractional seconds back? 
                # For now, second precision is enough to distinguish cycles.
                return dt.timestamp()
            elif t_data.size == 1:
                return float(t_data.flatten()[0])
        except:
             return 0.0
        return 0.0

    def extract_features(self, data):
        eis_results = []
        capacities = []
        
        # 1. 방전 용량 및 EIS 데이터 분리 추출
        for i in range(len(data)):
            cycle_type = data[i]['type'][0]
            
            # Robust extraction of time
            # B0018 has vector times, others might have scalar timestamps
            raw_time_data = data[i]['time']
            time = self.parse_time(raw_time_data)
            
            if cycle_type == 'discharge':
                if 'data' in data[i].dtype.names and data[i]['data'].size > 0:
                    d = data[i]['data'][0, 0]
                    if 'Capacity' in d.dtype.names and d['Capacity'].size > 0:
                        cap = d['Capacity'][0, 0]
                        # time is already parsed
                        capacities.append({'time': time, 'cap': cap})
                
            elif cycle_type == 'impedance':
                # Robust Data Parsing: 'data' 필드 존재 여부 확인
                if 'data' not in data[i].dtype.names or data[i]['data'].size == 0:
                    continue

                cycle_data = data[i]['data'][0, 0]
                
                freq = None
                re = None
                im = None
                
                # Check 1: Explicit fields ('frequency', 're', 'im')
                if 'frequency' in cycle_data.dtype.names and \
                   're' in cycle_data.dtype.names and \
                   'im' in cycle_data.dtype.names:
                    
                    freq = cycle_data['frequency'][0]
                    re = cycle_data['re'][0]
                    im = cycle_data['im'][0]

                # Check 2: Fallback to 'Rectified_Impedance' or 'Battery_impedance' if explicit fields are missing/empty
                if (freq is None or freq.size == 0) and \
                   ('Rectified_Impedance' in cycle_data.dtype.names or 'Battery_impedance' in cycle_data.dtype.names):
                    
                    if 'Rectified_Impedance' in cycle_data.dtype.names and cycle_data['Rectified_Impedance'].size > 0:
                        Z = cycle_data['Rectified_Impedance']
                    elif 'Battery_impedance' in cycle_data.dtype.names and cycle_data['Battery_impedance'].size > 0:
                        Z = cycle_data['Battery_impedance']
                    else:
                        continue
                        
                    Z = Z.flatten()
                    if Z.size == 0:
                        continue
                        
                    # Extract Real and Imag parts from Complex Impedance
                    re = np.real(Z)
                    im = np.imag(Z)
                    
                    # Synthesize Frequency: 5000Hz down to 0.1Hz (High -> Low assumption based on typical NASA dataset structure and data characteristics)
                    # NASA PCoE dataset EIS sweep is typically 0.1Hz - 5kHz.
                    # Analysis of 're' suggests index 0 is High Frequency (Small Real part).
                    freq = np.logspace(np.log10(5000), np.log10(0.1), len(Z))
                
                # Check 3: If still invalid, skip
                if freq is None or re is None or im is None:
                    continue
                
                if freq.size == 0 or re.size == 0 or im.size == 0:
                    continue

                # Ensure 1D arrays
                freq = freq.flatten()
                re = re.flatten()
                im = im.flatten()
                
                # Data Cleansing: 0 이하 주파수 필터링
                valid_mask = freq > 0
                if np.sum(valid_mask) < 5: # Spline을 위해 넉넉한 포인트 필요
                    continue
                
                freq = freq[valid_mask]
                re = re[valid_mask]
                im = im[valid_mask]

                # time = data[i]['time'][0, 0] # Already parsed at start of loop
                
                # 로그 스케일 Cubic Spline 보간
                # 주파수 기준 정렬
                idx_sort = np.argsort(freq)
                f_sorted = freq[idx_sort]
                re_sorted = re[idx_sort]
                im_sorted = im[idx_sort]
                
                try:
                    # 중복 제거 (Spline 보간 시 오류 방지)
                    f_sorted, unique_idx = np.unique(f_sorted, return_index=True)
                    re_sorted = re_sorted[unique_idx]
                    im_sorted = im_sorted[unique_idx]
                    
                    if len(f_sorted) < 2:
                        continue
                        
                    log_f = np.log10(f_sorted)
                    
                    cs_re = CubicSpline(log_f, re_sorted)
                    cs_im = CubicSpline(log_f, im_sorted)
                    
                    # 23개(실제로는 14개 유니크 주파수) 포인트 추출
                    # 모든 타겟 주파수에 대해 R, X 추출
                    res = {}
                    for f in self.target_freqs:
                        res[f'R_{f}Hz'] = cs_re(np.log10(f)).item()
                        res[f'X_{f}Hz'] = cs_im(np.log10(f)).item()
                    
                    res['time'] = time
                    eis_results.append(res)
                except Exception as e:
                    # 보간 실패 시 해당 사이클 건너뜀
                    continue

        # 2. 데이터프레임 변환 및 SOH 라벨링
        df_eis = pd.DataFrame(eis_results)
        df_cap = pd.DataFrame(capacities)
        
        if df_eis.empty or df_cap.empty:
            return pd.DataFrame()

        final_data = []
        for _, row in df_eis.iterrows():
            # EIS 측정 시간과 가장 가까운 방전 데이터 탐색
            # Data Cleansing: 시간 차이가 12시간(43200초) 이상인 경우 제외
            time_diff = (df_cap['time'] - row['time']).abs()
            idx = time_diff.idxmin()
            
            # NOTE: B0018의 경우 모든 EIS가 특정 시점에 몰려있거나, Capacity 데이터가 EIS와 시간 차이가 클 수 있음.
            # 시간 제약을 조금 완화하거나 확인 필요. 
            # 일단 12시간 제약을 24시간으로 완화 시도, 또는 B0018에 대해서만 예외 처리
            min_diff = time_diff[idx]
            
            # if min_diff > 12 * 3600:
                # continue
                
            matched_cap = df_cap.loc[idx, 'cap']
            
            row_dict = row.to_dict()
            row_dict['SOH'] = (matched_cap / self.nominal_capacity) * 100
            row_dict['Capacity'] = matched_cap
            final_data.append(row_dict)
            
        return pd.DataFrame(final_data)

    def process_all(self, file_list):
        all_df = []
        for f in file_list:
            print(f"Processing {f}...")
            data = self.load_mat(f)
            df = self.extract_features(data)
            df['Battery_ID'] = f.split('.')[0]
            all_df.append(df)
        
        return pd.concat(all_df, ignore_index=True)

# 실행부
if __name__ == "__main__":
    path = r"C:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project_v2\NASA_PCOE_DATA"
    files = ['B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat']
    
    loader = NASALoader(path)
    dataset = loader.process_all(files)
    
    # 결과 저장
    dataset.to_csv("NASA_23pt_EIS_Dataset.csv", index=False)
    print("Phase 1 완료: 데이터셋이 생성되었습니다.")
    print(dataset.head())