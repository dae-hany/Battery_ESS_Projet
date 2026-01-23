import scipy.io
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_and_inspect(file_path):
    mat = scipy.io.loadmat(file_path)
    data = mat['B0018'][0, 0]['cycle'][0]
    
    eis_times = []
    discharge_times = []
    discharge_caps = []
    
    for i, cycle in enumerate(data):
        cycle_type = cycle['type'][0]
        
        # Parse Time
        start_time_str = cycle['time'][0]
        # B0018 might have different date format or structure, but usually it's standard
        # However, let's just trust the parsing logic for now or rely on the relative time array if needed
        # But phase_1 uses the string conversion.
        # Let's extract the 'time' field which is a numeric timestamp sometimes, or we parse the date string.
        # matched_cap uses: 
        # t_str = cycle['time'][0]
        # dt = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S') -> timestamp
        
        try:
             # Standard NASA format
            t_str = str(cycle['time'][0])
            # Sometimes dates are in different formats in this dataset
            # But let's assume it works as phase_1.py didn't crash
            # Actually, let's just print the raw strings first 5
            pass
        except:
            continue

        if cycle_type == 'impedance':
            # In phase_1, we parse time
            try:
                t = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S').timestamp()
                eis_times.append(t)
            except:
                pass
                
        elif cycle_type == 'discharge':
            try:
                t = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S').timestamp()
                data_field = cycle['data']
                if 'Capacity' in data_field.dtype.names:
                    cap = data_field['Capacity'][0][0][0]
                    discharge_times.append(t)
                    discharge_caps.append(cap)
            except:
                pass

    print(f"Total EIS cycles: {len(eis_times)}")
    print(f"Total Discharge cycles: {len(discharge_times)}")
    
    if len(eis_times) > 0 and len(discharge_times) > 0:
        print("\nFirst 5 EIS Timestamps:")
        print(eis_times[:5])
        print("\nFirst 5 Discharge Timestamps:")
        print(discharge_times[:5])
        
        print("\nLast 5 EIS Timestamps:")
        print(eis_times[-5:])
        print("\nLast 5 Discharge Timestamps:")
        print(discharge_times[-5:])
        
        # Check average gap
        df_eis = pd.DataFrame({'time': eis_times})
        df_cap = pd.DataFrame({'time': discharge_times, 'cap': discharge_caps})
        
        print("\nAnalysing Matches:")
        for i in range(min(5, len(df_eis))):
            row = df_eis.iloc[i]
            time_diff = (df_cap['time'] - row['time']).abs()
            idx = time_diff.idxmin()
            min_diff = time_diff[idx]
            match_cap = df_cap.loc[idx, 'cap']
            print(f"EIS #{i}: Diff={min_diff/3600:.2f} hours, Cap={match_cap}")

file_path = 'NASA_PCOE_DATA/B0018.mat'
if os.path.exists(file_path):
    load_and_inspect(file_path)
else:
    print(f"File not found: {file_path}")
