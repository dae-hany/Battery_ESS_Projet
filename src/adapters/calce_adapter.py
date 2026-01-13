# -*- coding: utf-8 -*-
"""
CALCE Dataset Adapter
======================
Converts CALCE Battery Dataset (Excel files) to project-standard CSV format.

Output Structure:
    cleaned_dataset/
        calce_metadata.csv
        calce_data/
            filename_1.csv
            filename_2.csv
            ...
"""

import os
import pandas as pd
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_file(file_info):
    """
    Process a single CALCE Excel file.
    Args:
        file_info: tuple (file_path, bat_id, output_data_dir)
    Returns:
        List of metadata dictionaries
    """
    file_path, bat_id, output_data_dir = file_info
    
    try:
        # Load Excel File Object to check sheets
        xl = pd.ExcelFile(file_path)
        
        target_sheet = None
        for sheet in xl.sheet_names:
            # Peek columns
            try:
                # header=0 is default
                cols = pd.read_excel(file_path, sheet_name=sheet, nrows=0).columns
                if 'Cycle_Index' in cols:
                    target_sheet = sheet
                    break
            except:
                continue
                
        if target_sheet is None:
             # Try fallback to sheet 0 if we couldn't check columns efficiently
             target_sheet = 0
             
        # Load Data
        df = pd.read_excel(file_path, sheet_name=target_sheet)
        
        # Standardize Columns
        # CALCE columns: 'Cycle_Index', 'Voltage(V)', 'Current(A)', 'Test_Time(s)', 'Date_Time', 'Discharge_Capacity(Ah)'
        # We need: 'voltage', 'current', 'time', 'capacity' (for metadata)
        
        # Check if Cycle_Index exists
        if 'Cycle_Index' not in df.columns:
            return []
            
        metadata_list = []
        
        # Group by Cycle
        grouped = df.groupby('Cycle_Index')
        
        for cycle_idx, group in grouped:
            # Filter for Discharge Only
            # CALCE convention: Discharge Current is usually Negative? Or Positive?
            # Let's check 'Discharge_Capacity(Ah)'. It should be increasing.
            # Or usually: Current < 0 for discharge.
            # But relying on 'Discharge_Capacity(Ah)' > 0 is safer ensuring it is a discharge cycle.
            
            # Simple check: Does this cycle have significant discharge capacity?
            max_cap = group['Discharge_Capacity(Ah)'].max()
            if  max_cap < 0.01: # Skip low capacity cycles (rest/charge only)
                continue
                
            # Extract V, I, T
            # Re-index time to start at 0 for this cycle
            times = group['Test_Time(s)'].values
            if len(times) < 10:
                continue
            times = times - times[0]
            
            # Save to CSV
            # Naming: {bat_id}_cycle_{cycle_idx}.csv
            filename = f"{bat_id}_cycle_{int(cycle_idx)}.csv"
            save_path = os.path.join(output_data_dir, filename)
            
            cycle_df = pd.DataFrame({
                'time': times,
                'voltage': group['Voltage(V)'].values,
                'current': group['Current(A)'].values,
                # 'temp': group['Temperature(C)'].values # If exists
            })
            
            cycle_df.to_csv(save_path, index=False)
            
            # Metadata Entry
            metadata_list.append({
                'battery_id': bat_id,
                'type': 'discharge',
                'start_time': group['Date_Time'].iloc[0] if 'Date_Time' in group else None,
                'ambient_temperature': 25, # Default Assumption if not parsed from filename
                'test_id': int(cycle_idx),
                'uid': f"{bat_id}_{int(cycle_idx)}",
                'filename': filename,
                'capacity': max_cap
            })
            
        return metadata_list
        
    except Exception as e:
        print(f"[Error] Failed to process {file_path}: {e}")
        return []

def main():
    # Paths
    base_dir = r"c:\Users\User\daehan_study\Battery_ESS_Projet"
    raw_dir = os.path.join(base_dir, "calce")
    output_dir = os.path.join(base_dir, "cleaned_dataset")
    output_data_dir = os.path.join(output_dir, "calce_data")
    
    os.makedirs(output_data_dir, exist_ok=True)
    
    # Check Raw Files
    # Find all .xls, .xlsx recursively
    print(f"Searching for files in {raw_dir}...")
    files = glob.glob(os.path.join(raw_dir, "**", "*.xls*"), recursive=True)
    
    if not files:
        print("No Excel files found in 'calce/' directory.")
        return
        
    print(f"Found {len(files)} files to process.")
    
    # Prepare Args
    tasks = []
    for f in files:
        basename = os.path.basename(f)
        if basename.startswith('~$'): continue # Skip temp files
        
        # Extract Bat ID
        # Expected: CS2_35_... -> CS2_35
        # Or SP20-1...
        
        parts = basename.split('_')
        if len(parts) >= 2 and parts[0].startswith('CS2'):
            bat_id = f"{parts[0]}_{parts[1]}"
        elif 'SP20-1' in basename:
            bat_id = 'SP20-1'
        else:
            # Skip file if pattern doesn't match known batteries to avoid junk
            continue 
            
        tasks.append((f, bat_id, output_data_dir))

    if not tasks:
        print("No matching battery files (CS2_xx or SP20-1) found.")
        return

    print(f"Processing {len(tasks)} valid battery files...")
    
    # Execute with Parallel Processing
    all_metadata = []
    
    # Use max_workers = roughly CPU count. 
    # Excel reading is CPU bound (parsing XML) and IO bound.
    max_workers = min(os.cpu_count(), 8) 
    print(f"Starting conversion with {max_workers} workers (This may still take a few minutes)...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # map returns results in order
        results = list(tqdm(executor.map(process_file, tasks), total=len(tasks)))
        
    for meta in results:
        all_metadata.extend(meta)
        
    # Save Metadata
    if all_metadata:
        meta_df = pd.DataFrame(all_metadata)
        meta_path = os.path.join(output_dir, "calce_metadata.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"\n[Success] Converted {len(meta_df)} cycles.")
        print(f"Metadata saved to: {meta_path}")
        print(f"Data saved to: {output_data_dir}")
    else:
        print("\n[Warning] No valid cycles extracted.")

if __name__ == "__main__":
    main()
