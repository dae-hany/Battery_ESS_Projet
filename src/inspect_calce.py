
import pandas as pd
import os

file_path = r"c:\Users\User\daehan_study\Battery_ESS_Projet\calce\CS2_35\CS2_35\CS2_35_8_30_10.xlsx"

print(f"Inspecting: {file_path}")
try:
    # Read sheet names
    xl = pd.ExcelFile(file_path)
    print(f"Sheet names: {xl.sheet_names}")
    
    for sheet in xl.sheet_names:
        print(f"\n--- Sheet: {sheet} ---")
        try:
            df = xl.parse(sheet, nrows=5)
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error parsing sheet {sheet}: {e}")
    
except Exception as e:
    print(f"Error reading file: {e}")
