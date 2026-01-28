import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def extract_soh(filename):
    match = re.search(r'SOH(\d+\.?\d*)', filename)
    return float(match.group(1)) if match else None

def normalize_frequency(val):
    # Map raw frequency values to standard labels for sorting/plotting if needed
    # But for Nyquist, we plot -Imag vs Real directly.
    return val

def plot_nyquist_by_soh_group():
    base_path = Path(__file__).resolve().parent.parent / "datasets/raw_data/Spectroscopy_Individual"
    files = list(base_path.glob("*.csv"))
    
    if not files:
        print("No files found.")
        return

    # 1. Load Data
    data_list = []
    
    for f in files:
        soh = extract_soh(f.name)
        if soh is None: continue
        
        try:
            df = pd.read_csv(f)
            
            # Filter R_ and X_ columns
            cols = df.columns
            r_cols = [c for c in cols if c.startswith('R_')]
            x_cols = [c for c in cols if c.startswith('X_')]
            
            if not r_cols or not x_cols: continue
            
            # Extract values (Mean across timestamps/steps if multiple rows exist, usually 1 row per file in processed data)
            # data_loader methodology: mean() -> single vector
            r_vals = df[r_cols].mean().values
            x_vals = df[x_cols].mean().values
            
            # Frequency Check (Optional): Assumes columns are ordered by frequency?
            # Ideally, we should sort by frequency to draw lines correctly.
            # But columns R_1kHz, R_500Hz... are not sorted alphabetically.
            # Need to parse frequency from column name.
            
            freqs = []
            for c in r_cols:
                f_str = c.replace('R_', '').replace('Hz', '').strip()
                if 'k' in f_str.lower():
                    val = float(f_str.lower().replace('k', '')) * 1000
                else:
                    try:
                        val = float(f_str)
                    except:
                        val = 0
                freqs.append(val)
                
            # Zip and Sort by Frequency High -> Low (standard for Nyquist: left to right)
            # Usually Z' (Real) increases as Freq decreases.
            
            # Combine
            file_data = []
            for r, x, fr in zip(r_vals, x_vals, freqs):
                file_data.append({'R': r, 'X': x, 'Freq': fr})
            
            # Sort by Real part (easiest way to connect lines without frequency parsing if complex)
            # But Sorting by Frequency Descending is safer physically.
            file_data.sort(key=lambda k: k['Freq'], reverse=True) # High Freq -> Low Freq
            
            data_list.append({
                'filename': f.name,
                'SOH': soh,
                'data': file_data
            })
            
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not data_list:
        print("No valid data loaded for plotting.")
        return

    # 2. Group by SOH Ranges
    # Groups: High (>=90), Mid (85-90), Low (<85)
    groups = {'High SOH (>=90%)': [], 'Mid SOH (85-90%)': [], 'Low SOH (<85%)': []}
    
    for item in data_list:
        soh = item['SOH']
        if soh >= 90:
            groups['High SOH (>=90%)'].append(item)
        elif soh >= 85:
            groups['Mid SOH (85-90%)'].append(item)
        else:
            groups['Low SOH (<85%)'].append(item)
            
    # 3. Plot
    plt.figure(figsize=(10, 8))
    
    # Color palette
    colors = {'High SOH (>=90%)': 'green', 'Mid SOH (85-90%)': 'orange', 'Low SOH (<85%)': 'red'}
    
    print("\nPlotting Nyquist Curves...")
    
    for group_name, items in groups.items():
        if not items: continue
        
        print(f"  {group_name}: {len(items)} samples")
        color = colors[group_name]
        
        for i, item in enumerate(items):
            d = item['data']
            R = [p['R'] for p in d]
            X = [p['X'] for p in d]
            neg_X = [-val for val in X] # Nyquist: -Imag vs Real
            
            # Plot line
            label = group_name if i == 0 else "" # Label only once per group
            plt.plot(R, neg_X, '.-', color=color, alpha=0.6, label=label, markersize=4)
            
            # Annotate SOH on the last point (Low Freq)
            # plt.text(R[-1], neg_X[-1], f"{item['SOH']:.1f}", fontsize=8, color=color, alpha=0.8)

    plt.title("Nyquist Plot by SOH Group (Spectroscopy Data)", fontsize=14)
    plt.xlabel("Z' (Real Impedance) [Ohm]", fontsize=12)
    plt.ylabel("-Z'' (-Imaginary Impedance) [Ohm]", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Aspect ratio equal implies physical interpretation is strictly visual circle
    # But scales might differ significantly.
    plt.axis('equal')
    
    save_path = Path(__file__).parent / "nyquist_validation.png"
    plt.savefig(save_path)
    print(f"\nNyquist plot saved to {save_path}")
    print("Interpretation Guide:")
    print("1. Curve Shift: As SOH decreases (Red), curves should shift to the RIGHT (Higher Resistance).")
    print("2. Arc Size: The semi-circle arc (Charge Transfer) should generally GROW larger for aged batteries.")

if __name__ == "__main__":
    plot_nyquist_by_soh_group()
