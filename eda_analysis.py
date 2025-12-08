import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
BASE_DIR = r"c:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
METADATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset", "metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "cleaned_dataset", "data")

def load_and_plot_data():
    print("Loading metadata...")
    df = pd.read_csv(METADATA_PATH)
    
    # Filter for discharge cycles
    discharge_df = df[df['type'] == 'discharge'].copy()
    
    # Convert Capacity to numeric, coercing errors to NaN
    discharge_df['Capacity'] = pd.to_numeric(discharge_df['Capacity'], errors='coerce')
    
    # Drop rows with NaN Capacity
    discharge_df = discharge_df.dropna(subset=['Capacity'])
    
    # Convert start_time to datetime if needed, or just use index as proxy for time/cycle
    # The 'start_time' column seems to be a string representation of a list/array, which is messy.
    # However, 'test_id' or just the order of appearance might be sufficient for ordering.
    
    # Get unique batteries
    batteries = discharge_df['battery_id'].unique()
    print(f"Found batteries: {batteries}")
    
    plt.figure(figsize=(12, 8))
    
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
        
        # Plot Capacity
        plt.plot(range(cycles), bat_df['Capacity'], label=battery)
    
    stats_df = pd.DataFrame(stats)
    print("\nBattery Statistics:")
    print(stats_df.to_string())
    
    stats_path = os.path.join(BASE_DIR, "battery_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Statistics saved to {stats_path}")
    
    # Check for outliers (e.g., very low cycles or weird capacity)
    print("\nPotential Outliers (Cycles < 50 or Init_Cap < 1.0):")
    outliers = stats_df[(stats_df['Cycles'] < 50) | (stats_df['Init_Cap'] < 1.0)]
    print(outliers.to_string())
        
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (Ah)')
    plt.title('Battery Capacity Degradation over Cycles')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(BASE_DIR, "capacity_trends.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    load_and_plot_data()
