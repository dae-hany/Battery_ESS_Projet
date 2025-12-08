import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
BASE_DIR = r"c:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
METADATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset", "metadata.csv")
DATA_DIR = os.path.join(BASE_DIR, "cleaned_dataset", "data")

def plot_covariates(battery_id):
    print(f"Analyzing covariates for {battery_id}...")
    df = pd.read_csv(METADATA_PATH)
    
    # Filter for discharge cycles of the specific battery
    bat_df = df[(df['battery_id'] == battery_id) & (df['type'] == 'discharge')].copy()
    
    # Sort by test_id
    bat_df = bat_df.sort_values('test_id')
    
    if len(bat_df) == 0:
        print(f"No discharge data found for {battery_id}")
        return

    # Select 3 cycles: First, Middle, Last
    cycles_to_plot = [0, len(bat_df)//2, len(bat_df)-1]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    for i, idx in enumerate(cycles_to_plot):
        row = bat_df.iloc[idx]
        filename = row['filename']
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        try:
            # Load cycle data
            cycle_data = pd.read_csv(file_path)
            
            # Check columns. Usually: Voltage_measured, Current_measured, Temperature_measured, Time
            # Adjust column names if necessary based on actual file content
            # We'll assume standard NASA format first, but print columns if fail
            
            # Plot Voltage
            axes[0].plot(cycle_data['Time'], cycle_data['Voltage_measured'], label=f'Cycle {idx+1}')
            axes[0].set_ylabel('Voltage (V)')
            axes[0].set_title(f'{battery_id} - Voltage Curves')
            
            # Plot Current
            axes[1].plot(cycle_data['Time'], cycle_data['Current_measured'], label=f'Cycle {idx+1}')
            axes[1].set_ylabel('Current (A)')
            axes[1].set_title(f'{battery_id} - Current Curves')
            
            # Plot Temperature
            axes[2].plot(cycle_data['Time'], cycle_data['Temperature_measured'], label=f'Cycle {idx+1}')
            axes[2].set_ylabel('Temperature (C)')
            axes[2].set_title(f'{battery_id} - Temperature Curves')
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            # Try to print columns if possible
            try:
                print(f"Columns in {filename}: {pd.read_csv(file_path).columns.tolist()}")
            except:
                pass

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[2].set_xlabel('Time (s)')
    
    output_path = os.path.join(BASE_DIR, f"covariates_{battery_id}.png")
    plt.savefig(output_path)
    print(f"Covariate plot saved to {output_path}")

if __name__ == "__main__":
    # Analyze a representative "good" battery
    plot_covariates('B0005')
    # Maybe another one to compare
    plot_covariates('B0047')
