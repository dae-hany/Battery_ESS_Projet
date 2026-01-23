import pandas as pd
import numpy as np
import os

class DataAugmenter:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None
        self.augmented_df = None

    def load_data(self):
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"{self.input_file} not found.")
        self.df = pd.read_csv(self.input_file)
        print(f"Data Loaded. Shape: {self.df.shape}")

    def clean_data(self):
        if self.df is None: return

        initial_len = len(self.df)
        
        # 1. Phi_min Cleaning
        # Filter out Phi_min < -1.57 radians (approx -90 degrees) or > 0 (inductive/capacitive noise)
        # However, Phi can be slightly positive due to noise or wire inductance at high freq, 
        # but Phi_min tracks the capacitive loop which should be negative. 
        # Extreme negative values (<-1.6) are likely phase wrapping errors or noise.
        # User requested: filter physically impossible values (e.g. < -1.57)
        # Let's replace them with NaN and then interpolate or just clip. User said "Filter out".
        # Let's strictly filter out bad rows or just interpolate the value?
        # "Filter out" usually means remove rows. But let's try to preserve data by interpolation first if possible,
        # or just drop if it's too bad. Let's drop rows for reliability.
        valid_phi = (self.df['Phi_min'] >= -np.pi/2 - 0.1) & (self.df['Phi_min'] <= 0.1)
        self.df = self.df[valid_phi].copy()
        
        print(f"Phi_min filter dropped {initial_len - len(self.df)} rows.")

        # 2. Rct Cleaning (Moving Average for stuck values)
        # If Rct is constant (e.g. due to previous default fill), smooth it.
        # We did some interpolation in phase 2, but let's apply a general 3-point MA 
        # to ensure smoothness and fix any remaining artifacts.
        self.df['Rct'] = self.df.groupby('Battery_ID')['Rct'].transform(
            lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
        )

    def feature_engineering(self):
        # 1. R_Mid_Avg: Average of Real part between 10Hz and 400Hz
        mid_freqs = [10.0, 20.0, 40.0, 100.0, 200.0, 400.0]
        mid_cols = [f'R_{freq}Hz' for freq in mid_freqs]
        
        # Check which columns exist
        existing_cols = [col for col in mid_cols if col in self.df.columns]
        
        if existing_cols:
            self.df['R_Mid_Avg'] = self.df[existing_cols].mean(axis=1)
        else:
            self.df['R_Mid_Avg'] = np.nan
        
        # Ensure 'AUC_X', 'Rs' exist (created in Phase 2)
        if 'AUC_X' not in self.df.columns:
            print("Warning: AUC_X column missing.")
            self.df['AUC_X'] = 0
            
    def augment_data(self):
        # Generate 3x synthetic samples
        # Strategy: Create 3 copies with different perturbations
        
        dfs_to_concat = [self.df.copy()] # Original data
        
        # Define features to perturb
        # Rs, Rct, AUC_X, R_Mid_Avg are primary.
        # Also perturb other R_ and X_ columns to keep consistency if needed?
        # User prompt only mentions specific impedance features but it's better to perturb the derived ones or the trend ones.
        # Let's perturb the key features used for training.
        
        keys_to_perturb = ['Rs', 'Rct', 'AUC_X', 'R_Mid_Avg'] # derived
        # Also need to perturb raw columns if we want to save a full dataset that looks consistent,
        # but usually for ML we just care about the training features. 
        # Let's assume the output is for ML and perturb the key features.
        
        # Scenario 1: White Noise Injection (1-2% noise)
        df_noise = self.df.copy()
        for col in keys_to_perturb:
            if col in df_noise.columns:
                noise = np.random.normal(loc=0, scale=0.015, size=len(df_noise)) # 1.5% mean noise
                df_noise[col] = df_noise[col] * (1 + noise)
        
        df_noise['Augmentation_Type'] = 'White_Noise'
        dfs_to_concat.append(df_noise)
        
        # Scenario 2: Temperature Perturbation (+/- 2 deg C -> +/- 3% Rs)
        # Arrhenius: R decreases as T increases.
        # Simulating T variation means R varies.
        # Let's randomly apply +/- 3% shift
        df_temp = self.df.copy()
        
        # Random vector for +/- 3%
        temp_effect = np.random.uniform(-0.03, 0.03, size=len(df_temp))
        
        # Rs is directly affected. Rct is also affected usually more strongly, but let's apply provided scaling.
        # "affecting Rs by +/- 3%"
        if 'Rs' in df_temp.columns:
            df_temp['Rs'] = df_temp['Rs'] * (1 + temp_effect)
        
        # Assuming R_Mid_Avg also scales similarly as it's ohmic+CT
        if 'R_Mid_Avg' in df_temp.columns:
            df_temp['R_Mid_Avg'] = df_temp['R_Mid_Avg'] * (1 + temp_effect)

        df_temp['Augmentation_Type'] = 'Temp_Perturbation'
        dfs_to_concat.append(df_temp)
        
        # Scenario 3: Capacity Jitter (0.5% noise to SOH)
        df_cap = self.df.copy()
        noise_soh = np.random.normal(loc=0, scale=0.005, size=len(df_cap))
        df_cap['SOH'] = df_cap['SOH'] * (1 + noise_soh)
        
        df_cap['Augmentation_Type'] = 'Capacity_Jitter'
        dfs_to_concat.append(df_cap)
        
        # Concatenate all
        self.augmented_df = pd.concat(dfs_to_concat, ignore_index=True)
        
        # Fill Augmentation_Type for original
        self.augmented_df['Augmentation_Type'] = self.augmented_df['Augmentation_Type'].fillna('Original')

    def save_data(self, output_file):
        if self.augmented_df is not None:
            self.augmented_df.to_csv(output_file, index=False)
            print(f"Saved augmented dataset to {output_file}")
            print(f"Final Shape: {self.augmented_df.shape}")
            print("First 5 rows:")
            print(self.augmented_df[['Battery_ID', 'SOH', 'Rs', 'Rct', 'Augmentation_Type']].head())

if __name__ == "__main__":
    augmenter = DataAugmenter("NASA_Featured_EIS_Dataset.csv")
    augmenter.load_data()
    augmenter.clean_data()
    augmenter.feature_engineering()
    augmenter.augment_data()
    augmenter.save_data("NASA_Augmented_EIS_Dataset.csv")
