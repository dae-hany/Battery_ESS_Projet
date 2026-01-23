import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class FeatureExtractor:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None
        
    def load_data(self):
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"{self.input_file} not found.")
        self.df = pd.read_csv(self.input_file)
        print(f"Data Loaded. Shape: {self.df.shape}")

    def extract_features(self):
        if self.df is None:
            return
        
        # 1. Refine Rs (Ohmic Resistance)
        # Find frequency where abs(X) is minimum among high frequencies (>= 100Hz)
        high_freqs = [100.0, 200.0, 400.0, 1000.0]
        X_high_cols = [f'X_{f}Hz' for f in high_freqs]
        R_high_cols = [f'R_{f}Hz' for f in high_freqs]

        # Check existing columns
        valid_high_indices = [i for i, col in enumerate(X_high_cols) if col in self.df.columns and R_high_cols[i] in self.df.columns]
        
        if valid_high_indices:
            X_high_data = self.df[[X_high_cols[i] for i in valid_high_indices]].values
            R_high_data = self.df[[R_high_cols[i] for i in valid_high_indices]].values
            
            # Find index where abs(X) is minimum
            min_abs_X_idx = np.argmin(np.abs(X_high_data), axis=1)
            
            # Get Rs using that index
            row_indices = np.arange(len(self.df))
            self.df['Rs'] = R_high_data[row_indices, min_abs_X_idx]
        else:
            self.df['Rs'] = np.nan

        # 2. Refine Rct (Charge Transfer Resistance)
        # Semicircle peak: min X (most negative) in [10Hz, 500Hz] i.e. 10, 20, 40, 100, 200, 400
        mid_freqs = [10.0, 20.0, 40.0, 100.0, 200.0, 400.0]
        X_mid_cols = [f'X_{f}Hz' for f in mid_freqs]
        R_mid_cols = [f'R_{f}Hz' for f in mid_freqs]
        
        valid_mid_indices = [i for i, col in enumerate(X_mid_cols) if col in self.df.columns and R_mid_cols[i] in self.df.columns]
        
        if valid_mid_indices:
            X_mid_data = self.df[[X_mid_cols[i] for i in valid_mid_indices]].values
            R_mid_data = self.df[[R_mid_cols[i] for i in valid_mid_indices]].values
            
            # Find index of most negative X (peak of semicircle)
            min_X_idx = np.argmin(X_mid_data, axis=1)
            R_peak = R_mid_data[row_indices, min_X_idx]
            
            # Calculate provisional Rct
            Rct_provisional = R_peak - self.df['Rs']
            
            # Correction for negative values or very small values
            # If provisional Rct is too small or negative, it usually means 
            # the semi-circle is very flat or the Rs estimation point is weird.
            # In many cases for Li-ion, Rct grows as SOH drops.
            # Let's try to enforce a more robust check.
            # If Rct <= 0, we can take the difference between R_at_lowest_mid_freq and Rs
            
            R_lowest_mid = R_mid_data[:, 0] # 10Hz or similar
            Rct_fallback = R_lowest_mid - self.df['Rs']
            
            # Vectorized conditional assignment
            final_Rct = np.where(Rct_provisional > 1e-4, Rct_provisional, Rct_fallback)
            final_Rct = np.where(final_Rct > 1e-4, final_Rct, 0.01) # Final cliff
            
            self.df['Rct'] = final_Rct
            
            self.df['f_peak'] = [mid_freqs[valid_mid_indices[i]] for i in min_X_idx]
        else:
            self.df['Rct'] = np.nan
            self.df['f_peak'] = np.nan

        # 3. Warburg Coefficient (sigma)
        # Slope of R vs omega^(-0.5) at low freqs (0.05, 0.1, 0.2, 0.4)
        low_freqs = [0.05, 0.1, 0.2, 0.4]
        omega_inv_sqrt = np.array([1.0 / np.sqrt(2 * np.pi * f) for f in low_freqs])
        
        R_low_cols = [f'R_{f}Hz' for f in low_freqs]
        
        # Only proceed if columns exist
        valid_low = [col for col in R_low_cols if col in self.df.columns]
        if len(valid_low) == len(low_freqs):
            R_low_vals = self.df[valid_low].values
            
            # Vectorized linear regression for calculating slope
            # slope = (N * sum(xy) - sum(x)sum(y)) / (N * sum(x^2) - (sum(x))^2)
            # Here x is fixed (omega_inv_sqrt)
            x = omega_inv_sqrt
            N = len(x)
            sum_x = np.sum(x)
            sum_x2 = np.sum(x**2)
            denom = N * sum_x2 - sum_x**2
            
            sum_y = np.sum(R_low_vals, axis=1)
            sum_xy = np.dot(R_low_vals, x) # sum(y_i * x_i) for each row
            
            slope = (N * sum_xy - sum_x * sum_y) / denom
            self.df['sigma'] = slope
        else:
             self.df['sigma'] = np.nan

        # 4. Phase Minimum (Phi_min)
        # theta = arctan2(X, R) for all points
        all_freqs = [0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0, 20.0, 40.0, 100.0, 200.0, 400.0, 1000.0]
        
        phase_min_list = []
        for index, row in self.df.iterrows():
            phases = []
            for f in all_freqs:
                r_col = f'R_{f}Hz'
                x_col = f'X_{f}Hz'
                if r_col in row and x_col in row:
                    phi = np.arctan2(row[x_col], row[r_col])
                    phases.append(phi)
            if phases:
                phase_min_list.append(np.min(phases))
            else:
                phase_min_list.append(np.nan)
        
        self.df['Phi_min'] = phase_min_list

        # 5. Impedance Magnitude and Sum features
        # sqrt(R^2 + X^2) at 0.05Hz
        if 'R_0.05Hz' in self.df.columns and 'X_0.05Hz' in self.df.columns:
            self.df['Abs_Z_0.05Hz'] = np.sqrt(self.df['R_0.05Hz']**2 + self.df['X_0.05Hz']**2)
        
        # X_Sum: Sum of all imaginary parts (renamed to reflect request, although AUC_X is better)
        x_cols = [col for col in self.df.columns if col.startswith('X_')]
        if x_cols:
            self.df['X_Sum'] = self.df[x_cols].sum(axis=1)

        # New Geometric Features
        # AUC_X: Integral of Imaginary part between 1Hz and 1000Hz (Trapezoidal rule)
        # Using abs(X) because X is generally negative
        auc_freqs = [1.0, 2.0, 4.0, 10.0, 20.0, 40.0, 100.0, 200.0, 400.0, 1000.0]
        valid_auc_indices = [i for i, f in enumerate(auc_freqs) if f'X_{f}Hz' in self.df.columns]
        
        if len(valid_auc_indices) > 1:
            # We need to integrate wrt log(freq) or just freq? 
            # Usually AUC in Nyquist plot means integrating -Im(Z) vs Re(Z) or similar.
            # But prompt says "Integral of the Imaginary part (AUC_X) between 1Hz and 1000Hz".
            # This likely means area under the curve of X vs Frequency (or Log Frequency).
            # Given it's "growth of impedance semicircle", usually this is related to total charge transfer capability.
            # Let's simple sum of abs(X) normalized by number of points or trapezoidal integration.
            # For robustness, let's simply take the sum of absolute X values in this range.
            # Or better, trapezoidal integration over log10(frequency).
            
            x_vals = []
            y_vals = []
            for f in auc_freqs:
                 col = f'X_{f}Hz'
                 if col in self.df.columns:
                     x_vals.append(np.log10(f))
                     y_vals.append(self.df[col].values)
            
            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals).T # Shape (N_samples, N_freqs)
            
            # Use abs(X) because X is negative
            y_vals = np.abs(y_vals)
            
            # Composite trapezoidal rule
            # Integral = sum(0.5 * (y_{i} + y_{i+1}) * (x_{i+1} - x_{i}))
            # x is sorted? auc_freqs is sorted 1 -> 1000.
            
            auc_list = []
            for i in range(len(x_vals) - 1):
                dx = x_vals[i+1] - x_vals[i]
                y_avg = (y_vals[:, i] + y_vals[:, i+1]) / 2.0
                auc_list.append(y_avg * dx)
            
            self.df['AUC_X'] = np.sum(auc_list, axis=0)
        else:
            self.df['AUC_X'] = np.nan
            
        # Euclidean Distance from origin to 0.05Hz point
        if 'R_0.05Hz' in self.df.columns and 'X_0.05Hz' in self.df.columns:
             self.df['Z_magnitude_low'] = np.sqrt(self.df['R_0.05Hz']**2 + self.df['X_0.05Hz']**2)
        else:
             self.df['Z_magnitude_low'] = np.nan


        # 6. Per-Battery Normalization
        # Sort by battery and time first
        self.df.sort_values(by=['Battery_ID', 'time'], inplace=True)
        
        # Calculate Relative Features
        # For each battery, get the first valid Rs and Rct
        self.df['Relative_Rs'] = np.nan
        self.df['Relative_Rct'] = np.nan
        
        for bat_id in self.df['Battery_ID'].unique():
            bat_mask = self.df['Battery_ID'] == bat_id
            
            # Handle Rct outliers before normalization: Linear Interpolation/Moving Average
            # Replace 0.01 (the previous cliff value) or NaN with proper values
            # Assuming 'Rct' has been filled with 0.01 in very bad cases which acts as outlier.
            # But 'final_Rct' logic in previous step was: > 1e-4 check.
            # Let's replace very small values with NaN and then interpolate
            
            subset = self.df.loc[bat_mask].copy()
            
            # Clean Rs: Rolling mean for SOH_Trend
            subset['SOH_Trend'] = subset['Rs'].rolling(window=5, min_periods=1, center=True).mean()
            
            # Clean Rct
            subset['Rct'] = subset['Rct'].replace(0.01, np.nan)
            subset['Rct'] = subset['Rct'].interpolate(method='linear', limit_direction='both')
            # If still NaN (e.g. all bad), ffill/bfill
            subset['Rct'] = subset['Rct'].fillna(method='ffill').fillna(method='bfill')
            
            
            # Get initial values (Use mean of first 3 cycles for robustness)
            if len(subset) > 0:
                # Robust Initial Rs
                init_Rs_series = subset['Rs'].iloc[:3]
                init_Rs = init_Rs_series.mean() if not init_Rs_series.empty else subset['Rs'].iloc[0]
                
                # Robust Initial Rct
                init_Rct_series = subset['Rct'].iloc[:3]
                init_Rct = init_Rct_series.mean() if not init_Rct_series.empty else subset['Rct'].iloc[0]
                
                # Check 0 division
                if pd.isna(init_Rs) or init_Rs == 0: init_Rs = 1e-6
                if pd.isna(init_Rct) or init_Rct == 0: init_Rct = 1e-6 # fallback
                
                subset['Relative_Rs'] = subset['Rs'] / init_Rs
                subset['Relative_Rct'] = subset['Rct'] / init_Rct
            
            self.df.loc[bat_mask, 'SOH_Trend'] = subset['SOH_Trend']
            self.df.loc[bat_mask, 'Rct'] = subset['Rct']
            self.df.loc[bat_mask, 'Relative_Rs'] = subset['Relative_Rs']
            self.df.loc[bat_mask, 'Relative_Rct'] = subset['Relative_Rct']

    
    def cleanse_features(self):
        # Handle Inf/Nan that might have been produced
        initial_count = len(self.df)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Optional: Drop rows with NaNs in key features?
        # Typically we want to keep data, maybe fill?
        # User requirement: "interpolating or removing outliers"
        # Let's drop rows where any new feature is NaN (implies severe data missing)
        feature_cols = ['Rs', 'Rct', 'sigma', 'Phi_min', 'f_peak']
        self.df.dropna(subset=feature_cols, inplace=True)
        print(f"Cleaned data. Rows dropped: {initial_count - len(self.df)}")

    def save_data(self, output_file):
        self.df.to_csv(output_file, index=False)
        print(f"Saved featured dataset to {output_file}")

    def visualize(self):
        # Plot SOH vs Relative_Rs and SOH vs Relative_Rct
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        # SOH vs Relative Rs
        for bat_id in self.df['Battery_ID'].unique():
            subset = self.df[self.df['Battery_ID'] == bat_id]
            plt.scatter(subset['Relative_Rs'], subset['SOH'], alpha=0.5, label=bat_id)
            
        plt.xlabel('Relative Ohmic Resistance (Current/Initial)')
        plt.ylabel('SOH [%]')
        plt.title('SOH vs Relative Rs')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # SOH vs Relative Rct
        for bat_id in self.df['Battery_ID'].unique():
            subset = self.df[self.df['Battery_ID'] == bat_id]
            plt.scatter(subset['Relative_Rct'], subset['SOH'], alpha=0.5, label=bat_id)
            
        plt.xlabel('Relative Charge Transfer Resistance (Current/Initial)')
        plt.ylabel('SOH [%]')
        plt.title('SOH vs Relative Rct')
        plt.legend()
        plt.grid(True)
        # plt.xlim(left=0) # Relative can be 0 -> infinity
        
        plt.tight_layout()
        plt.savefig('phase_2_correlation_plot.png')
        print("Visualization saved as phase_2_correlation_plot.png")
        
        # Calculate Pearson correlations
        feature_list = ['Rs', 'Rct', 'Relative_Rs', 'Relative_Rct', 'sigma', 'Phi_min', 'f_peak', 'Abs_Z_0.05Hz', 'AUC_X', 'Z_magnitude_low', 'SOH_Trend']
        valid_features = [f for f in feature_list if f in self.df.columns]
        
        print("\nCorrelation with SOH:")
        correlations = self.df[valid_features + ['SOH']].corr()['SOH'].sort_values(ascending=False)
        print(correlations)

if __name__ == "__main__":
    extractor = FeatureExtractor("NASA_23pt_EIS_Dataset.csv")
    extractor.load_data()
    extractor.extract_features()
    extractor.cleanse_features()
    extractor.save_data("NASA_Featured_EIS_Dataset.csv")
    extractor.visualize()
    
    print("\nSample Data:")
    print(extractor.df[['Battery_ID', 'SOH', 'Rs', 'Rct', 'sigma', 'Phi_min']].head())
