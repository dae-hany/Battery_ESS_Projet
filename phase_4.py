import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class BatteryPrognostics:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.test_battery_id = 'B0018'
        self.train_battery_ids = ['B0005', 'B0006', 'B0007']
        # Key features selected from previous phases
        self.features = ['Rs', 'Rct', 'AUC_X', 'R_Mid_Avg', 'Relative_Rs', 'Relative_Rct', 'SOH_Trend', 'sigma', 'f_peak', 'Abs_Z_0.05Hz']

    def load_and_split_data(self):
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"{self.input_file} not found.")
        
        self.df = pd.read_csv(self.input_file)
        print(f"Data Loaded. Shape: {self.df.shape}")
        
        # Filter columns to ensure they exist
        existing_features = [f for f in self.features if f in self.df.columns]
        if len(existing_features) < len(self.features):
            missing = set(self.features) - set(existing_features)
            print(f"Warning: Missing features: {missing}")
        self.features = existing_features
        
        # Train/Test Split by Battery_ID
        train_mask = self.df['Battery_ID'].isin(self.train_battery_ids)
        test_mask = self.df['Battery_ID'] == self.test_battery_id
        
        train_df = self.df[train_mask]
        test_df = self.df[test_mask]
        
        # We only want to train on 'Original' data or Augmented?
        # Usually we train on augmented data to prevent overfitting, 
        # but validation/testing should ideally be on real 'Original' data to check real performance.
        # However, Task 1 says "Split the data... B0018 for independent testing". 
        # It implies using valid rows.
        # Let's use Augmented data for training to boost robustness, 
        # BUT for testing B0018, we should strictly use 'Original' augmentation type to evaluate real-world performance.
        
        train_df_final = train_df # Use all augmented data for training
        
        # For testing, filter only Original B0018 data to measure true accuracy
        if 'Augmentation_Type' in test_df.columns:
            test_df_final = test_df[test_df['Augmentation_Type'] == 'Original'].copy()
        else:
            test_df_final = test_df.copy()

        # Sort test data by time/cycle for plotting
        if 'time' in test_df_final.columns:
            test_df_final = test_df_final.sort_values('time')
        
        print(f"Training samples: {len(train_df_final)} ({self.train_battery_ids})")
        print(f"Test samples: {len(test_df_final)} ({self.test_battery_id})")
        
        self.X_train = train_df_final[self.features].values
        self.y_train = train_df_final['SOH'].values
        
        self.X_test = test_df_final[self.features].values
        self.y_test = test_df_final['SOH'].values
        
        # Scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train_soh_model(self):
        print("Training Gaussian Process Regressor...")
        # Kernel: Constant * RBF (Trend) + WhiteKernel (Noise)
        # RBF helps capture smooth non-linear trends in battery aging.
        # WhiteKernel handles the noise in EIS measurements.
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                 + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, alpha=0.1)
        self.model.fit(self.X_train_scaled, self.y_train)
        print(f"Model trained. Best kernel: {self.model.kernel_}")
        
        # Save model
        joblib.dump(self.model, 'soh_gpr_model.pkl')
        joblib.dump(self.scaler, 'soh_scaler.pkl')

    def evaluate_model(self):
        print("Evaluating on Test Battery (B0018)...")
        # Predict SOH and Uncertainty (std deviation)
        y_pred, y_std = self.model.predict(self.X_test_scaled, return_std=True)
        
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")
        
        return y_pred, y_std

    def visualize_predictions(self, y_pred, y_std):
        plt.figure(figsize=(12, 6))
        
        # X-axis: Cycle Index (or Time)
        x_axis = np.arange(len(self.y_test))
        
        plt.plot(x_axis, self.y_test, 'k-', label='True SOH', linewidth=2)
        plt.plot(x_axis, y_pred, 'r--', label='Predicted SOH', linewidth=2)
        
        # 95% Confidence Interval (1.96 * std)
        plt.fill_between(x_axis, y_pred - 1.96*y_std, y_pred + 1.96*y_std, 
                         color='red', alpha=0.2, label='95% Confidence Interval')
        
        plt.title(f'SOH Prediction for Test Battery {self.test_battery_id} (GPR)')
        plt.xlabel('EIS Cycle Index')
        plt.ylabel('State of Health (SOH) [%]')
        plt.legend()
        plt.grid(True)
        plt.savefig('phase_4_soh_prediction.png')
        print("SOH plot saved as phase_4_soh_prediction.png")
        # plt.show()

    def estimate_rul(self, y_pred):
        # RUL Estimation: Extrapolate SOH curve to end of life (usually 80% or end of data)
        # NASA dataset ends around 70-80% capacity for B0018.
        # We will fit a simple linear degradation on predicted values
        # and extrapolate if needed, or just compare trajectory.
        
        # Fit linear trend on predictions: SOH = m * cycle + c
        cycles = np.arange(len(y_pred))
        fit = np.polyfit(cycles, y_pred, 1)
        poly = np.poly1d(fit)
        
        # True trend
        fit_true = np.polyfit(cycles, self.y_test, 1)
        poly_true = np.poly1d(fit_true)
        
        print("\nRUL Estimation (Linear Trend Analysis):")
        print(f"Predicted Degradation Rate: {fit[0]:.4f} %SOH/cycle")
        print(f"True Degradation Rate:      {fit_true[0]:.4f} %SOH/cycle")
        
        # Plot RUL Trend
        plt.figure(figsize=(8, 5))
        plt.plot(cycles, self.y_test, 'k.', markersize=5, label='True SOH Points')
        plt.plot(cycles, poly_true(cycles), 'k-', alpha=0.5, label='True Trend')
        plt.plot(cycles, poly(cycles), 'r-', linewidth=2, label='Predicted Trend (RUL Basis)')
        
        # Threshold line
        min_soh = min(np.min(self.y_test), np.min(y_pred))
        plt.axhline(y=min_soh, color='gray', linestyle=':', label='End of Dataset')
        
        plt.title(f'RUL Trend Estimation ({self.test_battery_id})')
        plt.xlabel('Cycle')
        plt.ylabel('SOH [%]')
        plt.legend()
        plt.grid(True)
        plt.savefig('phase_4_rul_estimation.png')
        print("RUL plot saved as phase_4_rul_estimation.png")

if __name__ == "__main__":
    modeler = BatteryPrognostics("NASA_Augmented_EIS_Dataset.csv")
    modeler.load_and_split_data()
    modeler.train_soh_model()
    y_pred, y_std = modeler.evaluate_model()
    modeler.visualize_predictions(y_pred, y_std)
    modeler.estimate_rul(y_pred)
