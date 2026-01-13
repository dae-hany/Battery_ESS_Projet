import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class BreslowEstimator:
    """
    Breslow Estimator for Baseline Hazard and Survival Function.
    """
    def __init__(self):
        self.baseline_hazard = None
        self.cum_baseline_hazard = None
        self.unique_times = None
        
    def fit(self, risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray):
        """
        Fit the baseline hazard function.
        
        Args:
            risk_scores: Predicted log-hazard ratios (h_i)
            times: Observed times
            events: Event indicators (1=Event, 0=Censored)
        """
        # 1. Create DataFrame for processing
        df = pd.DataFrame({
            'time': times,
            'event': events,
            'risk_score': np.exp(risk_scores) # exp(h_i) is the risk score (theta)
        })
        
        # 2. Group by unique event times
        # Only consider times where events occurred
        event_times = df[df['event'] == 1]['time'].unique()
        event_times.sort() # T_1 < T_2 < ... < T_k
        
        baseline_hazards = []
        
        for t in event_times:
            # Risk Set R(t): Subjects who survived at least up to t
            risk_set = df[df['time'] >= t]
            
            # d_i: Number of events at time t
            d_i = df[(df['time'] == t) & (df['event'] == 1)].shape[0]
            
            # sum_theta: Sum of risk scores in the risk set
            sum_theta = risk_set['risk_score'].sum()
            
            # Baseline Hazard at t: d_i / sum_theta
            h_0 = d_i / sum_theta
            baseline_hazards.append(h_0)
            
        self.unique_times = event_times
        self.baseline_hazard = np.array(baseline_hazards)
        self.cum_baseline_hazard = np.cumsum(self.baseline_hazard)
        
        return self

    def get_survival_function(self, risk_scores: np.ndarray):
        """
        Calculate Survival Function S(t|x) for given risk scores.
        
        S(t|x) = exp(-H_0(t) * exp(h_x))
               = [exp(-H_0(t))] ^ exp(h_x)
               = [S_0(t)] ^ exp(h_x)
        
        Returns:
            pd.DataFrame: Index=Time, Columns=Samples, Values=Survival Probability
        """
        if self.cum_baseline_hazard is None:
            raise ValueError("Estimator must be fitted first.")
            
        # Baseline Survival S_0(t) = exp(-H_0(t))
        baseline_survival = np.exp(-self.cum_baseline_hazard)
        
        # Predict for each sample
        # risk_exp: exp(h_x)
        risk_exp = np.exp(risk_scores)
        
        # Result Matrix: (Time, Samples)
        # S(t|x) = S_0(t) ^ risk_exp
        # Log space: log S = log S_0 * risk_exp = -H_0 * risk_exp
        # S = exp(-H_0 * risk_exp)
        
        # Shape: (T,) x (N,) -> (T, N)
        H0_matrix = self.cum_baseline_hazard.reshape(-1, 1)
        risk_matrix = risk_exp.reshape(1, -1)
        
        surv_matrix = np.exp(-np.matmul(H0_matrix, risk_matrix))
        
        return pd.DataFrame(surv_matrix, index=self.unique_times)

def calculate_expected_rul(survival_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate Expected RUL from Survival Function.
    E[T] = Integral_0^inf S(t) dt
    
    Args:
        survival_df: Survival Function (Index=Time, Columns=Samples)
        
    Returns:
        np.ndarray: Expected RUL for each sample
    """
    times = survival_df.index.values
    
    # We need to integrate S(t) over t.
    # Use trapezoidal rule.
    # We usually treat S(0) = 1.
    
    # Prepend t=0, S=1
    times_ext = np.insert(times, 0, 0)
    
    # (T+1, N)
    surv_ext = np.vstack([np.ones(survival_df.shape[1]), survival_df.values])
    
    # Integration for each column
    expected_rul = np.trapz(surv_ext, x=times_ext, axis=0)
    
    return expected_rul

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))
