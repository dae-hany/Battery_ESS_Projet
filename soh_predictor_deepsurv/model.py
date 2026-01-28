import torch
import torch.nn as nn
import numpy as np

class DeepSurv(nn.Module):
    """
    DeepSurv Model: MLP for Cox Proportional Hazards Model
    Predicts Log Hazard Ratio (Risk Score)
    """
    def __init__(self, in_features, hidden_layers=[64, 32], dropout=0.2):
        super(DeepSurv, self).__init__()
        
        layers = []
        prev_dim = in_features
        
        for dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        # Output layer (1 dimension for log hazard)
        layers.append(nn.Linear(prev_dim, 1))
        # No activation for log hazard (can be any real number)
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def cox_ph_loss(log_h, events, eps=1e-7):
    """
    Cox Proportional Hazards Loss (Negative Log Partial Likelihood)
    
    Args:
        log_h: Log Hazard (Network Output), Shape (N, 1)
        events: Event Indicator (1 if event occurred, 0 if censored), Shape (N, 1)
        
    Note:
        Input must be sorted by Time (Duration) in DESCENDING order.
        Here, we use SOH as Time proxy, so sorted by SOH DESCENDING.
    """
    # log_h should be (N, 1)
    if log_h.dim() == 1:
        log_h = log_h.view(-1, 1)
    if events.dim() == 1:
        events = events.view(-1, 1)
        
    # Risk Set Calculation
    # Since data is sorted by time desc, for sample i, the risk set is R_i = {0, 1, ..., i} ?
    # No. 
    # Definition: Risk set R(t_i) contains all subjects who survived at least until t_i.
    # If sorted by time T descending: T_0 >= T_1 >= ...
    # At time T_i, subjects 0, 1, ..., i have times >= T_i. Use they are in risk set.
    # Wait.
    # If T_0 is largest (longest survival), then at T_0, only 0 is alive? No. Everyone is alive at T_small.
    # At T_large, fewer people are alive.
    # So Risk set at T_i (where T_i is large) is smaller?
    # Let's verify.
    # R(t) = {j : T_j >= t}.
    # Sorted Descending: T_0 >= T_1 ...
    # R(T_0) = {j : T_j >= T_0} = {0} (assuming unique).
    # R(T_k) = {j : T_j >= T_k} = {0, 1, ..., k}.
    # So for sample i (Time T_i), the risk set is {0, 1, ..., i}.
    
    # Partial Likelihood L = product ( exp(theta * x_i) / sum_{j in R(T_i)} exp(theta * x_j) ) ^ E_i
    # Log L = sum E_i * ( theta * x_i - log( sum_{j in R(T_i)} exp(theta * x_j) ) )
    #       = sum E_i * ( log_h[i] - log_risk[i] )
    
    # We need cumsum of exp(log_h).
    # Since R(T_i) = {0, ..., i}, we need sum from 0 to i.
    # This corresponds to cumsum if we iterate forward.
    
    exp_h = torch.exp(log_h)
    
    # cumulative sum of exp risk
    cum_risk = torch.cumsum(exp_h, dim=0) # [exp(h0), exp(h0)+exp(h1), ...]
    
    log_cum_risk = torch.log(cum_risk + eps)
    
    # Only calculate loss for events
    loss = events * (log_h - log_cum_risk)
    
    # Negative Log Likelihood
    return -torch.mean(loss)
