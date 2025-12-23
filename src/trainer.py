# -*- coding: utf-8 -*-
"""
DeepSurv 학습 및 평가 모듈 (Trainer Module)
===========================================================

이 모듈은 LSTM-DeepSurv 모델의 학습, 검증, 그리고 평가 로직을 담당합니다.
Cox Proportional Hazards Loss와 C-Index 계산 기능을 포함합니다.

주요 기능:
1. cox_ph_loss: 손실 함수
2. SurvivalMetrics: C-Index 등 평가 지표
3. DeepSurvTrainer: 학습 루프 및 조기 종료(Early Stopping) 관리

Author: ESS DeepSurv Research Team
Date: 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any

# ===================================================================================
# Loss Function & Metrics
# ===================================================================================

def cox_ph_loss(risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """
    Cox Proportional Hazards Negative Log Partial Likelihood.
    
    L = -Σ_{i: E_i=1} [h_i - log(Σ_{j: T_j >= T_i} exp(h_j))]
    
    Args:
        risk_scores: Predicted log-hazard ratios (Batch,)
        times: Survival times (Batch,)
        events: Event indicators (1=Uncensored, 0=Censored) (Batch,)
        
    Returns:
        Scalar Tensor (Loss)
    """
    # 1. 생존 시간(T) 기준 내림차순 정렬 (Cumulative Sum 계산 편의를 위해)
    idx = torch.argsort(times, descending=True)
    risk_sorted = risk_scores[idx]
    events_sorted = events[idx]
    
    # 2. Risk Score의 Log Cumulative Sum 계산 (Numerically Stable)
    # log(Σ exp(risk))
    log_cumsum = torch.logcumsumexp(risk_sorted, dim=0)
    
    # 3. 실제 이벤트가 발생한 샘플에 대해서만 Likelihood 계산
    # Partial Likelihood = risk_i - log_cumsum_i
    event_mask = events_sorted == 1
    
    if event_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
        
    log_likelihood = risk_sorted[event_mask] - log_cumsum[event_mask]
    
    # Negative Log Likelihood (Minimize this)
    return -log_likelihood.mean()

class SurvivalMetrics:
    """생존 분석 평가 지표 클래스"""
    
    @staticmethod
    def concordance_index(risk_scores: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        """
        Harrell's C-Index (Concordance Index) 계산.
        
        Args:
            risk_scores: 예측된 위험 점수 (높을수록 위험)
            times: 실제 생존 시간
            events: 이벤트 발생 여부 (1=발생)
            
        Returns:
            float: C-Index (0.5=Random, 1.0=Perfect)
        """
        n = len(times)
        concordant = 0
        permissible = 0
        
        # 모든 쌍(Pair)에 대해 비교
        for i in range(n):
            if events[i] == 1: # Uncensored 샘플만 기준이 될 수 있음
                for j in range(n):
                    if i != j:
                        # i가 j보다 먼저 이벤트가 발생했어야 비교 가능 (T_i < T_j)
                        if times[i] < times[j]:
                            permissible += 1
                            # 모델이 i의 위험도를 더 높게 예측했으면 올바른 예측 (Risk_i > Risk_j)
                            if risk_scores[i] > risk_scores[j]:
                                concordant += 1
                            elif risk_scores[i] == risk_scores[j]:
                                concordant += 0.5
                                
        return concordant / permissible if permissible > 0 else 0.5

# ===================================================================================
# Trainer Structure
# ===================================================================================

class DeepSurvTrainer:
    """DeepSurv 모델 학습 관리 클래스"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', lr: float = 1e-3, 
                 weight_decay: float = 1e-4, patience: int = 20):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.patience = patience
        
        # History 기록
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_cindex': [], 'val_cindex': []
        }
        
    def train_one_epoch(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """한 Epoch 학습 진행"""
        self.model.train()
        total_loss = 0
        
        # C-index 계산을 위한 리스트
        all_risks = []
        all_times = []
        all_events = []
        
        for batch_x, batch_t, batch_e in data_loader:
            batch_x = batch_x.to(self.device)
            batch_t = batch_t.to(self.device)
            batch_e = batch_e.to(self.device)
            
            # Forward
            risk_pred = self.model(batch_x)
            
            # Loss Calculation
            # Cox Loss는 이벤트가 하나도 없는 배치에서는 0이 될 수 있음
            if batch_e.sum() > 0:
                loss = cox_ph_loss(risk_pred, batch_t, batch_e)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Store for metrics
            all_risks.append(risk_pred.detach().cpu().numpy())
            all_times.append(batch_t.cpu().numpy())
            all_events.append(batch_e.cpu().numpy())
            
        # Epoch Metrics
        avg_loss = total_loss / len(data_loader)
        
        # Concatenate all batches
        if len(all_risks) > 0:
            flat_risks = np.concatenate(all_risks)
            flat_times = np.concatenate(all_times)
            flat_events = np.concatenate(all_events)
            
            c_index = SurvivalMetrics.concordance_index(flat_risks, flat_times, flat_events)
        else:
            c_index = 0.5
            
        return {'loss': avg_loss, 'c_index': c_index}
    
    def validate(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """검증 데이터셋 평가"""
        self.model.eval()
        total_loss = 0
        
        all_risks = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch_x, batch_t, batch_e in data_loader:
                batch_x = batch_x.to(self.device)
                batch_t = batch_t.to(self.device)
                batch_e = batch_e.to(self.device)
                
                risk_pred = self.model(batch_x)
                
                if batch_e.sum() > 0:
                    loss = cox_ph_loss(risk_pred, batch_t, batch_e)
                    total_loss += loss.item()
                
                all_risks.append(risk_pred.cpu().numpy())
                all_times.append(batch_t.cpu().numpy())
                all_events.append(batch_e.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        
        if len(all_risks) > 0:
            flat_risks = np.concatenate(all_risks)
            flat_times = np.concatenate(all_times)
            flat_events = np.concatenate(all_events)
            c_index = SurvivalMetrics.concordance_index(flat_risks, flat_times, flat_events)
        else:
            c_index = 0.5
            
        return {'loss': avg_loss, 'c_index': c_index}
    
    def fit(self, train_loader, val_loader, epochs: int = 100, verbose: int = 10):
        """
        전체 학습 실행 (Early Stopping 포함).
        """
        best_val_cindex = -1.0
        patience_counter = 0
        best_state_dict = None
        
        print(f"\n[Training Started] Epochs: {epochs}, Device: {self.device}")
        print("-" * 70)
        print(f"{'Epoch':^10} | {'Train Loss':^12} | {'Val Loss':^12} | {'Train C-Idx':^12} | {'Val C-Idx':^12}")
        print("-" * 70)
        
        for epoch in range(1, epochs + 1):
            # 1. Train
            train_metrics = self.train_one_epoch(train_loader)
            
            # 2. Validate
            val_metrics = self.validate(val_loader)
            
            # 3. Record History
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_cindex'].append(train_metrics['c_index'])
            self.history['val_cindex'].append(val_metrics['c_index'])
            
            # 4. Print Log
            if epoch % verbose == 0 or epoch == 1:
                print(f"{epoch:^10} | {train_metrics['loss']:^12.4f} | {val_metrics['loss']:^12.4f} | "
                      f"{train_metrics['c_index']:^12.4f} | {val_metrics['c_index']:^12.4f}")
            
            # 5. Early Stopping Check
            # C-Index가 높을수록 좋음
            if val_metrics['c_index'] > best_val_cindex:
                best_val_cindex = val_metrics['c_index']
                best_state_dict = self.model.state_dict()
                patience_counter = 0 # Reset
                # Save best model temporarily in memory
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"\n[Early Stopping] No improvement for {self.patience} epochs.")
                print(f"Best Validation C-Index: {best_val_cindex:.4f}")
                break
        
        # Restore best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            
        return self.history
