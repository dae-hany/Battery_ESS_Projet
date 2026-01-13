# -*- coding: utf-8 -*-
"""
Battery ESS Project - Main Execution Pipeline
===========================================================

이 스크립트는 LSTM-DeepSurv 모델의 전체 학습 파이프라인을 실행합니다.
데이터 로드 -> 전처리 -> 모델 학습 -> 결과 저장 -> 시각화 단계를 수행합니다.

Author: ESS DeepSurv Research Team
Date: 2025
"""

import os
import sys
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# 프로젝트 루트 경로를 sys.path에 추가하여 src 모듈 임포트 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 커스텀 모듈 임포트
from src.data_loader import DataConfig, BatteryDataProcessor, get_dataloaders
from src.models import LSTMDeepSurv
from src.trainer import DeepSurvTrainer
from src.metrics import BreslowEstimator, calculate_expected_rul, calculate_rmse

# ===================================================================================
# Configuration
# ===================================================================================

@dataclass
class ExperimentConfig:
    """실험 하이퍼파라미터 및 설정"""
    # Data Config
    window_size: int = 10
    batch_size: int = 32
    
    # Model Config
    input_dim: int = 4 # capacity, soh, derivatives... (자동 감지되지만 기본값 설정)
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    
    # Training Config
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 20
    weight_decay: float = 1e-4
    
    # Paths
    base_dir: str = r"C:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
    results_dir: str = "results"
    plots_dir: str = "plots"

# ===================================================================================
# Visualization Helper
# ===================================================================================

def plot_training_history(history, save_path):
    """학습 History (Loss, C-Index) 시각화 및 저장"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Cox PH)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. C-Index Curve
    ax2.plot(epochs, history['train_cindex'], 'b-', label='Train C-Index')
    ax2.plot(epochs, history['val_cindex'], 'r--', label='Val C-Index')
    ax2.set_title('Concordance Index (C-Index)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('C-Index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Info] Training history plot saved to: {save_path}")
    plt.close()

# ===================================================================================
# Main Pipeline
# ===================================================================================

def run_pipeline():
    # 1. 설정 초기화
    exp_config = ExperimentConfig()
    
    # 결과 디렉토리 생성
    results_dir = os.path.join(exp_config.base_dir, exp_config.results_dir)
    plots_dir = os.path.join(exp_config.base_dir, exp_config.plots_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Info] Running on Device: {device}")
    
    # 2. 데이터 로드 및 전처리
    print("\n[Step 1] Loading Data...")
    data_config = DataConfig(
        base_dir=exp_config.base_dir, 
        window_size=exp_config.window_size,
        feature_set_name='advanced' # Path Signature Enhanced
    )
    
    train_loader, val_loader = get_dataloaders(data_config, batch_size=exp_config.batch_size)
    
    # 입력 차원 자동 확인 (One batch fetch)
    sample_batch = next(iter(train_loader))[0]
    input_dim = sample_batch.shape[-1]
    print(f"[Info] Detected Input Dimension: {input_dim}")
    print(f"[Info] Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}")
    
    # 3. 모델 초기화
    print("\n[Step 2] Initializing Model...")
    model = LSTMDeepSurv(
        input_dim=input_dim,
        hidden_dim=exp_config.hidden_dim,
        num_layers=exp_config.num_layers,
        dropout=exp_config.dropout
    )
    
    # 4. 학습 시작
    print("\n[Step 3] Starting Training...")
    trainer = DeepSurvTrainer(
        model=model,
        device=device,
        lr=exp_config.learning_rate,
        weight_decay=exp_config.weight_decay,
        patience=exp_config.patience
    )
    
    history = trainer.fit(train_loader, val_loader, epochs=exp_config.epochs)
    
    # 5. 결과 저장 및 시각화
    print("\n[Step 4] Saving Results...")
    
    # A. Model Weights
    model_path = os.path.join(results_dir, "lstm_deepsurv_best.pth")
    torch.save(trainer.model.state_dict(), model_path)
    print(f"   Saved model weights: {model_path}")
    
    # B. Visualization
    plot_path = os.path.join(plots_dir, "training_history.png")
    plot_training_history(history, plot_path)
    
    # C. Final Metrics (Best Validation Score)
    best_val_idx = np.argmax(history['val_cindex'])
    final_metrics = {
        'best_epoch': int(best_val_idx + 1),
        'best_val_cindex': float(history['val_cindex'][best_val_idx]),
        'min_val_loss': float(min(history['val_loss'])),
        'config': {k:v for k,v in vars(exp_config).items() if not k.startswith('__')}
    }
    
    metrics_path = os.path.join(results_dir, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"[Result] Best Validation C-Index: {final_metrics['best_val_cindex']:.4f} (Epoch {final_metrics['best_epoch']})")
    
    # 6. Advanced Evaluation (RMSE via Breslow)
    print("\n[Step 5] Calculating Advanced Metrics (RMSE)...")
    
    # Needs model in eval mode
    trainer.model.eval()
    
    # Collect training data for Breslow Baseline
    train_risks, train_times, train_events = [], [], []
    with torch.no_grad():
        for bx, bt, be in train_loader:
            bx = bx.to(device)
            out = trainer.model(bx)
            train_risks.append(out.cpu().numpy())
            train_times.append(bt.numpy())
            train_events.append(be.numpy())
    
    train_risks = np.concatenate(train_risks)
    train_times = np.concatenate(train_times)
    train_events = np.concatenate(train_events)
    
    # Fit Breslow
    breslow = BreslowEstimator()
    breslow.fit(train_risks, train_times, train_events)
    
    # Validate
    val_risks, val_times = [], []
    with torch.no_grad():
        for bx, bt, be in val_loader:
            bx = bx.to(device)
            out = trainer.model(bx)
            val_risks.append(out.cpu().numpy())
            val_times.append(bt.numpy())
            
    val_risks = np.concatenate(val_risks)
    val_times = np.concatenate(val_times)
    
    # Predict RUL
    surv_df = breslow.get_survival_function(val_risks)
    pred_rul = calculate_expected_rul(surv_df)
    rmse_val = calculate_rmse(val_times, pred_rul)
    
    final_metrics['val_rmse'] = float(rmse_val)
    
    # Update JSON
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"[Result] Validation RMSE: {rmse_val:.4f} cycles")
    print(f"[Result] Full metrics updated in {metrics_path}")

if __name__ == "__main__":
    run_pipeline()
