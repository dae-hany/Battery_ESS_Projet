# -*- coding: utf-8 -*-
"""
Hyperparameter Optimization with Optuna
===========================================================

이 스크립트는 Optuna를 사용하여 LSTM-DeepSurv 모델의 하이퍼파라미터를 최적화합니다.
Validation C-Index를 최대화하는 파라미터 조합을 탐색합니다.

Author: ESS DeepSurv Research Team
Date: 2025
"""

import os
import sys
import json
import torch
import optuna
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import DataConfig, get_dataloaders
from src.models import LSTMDeepSurv
from src.trainer import DeepSurvTrainer

# ===================================================================================
# Optimization Configuration
# ===================================================================================

N_TRIALS = 20
EPOCHS_PER_TRIAL = 30  # 빠른 탐색을 위해 Epoch 제한 (30~50)
BASE_DIR = r"C:\Users\daeho\OneDrive\바탕 화면\Battery_ESS_Project"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def objective(trial):
    """
    Optuna Objective Function.
    
    1. Define Hyperparameters
    2. Load Data (Dynamic Window Size)
    3. Init Model
    4. Train & Evaluate
    5. Return Best Validation C-Index
    """
    
    # 1. Hyperparameter Search Space
    window_size = trial.suggest_int('window_size', 5, 30)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Data Loading (Window Size에 따라 데이터가 달라짐)
    # 매 trial마다 데이터를 다시 로드해야 함 (비효율적일 수 있으나 window_size 튜닝을 위해 필요)
    try:
        data_config = DataConfig(
            base_dir=BASE_DIR,
            window_size=window_size,
            feature_set_name='dynamic'
        )
        
        # num_workers=0 for stability in windows
        train_loader, val_loader = get_dataloaders(data_config, batch_size=32)
        
        # Input dim 확인
        sample_batch = next(iter(train_loader))[0]
        input_dim = sample_batch.shape[-1]
        
    except Exception as e:
        print(f"[Trial Failed] Data loading error: {e}")
        # 데이터 로드 실패 시 (e.g. window size가 너무 커서 데이터가 없는 경우)
        return 0.0

    # 3. Model Initialization
    model = LSTMDeepSurv(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # 4. Trainer Initialization
    # Pruning을 위해 Epoch마다 보고하는 로직이 필요하지만, 
    # 기존 Trainer를 수정하지 않고 Epoch 수를 줄여서 빠르게 탐색.
    trainer = DeepSurvTrainer(
        model=model,
        device=device,
        lr=learning_rate,
        patience=10 # 짧은 Epoch이므로 patience도 작게
    )
    
    print(f"\n[Trial {trial.number}] Params: Win={window_size}, Hin={hidden_dim}, Lay={num_layers}, LR={learning_rate:.5f}")
    
    # 5. Training
    try:
        history = trainer.fit(train_loader, val_loader, epochs=EPOCHS_PER_TRIAL, verbose=100) # verbose 높여서 로그 최소화
        
        # Best Validation C-Index
        best_cindex = max(history['val_cindex'])
        
        # Pruning check (Optional implementation if trainer supported it)
        # trial.report(best_cindex, step=EPOCHS_PER_TRIAL)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
            
        return best_cindex
        
    except Exception as e:
        print(f"[Trial Failed] Training error: {e}")
        return 0.0

def run_optimization():
    print("=== Optuna Hyperparameter Optimization ===")
    print(f"Trials: {N_TRIALS}, Max Epochs per Trial: {EPOCHS_PER_TRIAL}")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n" + "="*50)
    print("Optimization Finished!")
    print("="*50)
    
    # Print Best Results
    print(f"Best C-Index: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    # Save Best Params
    best_params_path = os.path.join(RESULTS_DIR, "best_params.json")
    results = {
        'best_cindex': study.best_value,
        'best_params': study.best_params,
        'n_trials': N_TRIALS
    }
    
    with open(best_params_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[Info] Best parameters saved to {best_params_path}")
    
    # Visualization (Optional if matplotlib available)
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(os.path.join(RESULTS_DIR, "optuna_history.png"))
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(os.path.join(RESULTS_DIR, "optuna_importance.png"))
        print("[Info] Optuna plots saved.")
    except Exception:
        pass # kaleido 등이 없으면 이미지 저장이 안될 수 있음

if __name__ == "__main__":
    run_optimization()
