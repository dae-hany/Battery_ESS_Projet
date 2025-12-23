# -*- coding: utf-8 -*-
"""
DeepSurv 모델 아키텍처 (LSTM + MLP)
===========================================================

이 모듈은 시계열 데이터를 처리하기 위한 LSTM 기반의 DeepSurv 모델을 정의합니다.

주요 클래스:
1. LSTMDeepSurv: LSTM으로 시계열 특징을 추출하고, MLP로 Risk Score를 예측하는 모델.

Author: ESS DeepSurv Research Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class LSTMDeepSurv(nn.Module):
    """
    LSTM 기반 DeepSurv 모델.
    
    구조:
        Input (Batch, Window, Features) 
        -> LSTM Layer (Time-series Feature Extraction)
        -> Last Time Step Extraction (Many-to-One)
        -> MLP Head (Risk Prediction)
        -> Output (Batch) - Log Hazard Ratio
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3):
        """
        모델 초기화.
        
        Args:
            input_dim (int): 입력 특성의 개수 (Feature Dimension)
            hidden_dim (int): LSTM 은닉층 및 MLP 첫 레이어의 차원
            num_layers (int): LSTM 레이어 수
            dropout (float): Dropout 비율
        """
        super(LSTMDeepSurv, self).__init__()
        
        # 1. LSTM Layer
        # batch_first=True: 입력 텐서가 (Batch, Seq, Feature) 형태임을 명시
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2. MLP Head (Risk Prediction)
        # LSTM의 마지막 출력을 받아 단일 Risk Score 예측
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 가중치 초기화
        self._initialize_weights()

    def _initialize_weights(self):
        """
        가중치 초기화 (Xavier Uniform).
        학습 안정성을 위해 Linear 및 LSTM 가중치를 초기화합니다.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    init.xavier_uniform_(param)
                else:
                    init.uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass.
        
        Args:
            x (torch.Tensor): 입력 데이터 (Batch_Size, Window_Size, Input_Dim)
            
        Returns:
            torch.Tensor: Log Hazard Ratio (Batch_Size,)
        """
        # LSTM Forward
        # out shape: (Batch, Window, Hidden)
        # h_n, c_n은 사용하지 않음
        out, _ = self.lstm(x)
        
        # Many-to-One: 마지막 타임스텝의 출력만 사용
        # out[:, -1, :] shape: (Batch, Hidden)
        last_step_out = out[:, -1, :]
        
        # MLP Head
        # risk_score shape: (Batch, 1)
        risk_score = self.mlp(last_step_out)
        
        # 1차원으로 변환하여 반환 (Batch,)
        return risk_score.squeeze(-1)

# ===================================================================================
# Main Execution (Verification)
# ===================================================================================

if __name__ == "__main__":
    print("=== LSTMDeepSurv Model Verification ===")
    
    # 1. 모델 인스턴스 생성 테스트
    batch_size = 32
    window_size = 10
    input_dim = 4 # capacity, soh, soh_deriv, cap_deriv
    
    model = LSTMDeepSurv(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.3)
    print(f"[Info] Model Architecture:\n{model}")
    
    # 2. 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Info] Total Parameters: {total_params:,}")
    
    # 3. Forward Pass 테스트
    dummy_input = torch.randn(batch_size, window_size, input_dim)
    print(f"[Info] Input Shape: {dummy_input.shape}")
    
    try:
        output = model(dummy_input)
        print(f"[Info] Output Shape: {output.shape}")
        
        # 차원 검증
        assert output.shape == (batch_size,), f"Expected output shape ({batch_size},), but got {output.shape}"
        print("[Verification Success] Forward pass successful.")
        
    except Exception as e:
        print(f"[Error] Forward pass failed: {e}")
