"""
설정 관리 모듈
=============
모든 하이퍼파라미터와 경로를 중앙 관리
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FrequencyConfig:
    """주파수별 가중치 설정 (EIS 분석 결과 기반)"""
    weights: Dict[str, float] = field(default_factory=lambda: {
        '1kHz': 3.0,    # 상관관계 -0.972
        '500Hz': 2.5,   # 변화율 570%
        '400Hz': 2.0,   # 상관관계 -0.961
        '250Hz': 1.8,   # 변화율 138%
        '200Hz': 1.5,   # 상관관계 -0.945
        '4Hz': 2.0,     # 변화율 223%
        '8Hz': 1.5,     # 변화율 93%
        '2Hz': 1.3,     # 변화율 76%
        '16Hz': 1.2,    # 변화율 75%
        '125Hz': 1.0,
        '62Hz': 1.0,
        '31Hz': 1.0,
        '100Hz': 1.0,
        '1Hz': 0.8,
    })
    
    key_frequencies: List[str] = field(default_factory=lambda: [
        '1kHz', '500Hz', '400Hz', '4Hz', '8Hz'
    ])


@dataclass(frozen=True)
class SOHEstimationConfig:
    """SOH 추정 설정"""
    condition_soh_map: Dict[str, float] = field(default_factory=lambda: {
        '신품': 95.0,
        '21만키로': 75.0,
        '21만': 75.0,
        '불량': 60.0,
    })
    
    default_pseudo_label: float = 95.0
    estimation_blend_ratio: float = 0.7  # 예측값 비중


@dataclass(frozen=True)
class ModelConfig:
    """모델 학습 설정"""
    n_features_max: int = 100
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    n_seeds: int = 5
    
    # 데이터 증강
    augmentation_factor: float = 2.0
    noise_factor: float = 0.02
    
    # GridSearch 파라미터
    rf_params: Dict = field(default_factory=lambda: {
        'n_estimators': [300, 500, 700],
        'max_depth': [15, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
    })
    
    gb_params: Dict = field(default_factory=lambda: {
        'n_estimators': [500, 700, 1000],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 10],
        'subsample': [0.8, 0.9, 1.0],
    })
    
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': [300, 500],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
    })
    
    lgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': [300, 500],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50],
    })


@dataclass
class PathConfig:
    """데이터 경로 설정"""
    base_dir: Path = field(default_factory=lambda: Path("."))
    
    @property
    def spectroscopy_dir(self) -> Path:
        return self.base_dir / "datasets/raw_data/Spectroscopy_Individual"
    
    @property
    def company_battery_dir(self) -> Path:
        return self.base_dir / "datasets/raw_data/company_battery_data"
    
    @property
    def mbv3bx_dir(self) -> Path:
        return self.base_dir / "mbv3bx847g1_analysis/mbv3bx847g-1"
    
    @property
    def samsung_dir(self) -> Path:
        return self.base_dir / "datasets/similar/samsung_icr18650_26j"
    
    @property
    def soc_estimation_dir(self) -> Path:
        return self.base_dir / "datasets/SoC Estimation on Li-ion Batteries A New EIS-based Dataset for data-driven applications"
    
    @property
    def model_output_dir(self) -> Path:
        return self.base_dir / "battery_soh_system/models"


@dataclass
class Config:
    """통합 설정"""
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    soh_estimation: SOHEstimationConfig = field(default_factory=SOHEstimationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)