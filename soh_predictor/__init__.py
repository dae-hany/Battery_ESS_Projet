"""
SOH Predictor Package
=====================

배터리 SOH(State of Health) 예측을 위한 머신러닝 파이프라인

주요 컴포넌트:
- Config: 설정 관리
- DataLoaderFactory: 데이터 로드
- FeatureEngineer: 피처 엔지니어링
- ModelTrainer: 모델 학습
- SOHPredictor: SOH 예측

사용 예시:
    >>> from soh_predictor import Config, SOHPredictorPipeline
    >>> pipeline = SOHPredictorPipeline(Config())
    >>> pipeline.run()
"""

from .config import (
    Config,
    FrequencyConfig,
    SOHEstimationConfig,
    ModelConfig,
    PathConfig,
)
from .utils import (
    logger,
    extract_frequency_from_column,
    extract_soh_from_filename,
    extract_battery_id,
    parse_complex_impedance,
    get_rx_columns,
    normalize_frequency_column_name,
)
from .data_loaders import (
    BaseDataLoader,
    SpectroscopyLoader,
    CompanyBatteryLoader,
    MendeleyFormatLoader,
    SoCEstimationLoader,
    DataLoaderFactory,
)
from .preprocessors import (
    FrequencyHarmonizer,
    FeatureEngineer,
    DomainAdapter,
    SOHEstimator,
    DataAugmenter,
)
from .models import (
    ModelTrainer,
    SOHPredictor,
)
from .main import SOHPredictorPipeline

__version__ = "1.0.0"
__author__ = "Battery SOH Team"

__all__ = [
    # Config
    "Config",
    "FrequencyConfig",
    "SOHEstimationConfig",
    "ModelConfig",
    "PathConfig",
    # Utils
    "logger",
    "extract_frequency_from_column",
    "extract_soh_from_filename",
    "extract_battery_id",
    "parse_complex_impedance",
    "get_rx_columns",
    "normalize_frequency_column_name",
    # Data Loaders
    "BaseDataLoader",
    "SpectroscopyLoader",
    "CompanyBatteryLoader",
    "MendeleyFormatLoader",
    "SoCEstimationLoader",
    "DataLoaderFactory",
    # Preprocessors
    "FrequencyHarmonizer",
    "FeatureEngineer",
    "DomainAdapter",
    "SOHEstimator",
    "DataAugmenter",
    # Models
    "ModelTrainer",
    "SOHPredictor",
    # Pipeline
    "SOHPredictorPipeline",
]