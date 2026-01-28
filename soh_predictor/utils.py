"""
유틸리티 모듈
=============
공통으로 사용되는 헬퍼 함수들
"""

import re
import unicodedata
import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_string(s: str) -> str:
    """유니코드 정규화"""
    return unicodedata.normalize('NFC', s)


def extract_frequency_from_column(col_name: str) -> Optional[float]:
    """
    컬럼명에서 주파수 값 추출
    
    Examples:
        >>> extract_frequency_from_column('R_1kHz')
        1000.0
        >>> extract_frequency_from_column('R_500Hz')
        500.0
    """
    freq_str = col_name.replace('R_', '').replace('X_', '').replace('Hz', '').strip()
    
    # kHz 처리
    if 'k' in freq_str.lower():
        freq_str = freq_str.lower().replace('k', '')
        try:
            return float(freq_str) * 1000
        except ValueError:
            return 1000.0
    
    # 숫자 추출
    try:
        return float(freq_str)
    except ValueError:
        try:
            return float(freq_str.replace('.', ''))
        except ValueError:
            return None


def extract_soh_from_filename(filename: str) -> Optional[float]:
    """파일명에서 SOH 값 추출"""
    match = re.search(r'SOH(\d+\.?\d*)', filename)
    return float(match.group(1)) if match else None


def extract_battery_id(filename: str) -> str:
    """파일명에서 배터리 ID 추출"""
    match = re.search(r'Batt(\d+)', filename)
    return match.group(1) if match else 'unknown'


def parse_complex_impedance(imp_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    복소수 임피던스 문자열 파싱
    
    Formats:
        - (0.110973-0.00547j)
        - 0.110973+0.00547j
    
    Returns:
        (R, X) 튜플 또는 (None, None)
    """
    try:
        imp_str = str(imp_str).strip().strip('()')
        
        # +j 형식
        if '+' in imp_str and 'j' in imp_str:
            parts = imp_str.replace('j', '').split('+')
            if len(parts) == 2:
                return float(parts[0]), float(parts[1])
        
        # -j 형식
        elif '-' in imp_str and 'j' in imp_str:
            # 첫 번째 마이너스는 실수부의 음수일 수 있으므로 주의
            parts = imp_str.replace('j', '').rsplit('-', 1)
            if len(parts) == 2:
                return float(parts[0]), -float(parts[1])
        
        return None, None
    except (ValueError, AttributeError):
        return None, None


def get_rx_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """DataFrame에서 R_, X_ 컬럼 추출"""
    r_cols = [col for col in df.columns if col.startswith('R_')]
    x_cols = [col for col in df.columns if col.startswith('X_')]
    return r_cols, x_cols


def normalize_frequency_column_name(freq_value: float) -> str:
    """주파수 값을 표준 컬럼명으로 변환"""
    if freq_value >= 1000:
        return '1kHz'
    return f"{int(freq_value)}Hz"


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                fill_value: float = 0.0) -> np.ndarray:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            np.abs(denominator) > 1e-10,
            numerator / denominator,
            fill_value
        )
    return result