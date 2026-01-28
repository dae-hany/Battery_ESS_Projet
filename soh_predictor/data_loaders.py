"""
데이터 로더 모듈
===============
다양한 형식의 배터리 EIS 데이터 로드
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from .config import Config, SOHEstimationConfig
from .utils import (
    logger, normalize_string, extract_soh_from_filename,
    extract_battery_id, parse_complex_impedance, get_rx_columns,
    normalize_frequency_column_name
)


class BaseDataLoader(ABC):
    """데이터 로더 베이스 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
    
    @abstractmethod
    def load(self, data_dir: Path) -> pd.DataFrame:
        """데이터 로드 (하위 클래스에서 구현)"""
        pass
    
    def _to_numeric_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """컬럼들을 숫자형으로 변환"""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def _log_load_result(self, name: str, df: pd.DataFrame) -> None:
        """로드 결과 로깅"""
        if df.empty:
            logger.warning(f"{name}: 데이터 없음")
        else:
            logger.info(f"{name}: {len(df)}개 샘플 로드 완료")
            if 'SOH' in df.columns:
                logger.info(f"  SOH 범위: {df['SOH'].min():.1f}% ~ {df['SOH'].max():.1f}%")


class SpectroscopyLoader(BaseDataLoader):
    """Spectroscopy Individual 데이터 로더 (실제 SOH 라벨)"""
    
    def load(self, data_dir: Path) -> pd.DataFrame:
        logger.info("📊 Spectroscopy 데이터 로드 중...")
        
        if not data_dir.exists():
            logger.warning(f"경로 없음: {data_dir}")
            return pd.DataFrame()
        
        csv_files = list(data_dir.glob("*.csv"))
        logger.info(f"  파일 수: {len(csv_files)}개")
        
        records = []
        for file_path in csv_files:
            record = self._process_file(file_path)
            if record is not None:
                records.append(record)
        
        df = pd.DataFrame(records) if records else pd.DataFrame()
        self._log_load_result("Spectroscopy", df)
        return df
    
    def _process_file(self, file_path: Path) -> Optional[Dict]:
        """단일 파일 처리"""
        try:
            soh = extract_soh_from_filename(file_path.name)
            if soh is None:
                return None
            
            df = pd.read_csv(file_path)
            r_cols, x_cols = get_rx_columns(df)
            df = self._to_numeric_columns(df, r_cols + x_cols)
            
            valid_df = df[r_cols + x_cols].dropna(how='all')
            if valid_df.empty:
                return None
            
            record = valid_df.mean().to_dict()
            record.update({
                'SOH': soh,
                'data_source': 'spectroscopy',
                'battery_id': extract_battery_id(file_path.name),
            })
            return record
            
        except Exception as e:
            logger.warning(f"파일 읽기 실패: {file_path.name} - {e}")
            return None


class CompanyBatteryLoader(BaseDataLoader):
    """Company Battery 데이터 로더"""
    
    def load(self, data_dir: Path) -> pd.DataFrame:
        logger.info("📊 Company Battery 데이터 로드 중...")
        
        if not data_dir.exists():
            logger.warning(f"경로 없음: {data_dir}")
            return pd.DataFrame()
        
        csv_files = list(data_dir.glob("*.csv"))
        logger.info(f"  파일 수: {len(csv_files)}개")
        
        records = []
        for file_path in csv_files:
            record = self._process_file(file_path)
            if record is not None:
                records.append(record)
        
        df = pd.DataFrame(records) if records else pd.DataFrame()
        self._log_load_result("Company Battery", df)
        
        if not df.empty:
            self._log_condition_counts(df)
        
        return df
    
    def _process_file(self, file_path: Path) -> Optional[Dict]:
        """단일 파일 처리"""
        try:
            condition = self._find_condition(file_path.name)
            if condition is None:
                return None
            
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            
            r_cols, x_cols = get_rx_columns(df)
            df = self._to_numeric_columns(df, r_cols + x_cols)
            
            valid_df = df[r_cols + x_cols].dropna(how='all')
            if valid_df.empty:
                return None
            
            soh_map = self.config.soh_estimation.condition_soh_map
            record = valid_df.mean().to_dict()
            record.update({
                'SOH': soh_map[condition],
                'data_source': 'company_battery',
                'condition': condition,
                'soh_estimated': True,
            })
            return record
            
        except Exception as e:
            logger.warning(f"파일 읽기 실패: {file_path.name} - {e}")
            return None
    
    def _find_condition(self, filename: str) -> Optional[str]:
        """파일명에서 배터리 상태 추출"""
        filename_norm = normalize_string(filename)
        soh_map = self.config.soh_estimation.condition_soh_map
        
        for key in soh_map:
            if normalize_string(key) in filename_norm:
                return key
        
        # 영문 매칭
        filename_lower = filename.lower()
        if 'new' in filename_lower:
            return '신품'
        if 'bad' in filename_lower or 'defect' in filename_lower:
            return '불량'
        
        return None
    
    def _log_condition_counts(self, df: pd.DataFrame) -> None:
        """상태별 개수 로깅"""
        for condition in df['condition'].unique():
            count = len(df[df['condition'] == condition])
            logger.info(f"  {condition}: {count}개")


class MendeleyFormatLoader(BaseDataLoader):
    """Mendeley 형식 데이터 로더 (mbv3bx847g, Samsung 등)"""
    
    def __init__(self, config: Config, source_name: str, pseudo_soh: float = 95.0):
        super().__init__(config)
        self.source_name = source_name
        self.pseudo_soh = pseudo_soh
    
    def load(self, data_dir: Path) -> pd.DataFrame:
        logger.info(f"📊 {self.source_name} 데이터 로드 중...")
        
        freq_file = data_dir / "frequencies.csv"
        imp_file = data_dir / "impedance.csv"
        
        # 대문자 파일명도 체크
        if not freq_file.exists():
            freq_file = data_dir / "FREQUENCIES.CSV"
        if not imp_file.exists():
            imp_file = data_dir / "IMPEDANCE.CSV"
        
        if not (freq_file.exists() and imp_file.exists()):
            logger.warning(f"파일 없음: {data_dir}")
            return pd.DataFrame()
        
        try:
            freq_df = pd.read_csv(freq_file)
            imp_df = pd.read_csv(imp_file)
            
            # 복소수 파싱
            imp_df[['R', 'X']] = imp_df['IMPEDANCE_VALUE'].apply(
                lambda x: pd.Series(parse_complex_impedance(x))
            )
            
            imp_df = imp_df.merge(freq_df, on='FREQUENCY_ID', how='left')
            
            # 피벗
            pivoted_data = self._pivot_data(imp_df)
            
            df = pd.DataFrame(pivoted_data)
            df['data_source'] = self.source_name
            df['SOH'] = self.pseudo_soh
            
            self._log_load_result(self.source_name, df)
            return df
            
        except Exception as e:
            logger.error(f"로드 실패: {e}")
            return pd.DataFrame()
    
    def _pivot_data(self, imp_df: pd.DataFrame) -> List[Dict]:
        """측정 단위로 피벗"""
        pivoted = []
        group_cols = ['MEASURE_ID', 'SOC', 'BATTERY_ID']
        available_cols = [c for c in group_cols if c in imp_df.columns]
        
        for keys, group in imp_df.groupby(available_cols):
            row = dict(zip(available_cols, keys if isinstance(keys, tuple) else [keys]))
            
            for _, freq_row in group.iterrows():
                freq_val = freq_row.get('FREQUENCY_VALUE')
                if pd.isna(freq_val):
                    continue
                
                freq_str = normalize_frequency_column_name(freq_val)
                row[f'R_{freq_str}'] = freq_row['R']
                row[f'X_{freq_str}'] = freq_row['X']
            
            pivoted.append(row)
        
        return pivoted


class SoCEstimationLoader(BaseDataLoader):
    """Li-Ion SoC Estimation 데이터 로더"""
    
    def __init__(self, config: Config, pseudo_soh: float = 95.0):
        super().__init__(config)
        self.pseudo_soh = pseudo_soh
    
    def load(self, data_dir: Path) -> pd.DataFrame:
        logger.info("📊 Li-Ion SoC Estimation 데이터 로드 중...")
        
        if not data_dir.exists():
            logger.warning(f"디렉토리 없음: {data_dir}")
            return pd.DataFrame()
        
        battery_dirs = sorted([
            d for d in data_dir.iterdir() 
            if d.is_dir() and d.name.startswith('B')
        ])
        
        if not battery_dirs:
            logger.warning("배터리 디렉토리 없음 (B01~B11 기대)")
            return pd.DataFrame()
        
        all_records = []
        for battery_dir in battery_dirs:
            records = self._process_battery_dir(battery_dir)
            all_records.extend(records)
        
        if not all_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        df['data_source'] = 'soc_estimation'
        df['SOH'] = self.pseudo_soh
        
        self._log_load_result("Li-Ion SoC Estimation", df)
        return df
    
    def _process_battery_dir(self, battery_dir: Path) -> List[Dict]:
        """배터리 디렉토리 처리"""
        battery_id = battery_dir.name
        eis_base = battery_dir / 'EIS measurements'
        
        if not eis_base.exists():
            return []
        
        # CSV 파일 수집
        csv_files = []
        for test_dir in eis_base.iterdir():
            if test_dir.is_dir() and test_dir.name.startswith('Test'):
                hioki_dir = test_dir / 'Hioki'
                if hioki_dir.exists():
                    csv_files.extend(hioki_dir.glob('*.csv'))
        
        if not csv_files:
            return []
        
        logger.info(f"  처리 중: {battery_id} ({len(csv_files)} 파일)")
        
        records = []
        for csv_file in csv_files:
            record = self._process_csv(csv_file, battery_id)
            if record:
                records.append(record)
        
        return records
    
    def _process_csv(self, csv_file: Path, battery_id: str) -> Optional[Dict]:
        """단일 CSV 처리"""
        try:
            df = pd.read_csv(csv_file)
            
            # 컬럼 찾기
            freq_col = self._find_column(df, ['Frequency(Hz)', 'frequency', 'freq'])
            r_col = self._find_column(df, ['R(ohm)', 'R', 'real'])
            x_col = self._find_column(df, ['X(ohm)', 'X', 'imag'])
            
            if not all([freq_col, r_col, x_col]):
                return None
            
            # SOC 추출
            soc = self._extract_soc_from_filename(csv_file.name)
            
            # 피벗
            row = {
                'BATTERY_ID': battery_id,
                'MEASURE_ID': csv_file.stem,
                'SOC': soc,
            }
            
            for _, freq_row in df.iterrows():
                freq_val = freq_row[freq_col]
                if pd.isna(freq_val):
                    continue
                
                freq_str = normalize_frequency_column_name(freq_val)
                row[f'R_{freq_str}'] = freq_row[r_col]
                row[f'X_{freq_str}'] = freq_row[x_col]
            
            return row
            
        except Exception as e:
            logger.warning(f"파일 읽기 실패: {csv_file.name} - {e}")
            return None
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """후보 중 존재하는 컬럼 찾기"""
        for col in df.columns:
            col_lower = col.lower()
            for candidate in candidates:
                if candidate.lower() in col_lower:
                    return col
        return None
    
    def _extract_soc_from_filename(self, filename: str) -> Optional[float]:
        """파일명에서 SOC 추출 (예: Hk_IFR14500_SoC_100_...)"""
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.lower() == 'soc' and i + 1 < len(parts):
                try:
                    return float(parts[i + 1])
                except ValueError:
                    pass
        return None


class DataLoaderFactory:
    """데이터 로더 팩토리"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """모든 데이터셋 로드"""
        paths = self.config.paths
        
        datasets = {
            'spectroscopy': SpectroscopyLoader(self.config).load(paths.spectroscopy_dir),
            'company_battery': CompanyBatteryLoader(self.config).load(paths.company_battery_dir),
        }
        
        # 선택적 데이터셋
        if paths.samsung_dir.exists():
            datasets['samsung'] = MendeleyFormatLoader(
                self.config, 'samsung_icr18650'
            ).load(paths.samsung_dir)
        
        if paths.soc_estimation_dir.exists():
            datasets['soc_estimation'] = SoCEstimationLoader(
                self.config
            ).load(paths.soc_estimation_dir)
        
        return {k: v for k, v in datasets.items() if not v.empty}