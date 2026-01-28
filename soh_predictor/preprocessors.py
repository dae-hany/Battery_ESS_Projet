"""
전처리 모듈
==========
데이터 정제, 피처 엔지니어링, 도메인 적응
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import Config
from .utils import (
    logger, extract_frequency_from_column, get_rx_columns,
    normalize_frequency_column_name, safe_divide
)


class FrequencyHarmonizer:
    """주파수 컬럼 조화 (데이터셋 간 표준화)"""
    
    def harmonize(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
        """모든 데이터프레임의 주파수 컬럼 통일"""
        logger.info("🔧 주파수 컬럼 조화 중...")
        
        # 주파수-컬럼 매핑 수집
        freq_mapping = self._collect_frequency_mapping(dataframes)
        
        if not freq_mapping:
            return pd.DataFrame()
        
        common_freqs = sorted(freq_mapping.keys())
        logger.info(f"  발견된 주파수: {len(common_freqs)}개")
        
        # 각 데이터프레임 변환
        harmonized_dfs = [
            self._transform_dataframe(df, freq_mapping, common_freqs)
            for df in dataframes if not df.empty
        ]
        
        if not harmonized_dfs:
            return pd.DataFrame()
        
        combined = pd.concat(harmonized_dfs, ignore_index=True)
        logger.info(f"  ✅ 조화 완료: {len(combined)}개 샘플")
        return combined
    
    def _collect_frequency_mapping(
        self, dataframes: Tuple[pd.DataFrame, ...]
    ) -> Dict[float, Dict[int, Tuple[str, Optional[str]]]]:
        """주파수별 컬럼 매핑 수집"""
        freq_map: Dict[float, Dict[int, Tuple[str, Optional[str]]]] = {}
        
        for df_idx, df in enumerate(dataframes):
            if df.empty:
                continue
            
            r_cols, _ = get_rx_columns(df)
            for r_col in r_cols:
                freq = extract_frequency_from_column(r_col)
                if freq is None:
                    continue
                
                # 정규화
                norm_freq = 1000 if freq >= 1000 else int(freq) if freq == int(freq) else freq
                
                if norm_freq not in freq_map:
                    freq_map[norm_freq] = {}
                
                x_col = r_col.replace('R_', 'X_')
                x_col = x_col if x_col in df.columns else None
                freq_map[norm_freq][df_idx] = (r_col, x_col)
        
        return freq_map
    
    def _transform_dataframe(
        self, 
        df: pd.DataFrame, 
        freq_mapping: Dict,
        common_freqs: List[float]
    ) -> pd.DataFrame:
        """단일 데이터프레임을 공통 형식으로 변환"""
        harmonized = pd.DataFrame()
        
        # 메타 컬럼 복사
        for col in ['SOH', 'data_source', 'condition', 'BATTERY_ID', 'battery_id', 'soh_estimated']:
            if col in df.columns:
                harmonized[col] = df[col].values
        
        # 주파수 컬럼 매핑
        df_idx = None
        for freq in common_freqs:
            if freq in freq_mapping:
                for idx in freq_mapping[freq]:
                    if idx not in [None]:  # df 인덱스 찾기
                        df_idx = idx
                        break
        
        for freq in common_freqs:
            target_r = f'R_{normalize_frequency_column_name(freq)}'
            target_x = f'X_{normalize_frequency_column_name(freq)}'
            
            # 원본에서 매칭되는 컬럼 찾기
            source_cols = self._find_source_columns(df, freq)
            
            if source_cols:
                harmonized[target_r] = df[source_cols[0]].values
                harmonized[target_x] = df[source_cols[1]].values if source_cols[1] else 0
            else:
                harmonized[target_r] = 0
                harmonized[target_x] = 0
        
        return harmonized
    
    def _find_source_columns(
        self, df: pd.DataFrame, target_freq: float
    ) -> Optional[Tuple[str, Optional[str]]]:
        """원본 데이터프레임에서 해당 주파수 컬럼 찾기"""
        r_cols, _ = get_rx_columns(df)
        
        for r_col in r_cols:
            freq = extract_frequency_from_column(r_col)
            if freq is None:
                continue
            
            norm_freq = 1000 if freq >= 1000 else int(freq) if freq == int(freq) else freq
            if norm_freq == target_freq:
                x_col = r_col.replace('R_', 'X_')
                return (r_col, x_col if x_col in df.columns else None)
        
        return None


class FeatureEngineer:
    """피처 엔지니어링"""
    
    def __init__(self, config: Config):
        self.config = config
        self.freq_weights = config.frequency.weights
        self.key_freqs = config.frequency.key_frequencies
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 피처 추출"""
        enhanced = df.copy()
        r_cols, x_cols = get_rx_columns(df)
        
        if not r_cols:
            return enhanced
        
        # 기본 통계
        enhanced = self._add_basic_stats(enhanced, r_cols, x_cols)
        
        # 주파수 대역별 통계
        enhanced = self._add_frequency_band_stats(enhanced, r_cols)
        
        # 핵심 주파수 특성
        enhanced = self._add_key_frequency_features(enhanced, df)
        
        # 비율 피처
        enhanced = self._add_ratio_features(enhanced)
        
        return enhanced
    
    def _add_basic_stats(
        self, df: pd.DataFrame, r_cols: List[str], x_cols: List[str]
    ) -> pd.DataFrame:
        """기본 통계 피처 추가"""
        df['R_mean'] = df[r_cols].mean(axis=1)
        df['R_std'] = df[r_cols].std(axis=1)
        df['R_min'] = df[r_cols].min(axis=1)
        df['R_max'] = df[r_cols].max(axis=1)
        df['R_range'] = df['R_max'] - df['R_min']
        
        if x_cols:
            df['X_mean'] = df[x_cols].mean(axis=1)
            df['X_std'] = df[x_cols].std(axis=1)
            df['X_min'] = df[x_cols].min(axis=1)
            df['X_max'] = df[x_cols].max(axis=1)
            df['X_range'] = df['X_max'] - df['X_min']
        
        return df
    
    def _add_frequency_band_stats(
        self, df: pd.DataFrame, r_cols: List[str]
    ) -> pd.DataFrame:
        """주파수 대역별 통계 추가"""
        bands = {
            'low': (0, 10),
            'mid': (10, 100),
            'high': (100, float('inf')),
        }
        
        for band_name, (low, high) in bands.items():
            band_cols = [
                col for col in r_cols
                if (freq := extract_frequency_from_column(col)) 
                and low < freq <= high
            ]
            
            if band_cols:
                df[f'{band_name}_freq_R_mean'] = df[band_cols].mean(axis=1)
                df[f'{band_name}_freq_R_std'] = df[band_cols].std(axis=1)
        
        return df
    
    def _add_key_frequency_features(
        self, df: pd.DataFrame, original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """핵심 주파수 가중 피처 추가"""
        for freq_str in self.key_freqs:
            r_col = f"R_{freq_str}"
            x_col = f"X_{freq_str}"
            
            if r_col not in original_df.columns or x_col not in original_df.columns:
                continue
            
            weight = self.freq_weights.get(freq_str, 1.0)
            
            df[f'R_{freq_str}_weighted'] = original_df[r_col] * weight
            df[f'X_{freq_str}_weighted'] = original_df[x_col] * weight
            
            # 임피던스 크기 및 위상각
            df[f'Z_{freq_str}'] = np.sqrt(
                original_df[r_col]**2 + original_df[x_col]**2
            )
            df[f'Phase_{freq_str}'] = np.arctan2(
                original_df[x_col], original_df[r_col]
            )
        
        return df
    
    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """비율 피처 추가"""
        if 'high_freq_R_mean' in df.columns and 'low_freq_R_mean' in df.columns:
            df['high_low_freq_ratio'] = safe_divide(
                df['high_freq_R_mean'].values,
                np.abs(df['low_freq_R_mean'].values)
            )
        return df


class DomainAdapter:
    """도메인 적응: 데이터셋 간 분포 차이 해결"""
    
    def __init__(self, reference_source: str = 'spectroscopy'):
        self.reference_source = reference_source
    
    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        """도메인 적응 적용"""
        logger.info("🌐 도메인 적응 적용 중...")
        
        if 'data_source' not in df.columns:
            logger.warning("data_source 컬럼 없음, 스킵")
            return df
        
        sources = df['data_source'].unique()
        if len(sources) < 2:
            logger.warning("데이터 소스 1개 이하, 스킵")
            return df
        
        # 소스별 통계 계산
        source_stats = self._compute_source_stats(df)
        
        if self.reference_source not in source_stats:
            self.reference_source = list(source_stats.keys())[0]
        
        ref_stats = source_stats[self.reference_source]
        logger.info(f"  기준: {self.reference_source}")
        
        # 적응 적용
        adapted_df = df.copy()
        r_cols, x_cols = get_rx_columns(df)
        
        for source, stats in source_stats.items():
            if source == self.reference_source:
                continue
            
            mask = adapted_df['data_source'] == source
            if stats['std'] > 0:
                for col in r_cols + x_cols:
                    if col in adapted_df.columns:
                        adapted_df.loc[mask, col] = (
                            (adapted_df.loc[mask, col] - stats['mean']) 
                            / stats['std'] * ref_stats['std'] + ref_stats['mean']
                        )
        
        logger.info("  ✅ 도메인 적응 완료")
        return adapted_df
    
    def _compute_source_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """소스별 통계 계산"""
        stats = {}
        r_cols, _ = get_rx_columns(df)
        
        for source in df['data_source'].unique():
            source_data = df[df['data_source'] == source]
            if r_cols:
                stats[source] = {
                    'mean': source_data[r_cols].mean().mean(),
                    'std': source_data[r_cols].mean().std(),
                }
        return stats


class SOHEstimator:
    """SOH 추정 개선기"""
    
    def __init__(self, config: Config):
        self.config = config
        self.blend_ratio = config.soh_estimation.estimation_blend_ratio
    
    def improve_estimates(
        self, 
        reference_df: pd.DataFrame, 
        target_df: pd.DataFrame
    ) -> pd.DataFrame:
        """참조 데이터를 사용하여 타겟의 SOH 추정 개선"""
        logger.info("🔧 SOH 추정 개선 중...")
        
        if reference_df.empty or target_df.empty:
            return target_df
        
        # 공통 피처 찾기
        common_cols = self._find_common_features(reference_df, target_df)
        
        if len(common_cols) < 4:
            logger.warning("공통 피처 부족, 초기 추정값 사용")
            return target_df
        
        # 모델 학습 및 예측
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        X_ref = reference_df[common_cols].fillna(0)
        y_ref = reference_df['SOH']
        model.fit(X_ref, y_ref)
        
        X_target = target_df[common_cols].fillna(0)
        predicted_soh = model.predict(X_target)
        
        # 블렌딩
        target_df = target_df.copy()
        original_soh = target_df['SOH'].values
        target_df['SOH'] = self.blend_ratio * predicted_soh + (1 - self.blend_ratio) * original_soh
        
        logger.info(f"  ✅ 조정 범위: {target_df['SOH'].min():.1f}% ~ {target_df['SOH'].max():.1f}%")
        return target_df
    
    def _find_common_features(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> List[str]:
        """공통 R/X 피처 찾기"""
        r_cols1, x_cols1 = get_rx_columns(df1)
        r_cols2, x_cols2 = get_rx_columns(df2)
        
        common = set(r_cols1 + x_cols1) & set(r_cols2 + x_cols2)
        return sorted(list(common))


class DataAugmenter:
    """데이터 증강"""
    
    def __init__(self, config: Config):
        self.config = config
        self.noise_factor = config.model.noise_factor
    
    def augment(
        self, df: pd.DataFrame, factor: float = 2.0
    ) -> pd.DataFrame:
        """데이터 증강 수행"""
        logger.info(f"📈 데이터 증강 중 (목표: {factor}배)...")
        
        original_size = len(df)
        target_size = int(original_size * factor)
        needed = target_size - original_size
        
        if needed <= 0:
            return df
        
        r_cols, x_cols = get_rx_columns(df)
        if not r_cols:
            return df
        
        augmented_rows = []
        np.random.seed(self.config.model.random_state)
        
        # 노이즈 추가
        augmented_rows.extend(
            self._add_noise_samples(df, r_cols + x_cols, min(needed, original_size))
        )
        
        # 보간
        remaining = needed - len(augmented_rows)
        if remaining > 0 and 'SOH' in df.columns:
            augmented_rows.extend(
                self._interpolate_samples(df, r_cols + x_cols, remaining)
            )
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            combined = pd.concat([df, augmented_df], ignore_index=True)
            logger.info(f"  ✅ {original_size}개 → {len(combined)}개")
            return combined
        
        return df
    
    def _add_noise_samples(
        self, df: pd.DataFrame, cols: List[str], n_samples: int
    ) -> List[pd.Series]:
        """노이즈 추가 샘플 생성"""
        samples = []
        for _ in range(n_samples):
            idx = np.random.randint(0, len(df))
            row = df.iloc[idx].copy()
            
            for col in cols:
                if col in row and pd.notna(row[col]):
                    noise = np.random.normal(0, abs(row[col]) * self.noise_factor)
                    row[col] = row[col] + noise
            
            samples.append(row)
        return samples
    
    def _interpolate_samples(
        self, df: pd.DataFrame, cols: List[str], n_samples: int
    ) -> List[pd.Series]:
        """보간 샘플 생성"""
        samples = []
        soh_range = (df['SOH'].min(), df['SOH'].max())
        soh_values = np.linspace(soh_range[0], soh_range[1], min(n_samples, 20))
        
        for target_soh in soh_values:
            distances = np.abs(df['SOH'].values - target_soh)
            closest = np.argsort(distances)[:2]
            
            if len(closest) >= 2:
                row1, row2 = df.iloc[closest[0]], df.iloc[closest[1]]
                w1 = 1 / (distances[closest[0]] + 1e-6)
                w2 = 1 / (distances[closest[1]] + 1e-6)
                w_sum = w1 + w2
                
                new_row = row1.copy()
                for col in cols:
                    if col in row1 and col in row2:
                        new_row[col] = (w1 * row1[col] + w2 * row2[col]) / w_sum
                new_row['SOH'] = target_soh
                samples.append(new_row)
        
        return samples