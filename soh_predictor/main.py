"""
메인 실행 모듈
=============

SOH 예측 파이프라인의 진입점
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data_loaders import DataLoaderFactory
from .preprocessors import (
    FrequencyHarmonizer,
    FeatureEngineer,
    DomainAdapter,
    SOHEstimator,
    DataAugmenter,
)
from .models import ModelTrainer, SOHPredictor
from .utils import logger, get_rx_columns


class SOHPredictorPipeline:
    """
    SOH 예측 통합 파이프라인
    
    전체 워크플로우:
        1. 데이터 로드 (다중 데이터셋)
        2. SOH 추정 개선
        3. 주파수 조화
        4. 도메인 적응
        5. 데이터 증강
        6. 피처 엔지니어링
        7. 모델 학습 (다중 시드)
        8. 모델 저장
    
    사용 예시:
        >>> config = Config()
        >>> pipeline = SOHPredictorPipeline(config)
        >>> results = pipeline.run()
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        파이프라인 초기화
        
        Args:
            config: 설정 객체. None이면 기본값 사용
        """
        self.config = config or Config()
        
        # 컴포넌트 초기화
        self.data_loader_factory = DataLoaderFactory(self.config)
        self.harmonizer = FrequencyHarmonizer()
        self.feature_engineer = FeatureEngineer(self.config)
        self.domain_adapter = DomainAdapter()
        self.soh_estimator = SOHEstimator(self.config)
        self.augmenter = DataAugmenter(self.config)
        self.trainer = ModelTrainer(self.config)
        
        # 상태
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.combined_df: Optional[pd.DataFrame] = None
        self.predictor: Optional[SOHPredictor] = None
    
    def run(
        self,
        n_seeds: Optional[int] = None,
        grid_search: bool = True,
        save_model: bool = True,
    ) -> Dict[str, Dict]:
        """
        전체 파이프라인 실행
        
        Args:
            n_seeds: 실험 시드 개수 (기본값: config에서 가져옴)
            grid_search: GridSearchCV 사용 여부
            save_model: 모델 저장 여부
        
        Returns:
            모델별 성능 결과 딕셔너리
        """
        self._print_header()
        
        # 1. 데이터 로드
        self.datasets = self._load_data()
        if not self._validate_datasets():
            return {}
        
        # 2. SOH 추정 개선
        self._improve_soh_estimates()
        
        # 3. 주파수 조화 및 통합
        self.combined_df = self._harmonize_and_combine()
        if self.combined_df is None or self.combined_df.empty:
            logger.error("❌ 통합 데이터가 없습니다.")
            return {}
        
        # 4. 학습 데이터 준비
        train_df = self._prepare_training_data()
        if train_df.empty:
            logger.error("❌ SOH 라벨이 있는 데이터가 없습니다.")
            return {}
        
        # 5. 전처리
        train_df = self._preprocess_data(train_df)
        
        # 6. 피처 �� 타겟 분리
        X, y, groups = self._extract_features_and_target(train_df)
        
        # 7. 다중 시드 실험
        n_seeds = n_seeds or self.config.model.n_seeds
        all_results = self._run_multi_seed_experiments(
            X, y, groups, n_seeds, grid_search
        )
        
        # 8. 결과 집계 및 저장
        aggregated = self._aggregate_results(all_results)
        
        if save_model:
            self._save_model()
        
        # 9. Predictor 생성
        self.predictor = SOHPredictor(self.trainer)
        
        logger.info("\n✅ SOH 예측 모델 개발 완료!")
        return aggregated
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        새 데이터에 대한 SOH 예측
        
        Args:
            X: 입력 피처 DataFrame
        
        Returns:
            SOH 예측값 배열
        """
        if self.predictor is None:
            raise RuntimeError("모델이 학습되지 않았습니다. run()을 먼저 실행하세요.")
        return self.predictor.predict(X)
    
    def _print_header(self) -> None:
        """헤더 출력"""
        logger.info("=" * 80)
        logger.info("🚀 최고의 SOH 예측 모델 개발")
        logger.info("=" * 80)
        logger.info("\n데이터셋 활용 전략:")
        logger.info("  1. Spectroscopy Individual: 실제 SOH 라벨 (주 학습 데이터)")
        logger.info("  2. Company Battery: 개선된 SOH 추정 (보조 학습 데이터)")
        logger.info("  3. Li-Ion SoC Estimation: 새 데이터셋 (pseudo-label)")
        logger.info("=" * 80)
    
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """모든 데이터셋 로드"""
        logger.info("\n📂 데이터 로드 시작...")
        return self.data_loader_factory.load_all()
    
    def _validate_datasets(self) -> bool:
        """데이터셋 유효성 검증"""
        if 'spectroscopy' not in self.datasets or self.datasets['spectroscopy'].empty:
            logger.error("❌ Spectroscopy 데이터가 없습니다. 모델 학습 불가능.")
            return False
        
        logger.info(f"\n📊 로드된 데이터셋: {list(self.datasets.keys())}")
        for name, df in self.datasets.items():
            logger.info(f"  {name}: {len(df)}개 샘플")
        
        return True
    
    def _improve_soh_estimates(self) -> None:
        """보조 데이터셋의 SOH 추정 개선"""
        reference_df = self.datasets.get('spectroscopy')
        
        if reference_df is None:
            return
        
        # Company Battery SOH 개선
        if 'company_battery' in self.datasets:
            self.datasets['company_battery'] = self.soh_estimator.improve_estimates(
                reference_df, self.datasets['company_battery']
            )
        
        # SoC Estimation SOH 개선
        if 'soc_estimation' in self.datasets:
            self.datasets['soc_estimation'] = self.soh_estimator.improve_estimates(
                reference_df, self.datasets['soc_estimation']
            )
    
    def _harmonize_and_combine(self) -> pd.DataFrame:
        """주파수 조화 및 데이터 통합"""
        dataframes = list(self.datasets.values())
        return self.harmonizer.harmonize(*dataframes)
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """학습 데이터 준비"""
        train_df = self.combined_df[self.combined_df['SOH'].notna()].copy()
        
        logger.info(f"\n📊 학습 데이터: {len(train_df)}개 샘플")
        
        if 'data_source' in train_df.columns:
            for source in train_df['data_source'].unique():
                count = len(train_df[train_df['data_source'] == source])
                logger.info(f"  {source}: {count}개")
        
        if 'SOH' in train_df.columns:
            logger.info(f"  SOH 범위: {train_df['SOH'].min():.1f}% ~ {train_df['SOH'].max():.1f}%")
        
        return train_df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리 (도메인 적응 + 증강)"""
        df = self.domain_adapter.adapt(df)
        df = self.augmenter.augment(df, factor=self.config.model.augmentation_factor)
        logger.info(f"\n📊 전처리 후 데이터: {len(df)}개 샘플")
        return df
    
    def _extract_features_and_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """피처, 타겟, 그룹 추출"""
        # 피처 컬럼 추출
        r_cols, x_cols = get_rx_columns(df)
        feature_cols = r_cols + x_cols
        X = df[feature_cols]
        
        # 고급 피처 추출
        X = self.feature_engineer.extract_features(X)
        logger.info(f"\n📈 피처 확장: {len(feature_cols)}개 → {len(X.columns)}개")
        
        # 타겟
        y = df['SOH']
        
        # 그룹 (배터리 단위 분할용)
        groups = None
        for col in ['BATTERY_ID', 'battery_id', 'data_source']:
            if col in df.columns:
                groups = df[col].fillna('unknown').astype(str)
                break
        
        return X, y, groups
    
    def _run_multi_seed_experiments(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series],
        n_seeds: int,
        grid_search: bool,
    ) -> List[Dict]:
        """다중 시드 실험 실행"""
        # 시드 생성
        if grid_search:
            seeds = [random.randint(0, 2**31 - 1) for _ in range(n_seeds)]
            logger.info(f"\n🎯 다중 시드 실험: {seeds}")
        else:
            seeds = [self.config.model.random_state]
        
        all_results = []
        for idx, seed in enumerate(seeds, 1):
            logger.info("\n" + "=" * 60)
            logger.info(f"  Seed {idx}/{len(seeds)}: random_state={seed}")
            logger.info("=" * 60)
            
            results = self.trainer.train(
                X, y,
                groups=groups,
                use_cv=True,
                grid_search=grid_search,
                random_state=seed,
            )
            all_results.append(results)
        
        return all_results
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Dict]:
        """다중 시드 결과 집계"""
        if not all_results:
            return {}
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 다중 시드 실험 결과 집계")
        logger.info("=" * 60)
        
        aggregated = {}
        model_names = all_results[0].keys()
        
        for model_name in model_names:
            mae_values = [r[model_name]['mae_mean'] for r in all_results]
            r2_values = [r[model_name]['r2_mean'] for r in all_results]
            
            aggregated[model_name] = {
                'mae_mean': np.mean(mae_values),
                'mae_std': np.std(mae_values),
                'r2_mean': np.mean(r2_values),
                'r2_std': np.std(r2_values),
            }
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  MAE: {aggregated[model_name]['mae_mean']:.4f}% (±{aggregated[model_name]['mae_std']:.4f}%)")
            logger.info(f"  R²: {aggregated[model_name]['r2_mean']:.4f} (±{aggregated[model_name]['r2_std']:.4f})")
        
        return aggregated
    
    def _save_model(self) -> None:
        """모델 저장"""
        output_dir = self.config.paths.model_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / "ultimate_soh_model.pkl"
        self.trainer.save(filepath)
        logger.info(f"\n✅ 모델 저장: {filepath}")


def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="배터리 SOH 예측 모델 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (GridSearch + 5 시드)
  python -m soh_predictor
  
  # 빠른 실행 (GridSearch 없이)
  python -m soh_predictor --fast
  
  # 시드 개수 지정
  python -m soh_predictor --n-seeds 10
  
  # 커스텀 베이스 디렉토리
  python -m soh_predictor --base-dir /path/to/data
        """
    )
    
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="데이터셋 베이스 디렉토리 (기본: 현재 디렉토리)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="실험 시드 개수 (기본: 5)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="빠른 실행 (GridSearch 비활성화)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="모델 저장 비활성화",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 로깅",
    )
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 설정 생성
    from .config import PathConfig
    config = Config(paths=PathConfig(base_dir=args.base_dir))
    
    # 파이프라인 실행
    pipeline = SOHPredictorPipeline(config)
    results = pipeline.run(
        n_seeds=args.n_seeds if not args.fast else 1,
        grid_search=not args.fast,
        save_model=not args.no_save,
    )
    
    return results


if __name__ == "__main__":
    main()