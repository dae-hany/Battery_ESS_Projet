"""
모델 모듈 (완성본)
==================

SOH 예측 모델 학습, 평가, 예측
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression

from .config import Config
from .utils import logger

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelTrainer:
    """모델 학습기"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_config = config.model
        self.models: Dict[str, Any] = {}
        self.scaler = RobustScaler()
        self.feature_selector: Optional[SelectKBest] = None
        self.feature_names: List[str] = []
        self.ensemble_weights: Dict[str, float] = {}
        self._X_scaled: Optional[np.ndarray] = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
        use_cv: bool = True,
        grid_search: bool = True,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """모델 학습"""
        random_state = random_state or self.model_config.random_state
        logger.info(f"🚀 모델 학습 시작 (random_state={random_state})")
        
        # 전처리
        self._X_scaled = self._preprocess_features(X, y)
        
        # 교차 검증 설정
        cv_splits = self._setup_cross_validation(self._X_scaled, y, groups, random_state)
        
        # 모델 학습
        results = {}
        results.update(self._train_random_forest(self._X_scaled, y, cv_splits, random_state, grid_search))
        results.update(self._train_gradient_boosting(self._X_scaled, y, cv_splits, random_state, grid_search))
        
        if XGBOOST_AVAILABLE and grid_search:
            results.update(self._train_xgboost(self._X_scaled, y, cv_splits, random_state))
        
        if LIGHTGBM_AVAILABLE and grid_search:
            results.update(self._train_lightgbm(self._X_scaled, y, cv_splits, random_state))
        
        if grid_search and len(self.models) >= 2:
            results.update(self._train_stacking(self._X_scaled, y, cv_splits))
        
        # 앙상블 가중치 계산
        self._compute_ensemble_weights(results)
        
        self._log_results(results)
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """SOH 예측"""
        X_processed = self._prepare_for_prediction(X)
        
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X_processed)
        
        # 앙상블 예측
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            weight = self.ensemble_weights.get(model_name, 1.0 / len(self.models))
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    def save(self, filepath: Path) -> None:
        """모델 저장"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'ensemble_weights': self.ensemble_weights,
            'config': self.config,
        }
        joblib.dump(model_data, filepath)
        logger.info(f"✅ 모델 저장 완료: {filepath}")
    
    def load(self, filepath: Path) -> None:
        """모델 로드"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data.get('feature_selector')
        self.feature_names = model_data['feature_names']
        self.ensemble_weights = model_data['ensemble_weights']
        logger.info(f"✅ 모델 로드 완료: {filepath}")
    
    def _preprocess_features(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """피처 전처리"""
        logger.info("📊 피처 선택 중...")
        
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        n_features = min(self.model_config.n_features_max, len(X_clean.columns))
        self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        X_selected = self.feature_selector.fit_transform(X_clean, y)
        
        self.feature_names = X_clean.columns[self.feature_selector.get_support()].tolist()
        logger.info(f"  {len(X_clean.columns)}개 → {len(self.feature_names)}개 피처")
        
        return self.scaler.fit_transform(X_selected)
    
    def _prepare_for_prediction(self, X: pd.DataFrame) -> np.ndarray:
        """예측을 위한 데이터 준비"""
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 누락된 피처 추가
        for col in self.feature_names:
            if col not in X_clean.columns:
                X_clean[col] = 0
        
        # 피처 선택 적용
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_clean)
        else:
            X_selected = X_clean[self.feature_names].values
        
        return self.scaler.transform(X_selected)
    
    def _setup_cross_validation(
        self,
        X: np.ndarray,
        y: pd.Series,
        groups: Optional[pd.Series],
        random_state: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """교차 검증 설정"""
        n_splits = self.model_config.cv_folds
        
        if groups is not None:
            n_groups = len(np.unique(groups))
            n_splits = min(n_splits, n_groups)
            
            if n_splits >= 2:
                cv = GroupKFold(n_splits=n_splits)
                logger.info(f"📊 GroupKFold ({n_splits}-fold, {n_groups}개 그룹)")
                return list(cv.split(X, y, groups))
        
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        logger.info(f"📊 KFold ({n_splits}-fold)")
        return list(cv.split(X, y))
    
    def _train_random_forest(
        self,
        X: np.ndarray,
        y: pd.Series,
        cv_splits: List,
        random_state: int,
        grid_search: bool,
    ) -> Dict[str, Dict]:
        """Random Forest 학습"""
        logger.info("🌲 Random Forest 학습 중...")
        
        params = self.model_config.rf_params if grid_search else {
            k: [v[0]] for k, v in self.model_config.rf_params.items()
        }
        
        grid = GridSearchCV(
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            params, cv=cv_splits, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0
        )
        grid.fit(X, y)
        self.models['RandomForest'] = grid.best_estimator_
        
        logger.info(f"  ✅ 최적 파라미터: {grid.best_params_}")
        return {'RandomForest': self._evaluate_model(grid.best_estimator_, X, y, cv_splits)}
    
    def _train_gradient_boosting(
        self,
        X: np.ndarray,
        y: pd.Series,
        cv_splits: List,
        random_state: int,
        grid_search: bool,
    ) -> Dict[str, Dict]:
        """Gradient Boosting 학습"""
        logger.info("📈 Gradient Boosting 학습 중...")
        
        params = self.model_config.gb_params if grid_search else {
            k: [v[0]] for k, v in self.model_config.gb_params.items()
        }
        
        grid = GridSearchCV(
            GradientBoostingRegressor(random_state=random_state),
            params, cv=cv_splits, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0
        )
        grid.fit(X, y)
        self.models['GradientBoosting'] = grid.best_estimator_
        
        logger.info(f"  ✅ 최적 파라미터: {grid.best_params_}")
        return {'GradientBoosting': self._evaluate_model(grid.best_estimator_, X, y, cv_splits)}
    
    def _train_xgboost(
        self,
        X: np.ndarray,
        y: pd.Series,
        cv_splits: List,
        random_state: int,
    ) -> Dict[str, Dict]:
        """XGBoost 학습"""
        logger.info("⚡ XGBoost 학습 중...")
        
        try:
            grid = GridSearchCV(
                xgb.XGBRegressor(random_state=random_state, n_jobs=-1),
                self.model_config.xgb_params,
                cv=cv_splits, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0
            )
            grid.fit(X, y)
            self.models['XGBoost'] = grid.best_estimator_
            
            logger.info(f"  ✅ 최적 파라미터: {grid.best_params_}")
            return {'XGBoost': self._evaluate_model(grid.best_estimator_, X, y, cv_splits)}
        except Exception as e:
            logger.warning(f"  ⚠️ XGBoost 학습 실패: {e}")
            return {}
    
    def _train_lightgbm(
        self,
        X: np.ndarray,
        y: pd.Series,
        cv_splits: List,
        random_state: int,
    ) -> Dict[str, Dict]:
        """LightGBM 학습"""
        logger.info("💡 LightGBM 학습 중...")
        
        try:
            grid = GridSearchCV(
                lgb.LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1),
                self.model_config.lgb_params,
                cv=cv_splits, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0
            )
            grid.fit(X, y)
            self.models['LightGBM'] = grid.best_estimator_
            
            logger.info(f"  ✅ 최적 파라미터: {grid.best_params_}")
            return {'LightGBM': self._evaluate_model(grid.best_estimator_, X, y, cv_splits)}
        except Exception as e:
            logger.warning(f"  ⚠️ LightGBM 학습 실패: {e}")
            return {}
    
    def _train_stacking(
        self,
        X: np.ndarray,
        y: pd.Series,
        cv_splits: List,
    ) -> Dict[str, Dict]:
        """Stacking 앙상블 학습"""
        logger.info("🎯 Stacking 앙상블 구성 중...")
        
        estimators = [(name.lower(), model) for name, model in self.models.items()]
        
        try:
            stacking = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=cv_splits, n_jobs=-1
            )
            stacking.fit(X, y)
            self.models['Stacking'] = stacking
            
            logger.info("  ✅ Stacking 앙상블 완료")
            return {'Stacking': self._evaluate_model(stacking, X, y, cv_splits)}
        except Exception as e:
            logger.warning(f"  ⚠️ Stacking 앙상블 실패: {e}")
            return {}
    
    def _evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: pd.Series,
        cv_splits: List,
    ) -> Dict[str, float]:
        """모델 평가"""
        mae_scores = cross_val_score(model, X, y, cv=cv_splits, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X, y, cv=cv_splits, scoring='r2')
        
        result = {
            'mae_mean': -mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
        }
        
        logger.info(f"  CV MAE: {result['mae_mean']:.4f}% (±{result['mae_std']:.4f}%)")
        logger.info(f"  CV R²: {result['r2_mean']:.4f} (±{result['r2_std']:.4f})")
        
        return result
    
    def _compute_ensemble_weights(self, results: Dict[str, Dict]) -> None:
        """앙상블 가중치 계산 (MAE 기반)"""
        if not results:
            return
        
        total_mae = sum(r['mae_mean'] for r in results.values())
        
        for name, res in results.items():
            weight = (1 - res['mae_mean'] / total_mae) / len(results) if total_mae > 0 else 1 / len(results)
            self.ensemble_weights[name] = max(weight, 0)
        
        # 정규화
        total = sum(self.ensemble_weights.values())
        if total > 0:
            self.ensemble_weights = {k: v / total for k, v in self.ensemble_weights.items()}
        
        logger.info("\n⚖️ 앙상블 가중치:")
        for name, weight in self.ensemble_weights.items():
            logger.info(f"  {name}: {weight:.3f}")
    
    def _log_results(self, results: Dict[str, Dict]) -> None:
        """결과 로깅"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 최종 모델 성능:")
        logger.info("-" * 60)
        
        for name, res in results.items():
            logger.info(
                f"  {name:15s}: MAE {res['mae_mean']:.4f}% (±{res['mae_std']:.4f}%), "
                f"R² {res['r2_mean']:.4f} (±{res['r2_std']:.4f})"
            )
        
        if results:
            best = min(results.keys(), key=lambda k: results[k]['mae_mean'])
            logger.info(f"\n🏆 최고 모델: {best}")


class SOHPredictor:
    """SOH 예측기 (학습된 ModelTrainer 래퍼)"""
    
    def __init__(self, trainer: ModelTrainer):
        """
        Args:
            trainer: 학습 완료된 ModelTrainer 인스턴스
        """
        self.trainer = trainer
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        SOH 예측
        
        Args:
            X: 입력 피처 DataFrame
        
        Returns:
            SOH 예측값 배열 (%)
        """
        return self.trainer.predict(X)
    
    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        불확실성을 포함한 SOH 예측
        
        Args:
            X: 입력 피처 DataFrame
        
        Returns:
            (예측값 배열, 표준편차 배열) 튜플
        """
        X_processed = self.trainer._prepare_for_prediction(X)
        
        predictions = []
        for model in self.trainer.models.values():
            predictions.append(model.predict(X_processed))
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred
    
    @classmethod
    def load(cls, filepath: Path, config: Optional[Config] = None) -> 'SOHPredictor':
        """
        저장된 모델 로드
        
        Args:
            filepath: 모델 파일 경로
            config: 설정 객체 (None이면 저장된 설정 사용)
        
        Returns:
            SOHPredictor 인스턴스
        """
        from .config import Config
        
        config = config or Config()
        trainer = ModelTrainer(config)
        trainer.load(filepath)
        
        return cls(trainer)