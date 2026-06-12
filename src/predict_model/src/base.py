# src/base.py

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import logging
from copy import deepcopy
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class BaseForecaster(ABC, BaseEstimator):
    """CatBoost ve Karar Odaklı Öğrenme için Saf Pandas Tabanlı Soyut Sınıf"""
    
    def __init__(
        self,
        target_column: str = "desi_hacmi",
        date_column: str = "date",
        group_column: Optional[str] = "TM_ID",
        train_test_split: float = 0.8,
        forecast_horizon: int = 7,
        logging_enabled: bool = True,
        random_state: Optional[int] = 42
    ):
        self.target_column = target_column
        self.date_column = date_column
        self.group_column = group_column
        self.train_test_split = train_test_split
        self.forecast_horizon = forecast_horizon
        self.logging_enabled = logging_enabled
        self.random_state = random_state
        
        if random_state is not None:
            self.rng_ = np.random.default_rng(random_state)
        else:
            self.rng_ = np.random.default_rng()
        
        self.is_fitted_ = False
        self.model_ = None
        self.backtest_results_ = {}
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Input validasyonu - Darts TimeSeries yerine Saf Pandas"""
        if X is None or X.empty:
            raise ValueError("❌ Boş DataFrame!")
        
        if self.date_column not in X.columns:
            raise ValueError(f"❌ '{self.date_column}' sütunu bulunamadı!")
        
        if self.target_column not in X.columns:
            raise ValueError(f"❌ '{self.target_column}' sütunu bulunamadı!")
        
        date_col_copy = X[self.date_column].copy()
        if not pd.api.types.is_datetime64_any_dtype(date_col_copy):
            result = pd.to_datetime(date_col_copy, errors='coerce')
            if result.isna().all():
                raise ValueError(
                    f"❌ '{self.date_column}' sütunu datetime'a çevrilemedi!"
                )
        
        if not pd.api.types.is_numeric_dtype(X[self.target_column]):
            raise ValueError(f"❌ '{self.target_column}' sütunu numeric değil!")
            
        if self.logging_enabled:
            logger.info(f"✅ Input validasyonu: {len(X)} satır")
            
    def _prepare_dataframe(self, df: pd.DataFrame, group_id: Optional[Any] = None) -> pd.DataFrame:
        """Zaman serisi sıralaması ve indexleme"""
        df = df.copy()
        if group_id is not None and self.group_column:
            df = df[df[self.group_column] == group_id].copy()
            
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            
        df = df.sort_values(self.date_column).reset_index(drop=True)
        return df

    def _train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Zaman bazlı (Walk-Forward) split - TM_ID bağımsız, kesin tarih ayrımı.
        
        Veriyi satır sayısına göre DEĞİL benzersiz tarihlerin sıralı dağılımına
        göre böler. Böylece train seti her zaman geçmiş, test seti gelecek
        tarihlerden oluşur; hiçbir şubenin (TM_ID) gelecek verisi train'e sızmaz.
        """
        unique_dates = df[self.date_column].sort_values().unique()
        split_idx = int(len(unique_dates) * self.train_test_split)
        # Test setinin boş kalmaması için sınır koruması
        split_idx = min(split_idx, len(unique_dates) - 1)
        split_date = unique_dates[split_idx]
        
        train = df[df[self.date_column] < split_date].copy()
        test = df[df[self.date_column] >= split_date].copy()
        
        if self.logging_enabled:
            logger.info(
                f"✅ Walk-Forward split tamamlandı:\n"
                f"   Split tarihi : {split_date}\n"
                f"   Train        : {len(train):,} satır "
                f"({df[self.date_column].min()} → train sonu)\n"
                f"   Test         : {len(test):,} satır "
                f"(split_date → {df[self.date_column].max()})"
            )
        
        return train, test
    
    def backtest(
        self,
        df: pd.DataFrame,
        num_backtests: int = 3,
        stride: Optional[int] = None
    ) -> Dict[str, Any]:
        """Saf Pandas ile Time Series Cross Validation (Walk-Forward).

        Pencere sınırları satır sayısına göre DEĞİL, benzersiz tarih listesi
        üzerinden kaydırılır. Böylece _train_test_split ile tutarlı şekilde
        her pencerede de günün tüm şubeleri (TM_ID) birlikte train veya
        test tarafına düşer; günün ortasında kesim yapılmaz.
        """
        if not self.is_fitted_ or self.model_ is None:
            raise ValueError("❌ Önce fit() çağrılmalı ve model kurulmalı!")

        df = self._prepare_dataframe(df)

        # Tüm pencere hesaplamaları unique date ekseninde yapılır
        unique_dates = df[self.date_column].sort_values().unique()
        n_dates = len(unique_dates)

        if stride is None:
            stride = max(n_dates // (num_backtests + 2), self.forecast_horizon)

        results = {
            'num_backtests': num_backtests,
            'stride': stride,          # gün cinsinden stride
            'errors': [],
            'successful_splits': 0
        }

        if self.logging_enabled:
            logger.info(
                f"🔄 Backtesting başlıyor ({num_backtests} split)...\n"
                f"   Stride: {stride} gün | Toplam benzersiz tarih: {n_dates}"
            )

        for i in range(num_backtests):
            # Pencere sınırları tarih dizisi üzerinden belirlenir
            train_end_date_idx = (i + 1) * stride
            test_end_date_idx  = min(train_end_date_idx + self.forecast_horizon, n_dates)

            if train_end_date_idx >= n_dates or test_end_date_idx <= train_end_date_idx:
                continue

            train_cutoff = unique_dates[train_end_date_idx]
            test_cutoff  = unique_dates[test_end_date_idx - 1]

            # Tarih bazlı filtreleme → günün tüm TM_ID'leri birlikte geçer
            train_backtest = df[df[self.date_column] < train_cutoff]
            test_backtest  = df[
                (df[self.date_column] >= train_cutoff) &
                (df[self.date_column] <= test_cutoff)
            ]

            if train_backtest.empty or test_backtest.empty:
                continue

            try:
                model_copy = deepcopy(self.model_)
                # Gelecekteki CatBoost yapısına uygun X, y ayırma mantığı
                X_train = train_backtest.drop(columns=[self.target_column, self.date_column])
                y_train = train_backtest[self.target_column]
                X_test  = test_backtest.drop(columns=[self.target_column, self.date_column])
                y_test  = test_backtest[self.target_column]

                model_copy.fit(X_train, y_train)
                preds = model_copy.predict(X_test)

                mae  = np.mean(np.abs(y_test.values - preds))
                rmse = np.sqrt(np.mean((y_test.values - preds) ** 2))

                results['errors'].append({
                    'mae': mae,
                    'rmse': rmse,
                    'train_cutoff': str(train_cutoff),
                    'test_cutoff':  str(test_cutoff),
                })
                results['successful_splits'] += 1

                if self.logging_enabled:
                    logger.info(
                        f"   Backtest {i+1}: train < {train_cutoff} | "
                        f"test [{train_cutoff} → {test_cutoff}] | "
                        f"MAE={mae:.4f} RMSE={rmse:.4f}"
                    )

            except Exception as e:
                logger.error(f"❌ Backtest {i+1} hatası: {str(e)}")
                results['errors'].append({'error': str(e)})
                continue

        if results['successful_splits'] == 0:
            raise ValueError("❌ Backtest başarısız! Hiçbir split çalışmadı")

        self.backtest_results_ = results
        return results

    @abstractmethod
    def _build_model(self) -> None:
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Parametreleri standart logger'a yazar (mlflow bağımlılığı kaldırıldı)."""
        if self.logging_enabled:
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    logger.info(f"  param | {key}: {value}")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            'target_column': self.target_column,
            'date_column': self.date_column,
            'group_column': self.group_column,
            'train_test_split': self.train_test_split,
            'forecast_horizon': self.forecast_horizon,
            'logging_enabled': self.logging_enabled,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'BaseForecaster':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if 'random_state' in params:
            if params['random_state'] is not None:
                self.rng_ = np.random.default_rng(params['random_state'])
            else:
                self.rng_ = np.random.default_rng()
        return self
    
__all__ = ['BaseForecaster']
