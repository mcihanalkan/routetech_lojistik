import pandas as pd
import numpy as np
from typing import Optional, List
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Sklearn-uyumlu feature scaling"""
    
    def __init__(
        self,
        method: str = "minmax",
        columns: Optional[List[str]] = None,
        logging_enabled: bool = False
    ):
        self.method = method
        self.columns = columns
        self.logging_enabled = logging_enabled
        self.scaler_ = None
        self.fitted_columns_ = None
    
    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureScaler':
        """Scaler'ı fit et"""
        if X is None or X.empty:
            raise ValueError("❌ Boş DataFrame!")
        
        if self.columns is None:
            columns = list(X.select_dtypes(include=[np.number]).columns)
        else:
            columns = list(self.columns)
        
        if not columns:
            raise ValueError("❌ Numeric sütun bulunamadı!")
        
        self.fitted_columns_ = columns
        
        if self.method == "minmax":
            self.scaler_ = MinMaxScaler()
        elif self.method == "zscore":
            self.scaler_ = StandardScaler()
        else:
            raise ValueError(
                f"❌ Bilinmeyen method: '{self.method}'\n"
                f"   Geçerli: 'minmax', 'zscore'"
            )
        
        self.scaler_.fit(X[columns])
        
        if self.logging_enabled:
            logger.info(f"✅ FeatureScaler fit: {self.method} ({len(columns)} sütun)")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scaling'i uygula"""
        if not hasattr(self, 'fitted_columns_') or self.fitted_columns_ is None:
            raise ValueError("❌ Önce fit() çağrılmalı!")
        
        if X is None or X.empty:
            return X.copy()
        
        missing_cols = set(self.fitted_columns_) - set(X.columns)
        if missing_cols:
            raise ValueError(
                f"❌ Transform'da sütunlar eksik: {missing_cols}\n"
                f"   Fit edilen: {self.fitted_columns_}\n"
                f"   Mevcut: {list(X.columns)}"
            )
        
        X = X.copy()
        X[self.fitted_columns_] = self.scaler_.transform(X[self.fitted_columns_])
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)