import pandas as pd
import numpy as np
from typing import Optional, List
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Categorical encoding:
    ⭐ catboost: No matrix expansion, native support (BEST FOR HIZ/RAM)
    ⭐ onehot: No ordinal bias (Good for deep learning/linear models)
    ⚠️  label: Ordinal bias risk
    """
    
    def __init__(
        self,
        method: str = "catboost",  # Varsayılan metot artık hız/RAM dostu catboost!
        columns: Optional[List[str]] = None,
        nan_value: str = "unknown",
        logging_enabled: bool = False,
        drop: Optional[str] = None
    ):
        self.method = method
        self.columns = columns
        self.nan_value = nan_value
        self.logging_enabled = logging_enabled
        self.drop = drop
        self.encoder_ = None
        self.encoder_mapping_ = {}
        self.fitted_columns_ = None
    
    def fit(self, X: pd.DataFrame, y=None) -> 'CategoricalEncoder':
        """Kategorileri öğren"""
        if X is None or X.empty:
            raise ValueError("❌ Boş DataFrame!")
        
        if self.columns is None:
            columns = list(X.select_dtypes(include=['object', 'category']).columns)
        else:
            columns = list(self.columns)
        
        if not columns:
            self.fitted_columns_ = []
            return self
            
        self.fitted_columns_ = columns
        
        if self.method == "catboost":
            if self.logging_enabled:
                logger.info(f"⚡ CatBoost Modu: {len(columns)} sütun One-Hot yapılmadan (RAM dostu) iletilecek.")
        
        elif self.method == "onehot":
            X_prep = X.copy()
            for col in columns:
                X_prep.loc[X_prep[col].isnull(), col] = self.nan_value
            
            self.encoder_ = OneHotEncoder(
                handle_unknown="ignore",
                 sparse_output=False,
                drop=self.drop
            )
            self.encoder_.fit(X_prep[columns])
            
            if self.logging_enabled:
                n_features = len(self.encoder_.get_feature_names_out(columns))
                logger.info(f"✅ OneHot fit: {n_features} features (⚠️ RAM uyarısı!)")
        
        elif self.method == "label":
            logger.warning(
                "⚠️  Label encoding kullanılıyor!\n"
                f"    RISK: Ordinal bias (0<1<2...)\n"
                f"    ÇÖZÜM: method='catboost' veya 'onehot' kullanın"
            )
            
            X_prep = X.copy()
            for col in columns:
                X_prep.loc[X_prep[col].isnull(), col] = self.nan_value
                unique_cats = sorted(X_prep[col].unique())
                self.encoder_mapping_[col] = {
                    cat: idx for idx, cat in enumerate(unique_cats)
                }
        
        else:
            raise ValueError(f"❌ Bilinmeyen method: '{self.method}'")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform"""
        if not hasattr(self, 'fitted_columns_') or self.fitted_columns_ is None:
            raise ValueError("❌ Önce fit() çağrılmalı!")
        
        if X is None or X.empty or not self.fitted_columns_:
            return X.copy()
        
        X = X.copy()
        columns = self.fitted_columns_
        
        if self.method == "catboost":
            # CatBoost string veya integer türündeki kategorikleri sever
            for col in columns:
                X[col] = X[col].fillna(self.nan_value).astype(str)
        
        elif self.method == "onehot":
            X_prep = X.copy()
            for col in columns:
                X_prep.loc[X_prep[col].isnull(), col] = self.nan_value
            
            onehot_array = self.encoder_.transform(X_prep[columns])
            onehot_df = pd.DataFrame(
                onehot_array,
                columns=self.encoder_.get_feature_names_out(columns),
                index=X.index
            )
            
            X = X.drop(columns, axis=1)
            X = pd.concat([X, onehot_df], axis=1)
        
        elif self.method == "label":
            for col in columns:
                X.loc[X[col].isnull(), col] = self.nan_value
                X[col] = X[col].map(self.encoder_mapping_[col])
                X.loc[:, col] = X[col].fillna(-1).astype(int)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)