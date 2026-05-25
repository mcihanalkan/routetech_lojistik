import pandas as pd
import numpy as np  
import logging
from typing import Optional, Any, Dict, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def validate_inputs(
    df: pd.DataFrame,
    strategy: str,
    columns: list,
    fill_value: Optional[Any],
    groupby_column: Optional[str],
    numeric_only: bool
) -> None:
    """Tüm parametreleri valide et"""
    
    valid_strategies = [
        "mean", "median", "mode", "forward_fill", "backward_fill", 
        "constant", "knn", "drop"
    ]
    
    if strategy not in valid_strategies:
        raise ValueError(
            f"❌ Bilinmeyen strategy: '{strategy}'\n"
            f"   Geçerli: {valid_strategies}"
        )
    
    invalid_columns = [col for col in columns if col not in df.columns]
    if invalid_columns:
        raise ValueError(
            f"❌ Sütunlar bulunamadı: {invalid_columns}\n"
            f"   Mevcut sütunlar: {list(df.columns)}"
        )
    
    if groupby_column and groupby_column not in df.columns:
        raise ValueError(
            f"❌ groupby_column '{groupby_column}' bulunamadı!\n"
            f"   Mevcut sütunlar: {list(df.columns)}"
        )
    
    if strategy == "constant" and fill_value is None:
        raise ValueError(
            "❌ strategy='constant' için fill_value gerekli!\n"
            f"   Örnek: fill_value=0 or fill_value='unknown'"
        )


def fill_forward_fill(
    df: pd.DataFrame,
    columns: list,
    logging_enabled: bool
) -> pd.DataFrame:
    """Forward fill"""
    missing_counts = df[columns].isnull().sum()
    df.loc[:, columns] = df[columns].ffill().bfill()
    
    if logging_enabled:
        for col in columns:
            if missing_counts[col] > 0:
                logger.info(f"   📊 {col}: {missing_counts[col]} → ffill")
    
    return df


def fill_backward_fill(
    df: pd.DataFrame,
    columns: list,
    logging_enabled: bool
) -> pd.DataFrame:
    """Backward fill"""
    missing_counts = df[columns].isnull().sum()
    df.loc[:, columns] = df[columns].bfill().ffill()
    
    if logging_enabled:
        for col in columns:
            if missing_counts[col] > 0:
                logger.info(f"   📊 {col}: {missing_counts[col]} → bfill")
    
    return df


def drop_missing(
    df: pd.DataFrame,
    columns: list,
    logging_enabled: bool
) -> pd.DataFrame:
    """Eksik satırları sil"""
    rows_before = len(df)
    mask = df[columns].isnull().any(axis=1)
    df = df[~mask].reset_index(drop=True)
    rows_after = len(df)
    
    if logging_enabled and rows_before > rows_after:
        logger.info(f"   🗑️  {rows_before - rows_after} satır silindi")
    
    return df


def remove_outliers(
    df: pd.DataFrame,
    columns: list,
    threshold: float = 3.0
) -> pd.DataFrame:
    """Z-score ile aykırı değerleri kaldır"""
    df = df.copy()
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())  # ⭐ np.abs
        mask = mask & (z_scores < threshold)
    
    return df[mask].reset_index(drop=True)


def train_test_split_timeseries(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> tuple:
    """Zaman serisi için split"""
    total_rows = len(df)
    split_idx = int(total_rows * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def get_column_transformer_pipeline(
    numeric_transformer: Optional[Pipeline] = None,
    categorical_transformer: Optional[Pipeline] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    COLUMNTRANSFORMER - Different transformers for different types
    
    Examples:
    ---------
    >>> from src.missing import DataPreprocessor
    >>> from src.scaling import FeatureScaler
    >>> numeric_pipe = Pipeline([
    ...     ('impute', DataPreprocessor(strategy='mean')),
    ...     ('scale', FeatureScaler(method='minmax'))
    ... ])
    >>> categorical_pipe = Pipeline([
    ...     ('encode', CategoricalEncoder(method='onehot'))
    ... ])
    >>> ct = get_column_transformer_pipeline(
    ...     numeric_transformer=numeric_pipe,
    ...     categorical_transformer=categorical_pipe,
    ...     numeric_features=['fuel', 'distance'],
    ...     categorical_features=['vehicle_type']
    ... )
    >>> ct.fit(X_train)
    >>> X_train_clean = ct.transform(X_train)
    """
    
    transformers = []
    
    if numeric_features and numeric_transformer:
        transformers.append((
            'num',
            numeric_transformer,
            numeric_features
        ))
    
    if categorical_features and categorical_transformer:
        transformers.append((
            'cat',
            categorical_transformer,
            categorical_features
        ))
    
    return ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )