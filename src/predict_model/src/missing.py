import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Eksik değer işleme - 8 strateji
    
    fit()/transform() sklearn pattern ile data leakage önleme
    """
    
    def __init__(
        self,
        strategy: str = "mean",
        columns: Optional[List[str]] = None,
        fill_value: Optional[Any] = None,
        groupby_column: Optional[str] = None,
        numeric_only: bool = True,
        knn_neighbors: int = 5,
        logging_enabled: bool = False
    ):
        self.strategy = strategy
        self.columns = columns
        self.fill_value = fill_value
        self.groupby_column = groupby_column
        self.numeric_only = numeric_only
        self.knn_neighbors = knn_neighbors
        self.logging_enabled = logging_enabled
        
        # Fit sırasında kaydedilecek
        self.fill_values_ = {}
        self.global_fill_value_ = None
        self.scaler_ = None
        self.imputer_ = None
        self.fitted_columns_ = None
        self._knn_columns = None
    
    def fit(self, X: pd.DataFrame, y=None) -> 'DataPreprocessor':
        """Training veriden parametreleri öğren"""
        if X is None or X.empty:
            raise ValueError("❌ Boş DataFrame!")
        
        if self.columns is None:
            columns = list(X.columns)
        else:
            columns = list(self.columns)
        
        from src.utils import validate_inputs
        validate_inputs(X, self.strategy, columns, self.fill_value, 
                       self.groupby_column, self.numeric_only)
        
        if self.numeric_only and self.strategy in ["mean", "median", "knn"]:
            columns = list(X[columns].select_dtypes(include=[np.number]).columns)
        
        self.fitted_columns_ = columns
        
        if self.logging_enabled:
            logger.info(f"🔧 Fit: strategy={self.strategy}, columns={len(columns)}")
        
        if self.strategy == "mean":
            self._fit_mean(X, columns)
        elif self.strategy == "median":
            self._fit_median(X, columns)
        elif self.strategy == "mode":
            self._fit_mode(X, columns)
        elif self.strategy == "knn":
            self._fit_knn(X, columns)
        elif self.strategy == "constant":
            self._fit_constant()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform - LEAKAGE YOK!"""
        if not hasattr(self, 'fitted_columns_') or self.fitted_columns_ is None:
            raise ValueError("❌ Önce fit() çağrılmalı!")
        
        if X is None or X.empty:
            return X.copy()
        
        X = X.copy()
        columns = list(self.fitted_columns_)
        
        if self.strategy == "mean":
            X = self._transform_mean(X, columns)
        elif self.strategy == "median":
            X = self._transform_median(X, columns)
        elif self.strategy == "mode":
            X = self._transform_mode(X, columns)
        elif self.strategy == "forward_fill":
            from src.utils import fill_forward_fill
            X = fill_forward_fill(X, columns, self.logging_enabled)
        elif self.strategy == "backward_fill":
            from src.utils import fill_backward_fill
            X = fill_backward_fill(X, columns, self.logging_enabled)
        elif self.strategy == "constant":
            X = self._transform_constant(X, columns)
        elif self.strategy == "knn":
            X = self._transform_knn(X, columns)
        elif self.strategy == "drop":
            from src.utils import drop_missing
            X = drop_missing(X, columns, self.logging_enabled)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    
    def _fit_mean(self, X: pd.DataFrame, columns: list) -> None:
        """Mean fit - Training parametreleri kaydet"""
        try:
            if self.groupby_column:
                self.fill_values_ = {
                    col: X.groupby(self.groupby_column)[col].mean().to_dict() 
                    for col in columns
                }
            else:
                self.fill_values_ = {col: X[col].mean() for col in columns}
            
            self.global_fill_value_ = {col: X[col].mean() for col in columns}
        except TypeError as e:
            raise ValueError(
                f"❌ '{columns}' sütunlarında mean hesaplanamıyor!\n"
                f"   Sebepler: Numeric olmayan veri tipi\n"
                f"   Çözüm: numeric_only=True kullanın"
            )
    
    def _fit_median(self, X: pd.DataFrame, columns: list) -> None:
        """Median fit"""
        try:
            if self.groupby_column:
                self.fill_values_ = {
                    col: X.groupby(self.groupby_column)[col].median().to_dict() 
                    for col in columns
                }
            else:
                self.fill_values_ = {col: X[col].median() for col in columns}
            
            self.global_fill_value_ = {col: X[col].median() for col in columns}
        except TypeError as e:
            raise ValueError(
                f"❌ '{columns}' sütunlarında median hesaplanamıyor!\n"
                f"   dtype kontrol edin"
            )
    
    def _fit_mode(self, X: pd.DataFrame, columns: list) -> None:
        """Mode fit"""
        if self.groupby_column:
            self.fill_values_ = {}
            for col in columns:
                group_modes = {}
                for group_name, group_df in X.groupby(self.groupby_column):
                    mode_vals = group_df[col].mode()
                    group_modes[group_name] = (
                        mode_vals.iloc[0] if len(mode_vals) > 0 else group_df[col].iloc[0]
                    )
                self.fill_values_[col] = group_modes
        else:
            self.fill_values_ = {}
            for col in columns:
                mode_vals = X[col].mode()
                self.fill_values_[col] = (
                    mode_vals.iloc[0] if len(mode_vals) > 0 else X[col].iloc[0]
                )
        
        self.global_fill_value_ = {
            col: self.fill_values_.get(col, X[col].iloc[0]) for col in columns
        }
    
    def _fit_knn(self, X: pd.DataFrame, columns: list) -> None:
        """KNN fit"""
        numeric_cols = list(X[columns].select_dtypes(include=[np.number]).columns)
        
        if not numeric_cols:
            raise ValueError(f"❌ KNN için numeric sütun yok! ({columns})")
        
        self._knn_columns = numeric_cols
        
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[numeric_cols])
        
        X_scaled = X.copy()
        X_scaled.loc[:, numeric_cols] = self.scaler_.transform(X[numeric_cols])
        
        self.imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
        self.imputer_.fit(X_scaled[numeric_cols])
        
        if self.logging_enabled:
            logger.info(f"   KNN fit: {self._knn_columns}")
    
    def _fit_constant(self) -> None:
        """Constant fit"""
        if self.fill_value is None:
            raise ValueError("❌ strategy='constant' için fill_value gerekli!")
        self.fill_values_ = {"constant": self.fill_value}
    
    def _transform_mean(self, X: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Mean transform - LEAKAGE YOK"""
        for col in columns:
            if self.groupby_column:
                X.loc[:, col] = X.groupby(self.groupby_column)[col].transform(
                    lambda x: x.fillna(
                        self.fill_values_[col].get(x.name, self.global_fill_value_[col])
                    )
                )
            else:
                X.loc[:, col] = X[col].fillna(self.fill_values_[col])
        return X
    
    def _transform_median(self, X: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Median transform - LEAKAGE YOK"""
        for col in columns:
            if self.groupby_column:
                X.loc[:, col] = X.groupby(self.groupby_column)[col].transform(
                    lambda x: x.fillna(
                        self.fill_values_[col].get(x.name, self.global_fill_value_[col])
                    )
                )
            else:
                X.loc[:, col] = X[col].fillna(self.fill_values_[col])
        return X
    
    def _transform_mode(self, X: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Mode transform - LEAKAGE YOK"""
        for col in columns:
            if self.groupby_column:
                X.loc[:, col] = X.groupby(self.groupby_column)[col].transform(
                    lambda x: x.fillna(
                        self.fill_values_[col].get(x.name, self.global_fill_value_[col])
                    )
                )
            else:
                X.loc[:, col] = X[col].fillna(self.fill_values_[col])
        return X
    
    def _transform_constant(self, X: pd.DataFrame, columns: list) -> pd.DataFrame:
        """Constant transform"""
        fill_val = self.fill_values_["constant"]
        X.loc[:, self.fitted_columns_] = X[self.fitted_columns_].fillna(fill_val)
        return X
    
    def _transform_knn(self, X: pd.DataFrame, columns: list) -> pd.DataFrame:
        """KNN transform - Column order & missing check"""
        if not self._knn_columns or not self.scaler_ or not self.imputer_:
            return X
        
        numeric_cols = self._knn_columns
        
        missing_cols = set(numeric_cols) - set(X.columns)
        if missing_cols:
            raise ValueError(
                f"❌ KNN transform'da sütunlar eksik: {missing_cols}\n"
                f"   Fit edilen sütunlar: {numeric_cols}\n"
                f"   Mevcut sütunlar: {list(X.columns)}"
            )
        
        X_subset = X[numeric_cols].copy()
        X_scaled = X_subset.copy()
        X_scaled.loc[:, numeric_cols] = self.scaler_.transform(X_subset)
        X_scaled.loc[:, numeric_cols] = self.imputer_.transform(X_scaled[numeric_cols])
        X_transformed = self.scaler_.inverse_transform(X_scaled[numeric_cols])
        
        X.loc[:, numeric_cols] = X_transformed
        return X

# ===========================================================================
# handle_missing_values — Geriye Uyumlu Wrapper
# ===========================================================================

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[List[str]] = None,
    fill_value: Optional[Any] = None,
    groupby_column: Optional[str] = None,
    numeric_only: bool = True,
    knn_neighbors: int = 5,
    logging_enabled: bool = False,
) -> tuple:
    """
    DataPreprocessor için geriye uyumlu wrapper.

    DataPreprocessor'ın fit/transform çiftini tek seferde uygular ve
    işlem raporuyla birlikte temizlenmiş DataFrame döndürür.

    Test dosyaları ve hızlı prototipleme için kullanılır.
    Production pipeline'da data leakage'ı önlemek için
    DataPreprocessor'ı doğrudan fit/transform ile kullanın.

    Parameters
    ----------
    df             : İşlenecek DataFrame (kopyalanır, orijinal bozulmaz)
    strategy       : Eksik değer stratejisi — DataPreprocessor ile aynı seçenekler:
                     'mean' | 'median' | 'mode' | 'forward_fill' | 'backward_fill'
                     | 'constant' | 'knn' | 'drop'
    columns        : İşlenecek sütunlar; None → otomatik seçim
    fill_value     : strategy='constant' için zorunlu doldurma değeri
    groupby_column : Grup bazlı doldurma için grup sütunu (örn. 'vehicle_id')
    numeric_only   : True → sadece sayısal sütunları işle (mean/median/knn için)
    knn_neighbors  : KNN stratejisinde komşu sayısı
    logging_enabled: True → detaylı log çıktısı

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (temizlenmiş_df, rapor)

        rapor anahtarları:
          strategy            : Kullanılan strateji adı
          columns_processed   : İşlenen sütun listesi
          total_missing_before: İşlem öncesi toplam eksik değer sayısı
          total_missing_after : İşlem sonrası toplam eksik değer sayısı
          missing_per_column  : {col: (önce, sonra)} — sütun bazında özet

    Raises
    ------
    ValueError
        Boş DataFrame, geçersiz strateji, eksik fill_value veya bilinmeyen sütun.

    Examples
    --------
    >>> clean_df, report = handle_missing_values(
    ...     df, strategy='mean', columns=['fuel', 'distance']
    ... )
    >>> print(report['total_missing_before'])  # 3
    >>> print(report['total_missing_after'])   # 0

    >>> # GroupBy ile grup ortalaması (data leakage'sız)
    >>> clean_df, report = handle_missing_values(
    ...     df, strategy='mean', columns=['fuel'],
    ...     groupby_column='vehicle_id'
    ... )
    """
    if df is None or df.empty:
        raise ValueError("❌ Boş DataFrame!")

    # --- İşlenecek sütunları önceden belirle (rapor için) ---
    if columns is not None:
        report_columns = list(columns)
    elif numeric_only and strategy in ["mean", "median", "knn"]:
        report_columns = list(df.select_dtypes(include=[np.number]).columns)
    else:
        report_columns = list(df.columns)

    # --- İşlem öncesi eksik değer sayılarını kaydet ---
    missing_before: Dict[str, int] = {
        col: int(df[col].isnull().sum())
        for col in report_columns
        if col in df.columns
    }
    total_missing_before = sum(missing_before.values())

    # --- DataPreprocessor: fit + transform (wrapper'da train/test ayrımı yok) ---
    preprocessor = DataPreprocessor(
        strategy=strategy,
        columns=columns,
        fill_value=fill_value,
        groupby_column=groupby_column,
        numeric_only=numeric_only,
        knn_neighbors=knn_neighbors,
        logging_enabled=logging_enabled,
    )
    clean_df = preprocessor.fit_transform(df)

    # --- İşlem sonrası eksik değerleri say ---
    processed_cols: List[str] = preprocessor.fitted_columns_ or report_columns
    missing_after: Dict[str, int] = {
        col: int(clean_df[col].isnull().sum())
        for col in processed_cols
        if col in clean_df.columns
    }
    total_missing_after = sum(missing_after.values())

    # --- Sütun bazında özet {col: (önce, sonra)} ---
    missing_per_column: Dict[str, tuple] = {
        col: (missing_before.get(col, 0), missing_after.get(col, 0))
        for col in processed_cols
        if col in df.columns
    }

    report: Dict[str, Any] = {
        "strategy":             strategy,
        "columns_processed":    list(processed_cols),
        "total_missing_before": total_missing_before,
        "total_missing_after":  total_missing_after,
        "missing_per_column":   missing_per_column,
    }

    if logging_enabled:
        logger.info(
            f"📋 handle_missing_values raporu:\n"
            f"   Strateji      : {strategy}\n"
            f"   Eksik (önce)  : {total_missing_before}\n"
            f"   Eksik (sonra) : {total_missing_after}\n"
            f"   Sütunlar      : {list(processed_cols)}"
        )

    return clean_df, report