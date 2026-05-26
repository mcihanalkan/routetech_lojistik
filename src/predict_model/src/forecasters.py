"""
DemandForecaster — Predict-then-Optimize Talep Tahmin Motoru

Mimari Kararlar (Teknofest kısıtlarına göre):
  ┌─────────────────────────────────────────────────────────────────┐
  │  ✅ CatBoostRegressor   → LightGBM/XGBoost YOK                  │
  │  ✅ cat_features        → One-Hot Encoding YOK (RAM koruması)    │
  │  ✅ MultiQuantile       → TEK MODEL ile q10/q50/q90 bantları    │
  │  ✅ Log1p Dönüşümü      → Uç değerlerde q90 paniğini önler      │
  │  ✅ In-memory JSON      → Disk I/O YOK (10 dk bütçesi korunur)  │
  └─────────────────────────────────────────────────────────────────┘

Quantile Anlamları (ALNS motoruna):
  q10 → Düşük senaryo  : "En kötümser, ama gerçekçi alt sınır"
  q50 → Medyan         : "En olası talep tahmini"
  q90 → Yüksek senaryo : "Spot araç alarmı — bu aşılırsa kira patlar"

Asimetrik Kayıp Mantığı:
  Lojistikte eksik tahmin → spot araç → ~3-9x maliyet artışı.
  Bu yüzden MultiQuantile kayıp fonksiyonunda q90 için alpha=0.9
  kullanılarak underestimation'a (eksik tahmine) 9 kat daha ağır ceza uygulanır.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from copy import deepcopy

from catboost import CatBoostRegressor, Pool
from .base import BaseForecaster
from .features import build_feature_matrix, get_categorical_columns, compute_target_skewness
from .missing import DataPreprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hiperparametre Yükleme — hyperparams_map.json
# ---------------------------------------------------------------------------

def _load_hyperparams(
    data_size: int,
    logging_enabled: bool = True,
) -> tuple:
    """
    hyperparams_map.json'dan veri boyutuna en yakın parametreleri yükler.

    JSON'daki her entry'nin 'row_count' alanına göre en yakın büyük bucket
    seçilir. Hiçbiri uymuyorsa en büyük bucket kullanılır.

    optimize.py ile yeni bucket'lar eklendiğinde bu fonksiyon
    otomatik olarak onları da kullanır — kod değişikliği gerekmez.
    """
    import json
    from pathlib import Path

    # forecasters src/ içinde, JSON proje kökünde
    map_path = Path(__file__).parent.parent / "hyperparams_map.json"
    if not map_path.exists():
        # Fallback: makul varsayılanlar
        if data_size < 50_000:
            p = {"iterations": 1000, "depth": 4, "learning_rate": 0.0476}
            label = "FALLBACK-SMALL"
        else:
            p = {"iterations": 900,  "depth": 6, "learning_rate": 0.0146}
            label = "FALLBACK-LARGE"
    else:
        with open(map_path) as f:
            hmap = json.load(f)

        # row_count'a göre sırala, data_size'a en uygun bucket'ı seç
        entries = sorted(hmap.values(), key=lambda e: e["row_count"])
        selected = entries[0]
        for entry in entries:
            if data_size >= entry["row_count"]:
                selected = entry
        p     = selected["params"]
        label = f"JSON ({selected['row_count']:,} satır bucket, WAPE={selected.get('best_wape', '?')})"

    iterations    = int(p["iterations"])
    depth         = int(p["depth"])
    learning_rate = float(p["learning_rate"])

    if logging_enabled:
        logger.info(
            f"⚖️  Hiperparametre yüklendi: {label}\n"
            f"   iter={iterations} | depth={depth} | lr={learning_rate:.4f}\n"
            f"   (Veri: {data_size:,} satır)"
        )

    return iterations, depth, learning_rate, label


# ---------------------------------------------------------------------------
# Sabitler — Asimetrik Kantil Kayıp Konfigürasyonu
# ---------------------------------------------------------------------------

# Lojistik kısıt: Eksik tahmin → spot araç → 9x ceza
# Quantile(alpha) kayıp matematiği:
#   underestimate cezası = alpha       × |hata|
#   overestimate  cezası = (1 - alpha) × |hata|
#   alpha=0.9 → oran = 0.9 / 0.1 = 9x asimetri  ✅
#
# ⚠️  NOT: Kodun önceki sürümünde AsymmetricMAE:alpha=9.0 kullanılıyordu.
#   Bu kayıp fonksiyonu CatBoost 1.2.x'te MEVCUT DEĞİL ve hata verir.
#   Quantile:alpha=0.9 matematiksel olarak aynı 9x asimetriyi sağlar.
UNDERESTIMATION_PENALTY: float = 9.0   # ← Sadece Decision Regret hesabında kullanılır

# q90 Quantile alpha'sı: 0.9 → 9x asimetrik kayıp (spot araç alarm seviyesi)
Q90_ALPHA: float = 0.9


# ---------------------------------------------------------------------------
# DemandForecaster
# ---------------------------------------------------------------------------

class DemandForecaster(BaseForecaster):
    """
    Predict-then-Optimize Talep Tahmincisi.

    BaseForecaster'dan miras alır; sklearn API uyumlu (fit/predict/get_params).

    Parameters
    ----------
    target_column : str
        Tahmin edilecek hedef sütun. Varsayılan: "desi_hacmi"
    date_column : str
        Tarih sütunu adı. Varsayılan: "tarih"
    group_column : str, optional
        Transfer Merkezi grubu. Varsayılan: "TM_ID"
    train_test_split : float
        Eğitim/test oranı (walk-forward). Varsayılan: 0.8
    forecast_horizon : int
        Kaç gün ileri tahmin. Varsayılan: 7
    iterations : int
        CatBoost ağaç sayısı. Varsayılan: 1000
        ⚠️  10 dk bütçesi için 500-800 arası önerilir.
    learning_rate : float
        CatBoost öğrenme oranı. Varsayılan: 0.05
    depth : int
        CatBoost ağaç derinliği. Varsayılan: 6
    lags : List[int]
        Feature engineering lag günleri. Varsayılan: [1, 7, 14, 30]
    rolling_windows : List[int]
        Rolling istatistik pencereleri. Varsayılan: [7, 14]
    underestimation_penalty : float
        q90 modelinde eksik tahmin cezası katsayısı. Varsayılan: 9.0
    outlier_clip_multiplier : float
        Target sütunundaki outlier'lar için kırpma eşiği.
        median + outlier_clip_multiplier × IQR üzerindeki değerler kırpılır.
        0.0 → kırpma yok. Varsayılan: 3.0
        IQR tabanlı olduğu için gruba göre hesaplanır — rota bazında adil.
    log_transform_enabled : bool
        True ise fit() sırasında hedef değişkene np.log1p() uygulanır;
        predict() çıktısı otomatik olarak np.expm1() ile geri çevrilir.
        Kampanya günleri gibi aşırı yüksek desi değerlerini evcilleştirir.
        Varsayılan: False.
        İpucu: Önce compute_target_skewness() ile çarpıklığı ölçün;
        |skewness| ≥ 1.0 ise bu bayrağı True yapın.
    logging_enabled : bool
        Detaylı log. Varsayılan: True
    random_state : int, optional
        Tekrarlanabilirlik. Varsayılan: 42

    Examples
    --------
    >>> forecaster = DemandForecaster(iterations=800)
    >>> forecaster.fit(train_df)
    >>> results = forecaster.predict(test_df)
    >>> # results → List[Dict]: ALNS motoruna RAM üzerinden aktarılır
    >>> # [{"tarih": "2026-01-08", "TM_ID": "IST-01", "q10": 120, ...}, ...]
    """

    def __init__(
        self,
        target_column: str = "desi_hacmi",
        date_column: str = "tarih",
        group_column: Optional[str] = "TM_ID",
        train_test_split: float = 0.8,
        forecast_horizon: int = 7,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        underestimation_penalty: float = UNDERESTIMATION_PENALTY,
        outlier_clip_multiplier: float = 3.0,
        log_transform_enabled: bool = True,
        logging_enabled: bool = True,
        random_state: Optional[int] = 42,
    ):
        super().__init__(
            target_column=target_column,
            date_column=date_column,
            group_column=group_column,
            train_test_split=train_test_split,
            forecast_horizon=forecast_horizon,
            logging_enabled=logging_enabled,
            random_state=random_state,
        )
        self.iterations            = iterations
        self.learning_rate         = learning_rate
        self.depth                 = depth
        self.lags                  = lags or [1, 7, 14, 30]
        self.rolling_windows       = rolling_windows or [7, 14]
        self.underestimation_penalty = underestimation_penalty
        self.outlier_clip_multiplier = outlier_clip_multiplier
        self.log_transform_enabled   = log_transform_enabled

        # Runtime'da dolacak
        self.model_: CatBoostRegressor = None
        self.cat_features_: List[str] = []
        self.feature_names_: List[str] = []

        # predict() sırasında lag/rolling değerlerini gerçek tarihsel
        # veriden hesaplayabilmek için fit() sonunda saklanan buffer.
        # Her grup için son max(lags) satır + max(rolling_windows) satır
        # tutulur; fillna(0) yanılgısı bu sayede ortadan kalkar.
        self.context_buffer_: Optional[pd.DataFrame] = None

    # -----------------------------------------------------------------------
    # BaseForecaster abstract method: _build_model
    # -----------------------------------------------------------------------

    def _build_model(self) -> None:
        """
        3 ayrı model yerine TEK bir MultiQuantile modeli başlatır.
        Bu sayede kantillerin birbirini kesmesi (crossing) engellenir
        ve eğitim süresi 3 kat kısalır!
        """
        # alpha listesi: q10, q50 ve q90(Asimetrik 9x ceza)
        loss_fn = f"MultiQuantile:alpha=0.1,0.5,{Q90_ALPHA}"

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function=loss_fn,
            random_seed=self.random_state,
            verbose=False,
            allow_writing_files=False, # Disk I/O Yok
            thread_count=-1,
        )

        if self.logging_enabled:
            logger.info(
                f"🏗️  Model oluşturuldu: TEK MODEL ile MultiQuantile\n"
                f"   Kayıp Fonksiyonu: {loss_fn}"
            )

    # -----------------------------------------------------------------------
    # Veri Temizleme — DataPreprocessor entegrasyonu (leakage-safe)
    # -----------------------------------------------------------------------

    def _fit_clip(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Kırpma eşiklerini YALNIZCA train verisinden öğrenir ve uygular.

        sklearn fit/transform ayrımı:
          _fit_clip(train_df)  → eşikleri öğren + train'e uygula
          _apply_clip(test_df) → aynı eşikleri test/predict'e uygula

        Bu ayrım data leakage'ı önler:
          IQR eşikleri test/predict setinin dağılımını görmez.

        Adımlar
        -------
        1. Negatif desi → 0.0  (fiziksel kural, her zaman)
        2. IQR outlier kırpma  (eğer outlier_clip_multiplier > 0)
           Eşik = Q75 + multiplier × IQR, grup bazlı hesaplanır.
           self._clip_upper_'a kaydedilir → _apply_clip'te kullanılır.
        """
        df = train_df.copy()

        # --- 1. Negatif desi → 0.0 ---
        neg_count = (df[self.target_column] < 0).sum()
        if neg_count > 0:
            df[self.target_column] = df[self.target_column].clip(lower=0.0)
            if self.logging_enabled:
                logger.info(f"   🔧 {neg_count} negatif desi değeri → 0.0 kırpıldı")

        # --- 2. IQR outlier kırpma (eşikleri öğren) ---
        self._clip_upper_: Dict[str, float] = {}

        if self.outlier_clip_multiplier > 0.0:
            if self.group_column and self.group_column in df.columns:
                for grp, grp_df in df.groupby(self.group_column):
                    q25 = grp_df[self.target_column].quantile(0.25)
                    q75 = grp_df[self.target_column].quantile(0.75)
                    self._clip_upper_[grp] = q75 + self.outlier_clip_multiplier * (q75 - q25)
            else:
                q25 = df[self.target_column].quantile(0.25)
                q75 = df[self.target_column].quantile(0.75)
                self._clip_upper_["_global_"] = q75 + self.outlier_clip_multiplier * (q75 - q25)

            df, clipped = self._apply_clip(df)
            if self.logging_enabled and clipped > 0:
                logger.info(
                    f"   🔧 Outlier kırpma (train): {clipped} değer kırpıldı "
                    f"(IQR × {self.outlier_clip_multiplier})"
                )

        return df

    def _apply_clip(self, df: pd.DataFrame) -> tuple:
        """
        _fit_clip()'te öğrenilen eşikleri verilen DataFrame'e uygular.

        fit() → test_df'e, predict() → tahmin verisine çağrılır.
        Eşikler self._clip_upper_'dan okunur.

        Returns
        -------
        (temizlenmiş DataFrame, kırpılan değer sayısı)
        """
        df = df.copy()

        # Negatif → 0 (fit olmayan fiziksel kural)
        df[self.target_column] = df[self.target_column].clip(lower=0.0)

        clipped = 0
        if not hasattr(self, "_clip_upper_") or not self._clip_upper_:
            return df, clipped

        before = df[self.target_column].copy()

        if "_global_" in self._clip_upper_:
            df[self.target_column] = df[self.target_column].clip(
                upper=self._clip_upper_["_global_"]
            )
        elif self.group_column and self.group_column in df.columns:
            for grp, upper in self._clip_upper_.items():
                mask = df[self.group_column] == grp
                df.loc[mask, self.target_column] = (
                    df.loc[mask, self.target_column].clip(upper=upper)
                )

        clipped = int((before != df[self.target_column]).sum())
        return df, clipped

    def _engineer_features(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        Ham veriyi feature matrix'e dönüştürür.

        `build_feature_matrix` fonksiyonunu çağırır:
          - Zaman özellikleri
          - Türkiye tatil takvimi (holidays kütüphanesi)
          - Lag özellikleri (group bazında, leakage yok)
          - Rolling istatistikler
          - Spatio-temporal etkileşim

        Parameters
        ----------
        df       : Ham DataFrame
        drop_na  : Lag'den kaynaklanan NaN satırları at

        Returns
        -------
        pd.DataFrame : Feature matrix (kategorikler STRING olarak kalır)
        """
        return build_feature_matrix(
            df=df,
            target_column=self.target_column,
            date_column=self.date_column,
            group_column=self.group_column,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            drop_na=drop_na,
        )

    def _split_X_y(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Feature matrix'ten X ve y'yi ayırır.

        Modele girmeyen sütunları (date, target) X'ten çıkarır.
        Bu sayede date sütunu tahmine sızmaz (leakage önlemi).
        """
        drop_cols = [self.target_column, self.date_column]
        drop_cols = [c for c in drop_cols if c in df.columns]

        X = df.drop(columns=drop_cols)
        y = df[self.target_column]
        return X, y

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, y=None) -> "DemandForecaster":
        """
        Modeli eğitir.

        Adımlar:
          1. Input validasyonu (base class)
          2. Feature engineering
          3. Train/test split (walk-forward)
          4. Outlier kırpma (IQR, sadece train)
          5. Log1p dönüşümü (log_transform_enabled=True ise)
          6. 3 CatBoost modeli eğitimi (q10, q50, q90)
          7. Test seti üzerinde self-evaluation (WAPE + Decision Regret)

        Parameters
        ----------
        df : pd.DataFrame
            Ham veri. date_column ve target_column içermeli.
        y  : Yok sayılır (sklearn uyumluluğu için imzada var)

        Returns
        -------
        self
        """
        t_start = time.time()

        # --- 0. Dinamik Hiperparametre Seçimi — hyperparams_map.json'dan ---
        data_size = len(df)
        self.iterations, self.depth, self.learning_rate, config_label = \
            _load_hyperparams(data_size, self.logging_enabled)

        # --- 1. Validasyon ---

        self._validate_input(df)

        if self.logging_enabled:
            logger.info(
                f"\n{'='*60}\n"
                f"🚀 DemandForecaster.fit() başlıyor\n"
                f"   Veri: {len(df)} satır | "
                f"Hedef: {self.target_column} | "
                f"Grup: {self.group_column}\n"
                f"{'='*60}"
            )

        # --- 2. Feature Engineering (temizlemeden önce — ham veri üzerinde) ---
        if self.logging_enabled:
            logger.info("⚙️  Feature engineering başlıyor...")

        df_features = self._engineer_features(df, drop_na=True)

        # --- 3. Train/Test Split (walk-forward, zaman sıralı) ---
        train_df, test_df = self._train_test_split(df_features)

        # --- 4. Veri Temizleme — SADECE train üzerinde fit et (leakage önlemi) ---
        # IQR eşikleri yalnızca train_df'ten öğrenilir.
        # Aynı eşikler test_df'e uygulanır — test dağılımı öğrenmeye girmez.
        if self.logging_enabled:
            logger.info("🧹 Veri temizleme başlıyor (train only fit)...")
        train_df = self._fit_clip(train_df)   # eşikleri öğren + uygula
        test_df, _ = self._apply_clip(test_df)  # sadece uygula

        # --- Log1p Dönüşümü (opsiyonel) — kampanya günlerini evcilleştir ---
        # Uygulama sırası: clip → log1p (önce uç değerleri kırp, sonra sıkıştır)
        # self.log_transform_enabled_ fit sonunda predict()'e sinyal verir.
        if self.log_transform_enabled:
            skew_stats = compute_target_skewness(
                df         = train_df,
                target_column = self.target_column,
                group_column  = self.group_column,
                log_transform = True,
            )
            if self.logging_enabled:
                logger.info(
                    f"🔁 Log1p dönüşümü uygulanıyor\n"
                    f"   Ham çarpıklık   : {skew_stats.get('skewness_raw', '?'):+.4f}\n"
                    f"   Log1p çarpıklık : {skew_stats.get('skewness_log1p', '?'):+.4f}\n"
                    f"   predict() çıktısı otomatik expm1() ile geri çevrilecek."
                )
            train_df[self.target_column] = np.log1p(train_df[self.target_column])
            test_df[self.target_column]  = np.log1p(test_df[self.target_column])
        else:
            # log_transform kapalıysa yine de çarpıklık raporla (tavsiye için)
            compute_target_skewness(
                df            = train_df,
                target_column = self.target_column,
                group_column  = self.group_column,
                log_transform = False,
            )

        X_train, y_train = self._split_X_y(train_df)
        X_test,  y_test  = self._split_X_y(test_df)

        # Kategorik sütunları tespit et (OHE YOK — string olarak kalır)
        self.cat_features_ = get_categorical_columns(X_train)
        self.feature_names_ = list(X_train.columns)

        if self.logging_enabled:
            logger.info(
                f"   Train: {len(X_train)} satır | "
                f"Test: {len(X_test)} satır\n"
                f"   Kategorik kolonlar (OHE yapılmadı): {self.cat_features_}\n"
                f"   Toplam feature sayısı: {len(self.feature_names_)}"
            )

        # --- 4. Model Eğitimi ---
        self._build_model()

        t_q = time.time()
        if self.logging_enabled:
            logger.info("⏳ MultiQuantile Modeli eğitiliyor...")

        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=self.cat_features_,
        )

        self.model_.fit(train_pool)

        elapsed = time.time() - t_q
        if self.logging_enabled:
            logger.info(f"   ✅ Eğitim tamamlandı ({elapsed:.1f}s)")

        self.is_fitted_ = True

        # --- 5. Context Buffer — predict() için lag kaynağı ---
        # Eğitim verisinin sonundan max(lags) + max(rolling_windows) satır saklanır.
        # predict() bu satırları tahmin verisinin önüne ekleyerek lag/rolling
        # değerlerini gerçek tarihsel veriden hesaplar; fillna(0) yanılgısı yok.
        self._save_context_buffer(df)

        # --- 6. Self-Evaluation ---
        if len(X_test) > 0:
            # Overfit analizi için X_train ve y_train'i de gönderiyoruz
            self._evaluate_on_test(X_test, y_test, X_train, y_train)

        total_elapsed = time.time() - t_start
        if self.logging_enabled:
            logger.info(
                f"\n{'='*60}\n"
                f"✅ fit() tamamlandı — toplam süre: {total_elapsed:.1f}s\n"
                f"{'='*60}"
            )

        return self

    # -----------------------------------------------------------------------
    # predict → In-memory JSON (ALNS motoru için)
    # -----------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        include_features: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Talep tahminlerini in-memory JSON formatında döndürür.

        ⚠️  DISK I/O YOK — CSV/XLSX kaydedilmez.
        Çıktı doğrudan ALNS motoruna RAM üzerinden aktarılır.

        Çıktı Formatı (List[Dict]):
        ---------------------------
        [
          {
            "tarih":  "2026-01-08",
            "TM_ID":  "IST-01",
            "q10":    142.3,   ← Düşük senaryo (alt güven sınırı)
            "q50":    198.7,   ← Medyan tahmin (en olası)
            "q90":    267.4,   ← Yüksek senaryo (spot araç alarm seviyesi)
            "uncertainty_range": 125.1  ← q90 - q10 (belirsizlik genişliği)
          },
          ...
        ]

        Parameters
        ----------
        df : pd.DataFrame
            Ham tahmin verisi (aynı şema, target_column boş/NaN olabilir)
        include_features : bool
            True ise feature sütunları da çıktıya eklenir (debug için)

        Returns
        -------
        List[Dict[str, Any]]
            ALNS motoruna aktarılmaya hazır in-memory JSON

        Raises
        ------
        ValueError
            Model eğitilmemişse
        """
        if not self.is_fitted_:
            raise ValueError(
                "❌ Model eğitilmedi. Önce fit() çağırın!\n"
                "   Kullanım: forecaster.fit(train_df)"
            )

        # --- Feature Engineering (predict — context buffer ile) ---
        # fillna(0) KULLANILMAZ: sıfır, modeli "talep yok" yönünde yanıltır.
        # Bunun yerine fit() sırasında kaydedilen context_buffer_ tahmin
        # verisinin önüne eklenir; lag/rolling değerleri gerçek tarihsel
        # veriden hesaplanır. Buffer satırları sonunda çıkarılır.
        df_predict = self._prepend_context_buffer(df)
        n_buffer_rows = len(df_predict) - len(df)

        df_features = self._engineer_features(df_predict, drop_na=False)

        # Buffer satırlarını çıkar — sadece asıl tahmin dönemini tut
        if n_buffer_rows > 0:
            df_features = df_features.iloc[n_buffer_rows:].reset_index(drop=True)

        # Kalan küçük NaN'ları (buffer yetersizse) son bilinen değerle doldur
        lag_cols  = [c for c in df_features.columns if c.startswith("lag_")]
        roll_cols = [c for c in df_features.columns if c.startswith("rolling_")]
        if lag_cols or roll_cols:
            # ffill: son bilinen değeri taşı; ardından bfill: serinin başındaki boşlukları kapat
            df_features[lag_cols + roll_cols] = (
                df_features[lag_cols + roll_cols]
                .ffill()
                .bfill()
            )

        # X'i hazırla (target ve date çıkar)
        drop_cols = [
            c for c in [self.target_column, self.date_column]
            if c in df_features.columns
        ]
        X_pred = df_features.drop(columns=drop_cols)

        # Eksik feature sütunlarını sıfırla tamamla (train ile uyumsuzluk güvencesi)
        for col in self.feature_names_:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[self.feature_names_]  # train ile aynı sütun sırası

        # --- 3 Kantil Tahmini (MultiQuantile) ---
        pred_pool = Pool(data=X_pred, cat_features=self.cat_features_)

        # MultiQuantile tek seferde (N, 3) boyutunda matris döner
        multi_preds = self.model_.predict(pred_pool)

        q10_vals = multi_preds[:, 0]
        q50_vals = multi_preds[:, 1]
        q90_vals = multi_preds[:, 2]

        # Negatif tahminleri sıfırla (hacim negatif olamaz)
        q10_vals = np.maximum(q10_vals, 0)
        q50_vals = np.maximum(q50_vals, 0)
        q90_vals = np.maximum(q90_vals, 0)

        # --- Log1p Geri Çevirme (fit() log dönüşümü uyguladıysa) ---
        # Model log-uzayında eğitildi; tahminleri orijinal desi ölçeğine çevir.
        # expm1(x) = e^x - 1  →  log1p'nin tam tersi, sayısal olarak kararlı.
        # Monotonluk: log1p monoton artan olduğundan q10 ≤ q50 ≤ q90 korunur.
        if self.log_transform_enabled:
            q10_vals = np.expm1(q10_vals)
            q50_vals = np.expm1(q50_vals)
            q90_vals = np.expm1(q90_vals)
            # expm1 sonrası da negatif olamaz güvencesi
            q10_vals = np.maximum(q10_vals, 0)
            q50_vals = np.maximum(q50_vals, 0)
            q90_vals = np.maximum(q90_vals, 0)

        # --- In-memory JSON Oluşturma (ALNS formatı) ---
        # ⚠️  CSV/XLSX YOK — direkt List[Dict] return
        results: List[Dict[str, Any]] = []

        date_vals = (
            pd.to_datetime(df_features[self.date_column])
            .dt.strftime("%Y-%m-%d")
            .values
            if self.date_column in df_features.columns
            else ["N/A"] * len(q50_vals)
        )

        group_vals = (
            df_features[self.group_column].values
            if self.group_column and self.group_column in df_features.columns
            else [None] * len(q50_vals)
        )

        for i in range(len(q50_vals)):
            record: Dict[str, Any] = {
                self.date_column:       date_vals[i],
                self.group_column:      str(group_vals[i]) if group_vals[i] else None,
                "q10":                  round(float(q10_vals[i]), 4),
                "q50":                  round(float(q50_vals[i]), 4),
                "q90":                  round(float(q90_vals[i]), 4),
                # Belirsizlik genişliği: ALNS için kapasite tamponu hesabında kullanılır
                "uncertainty_range":    round(float(q90_vals[i] - q10_vals[i]), 4),
            }

            if include_features:
                # Debug modu: feature değerlerini de ekle
                for col in self.feature_names_:
                    record[f"feat_{col}"] = X_pred.iloc[i][col]

            results.append(record)

        if self.logging_enabled:
            logger.info(
                f"✅ predict() tamamlandı: {len(results)} tahmin üretildi "
                f"(format: in-memory JSON, disk I/O yok)"
            )

        return results

    # -----------------------------------------------------------------------
    # Context Buffer — predict() lag güvencesi
    # -----------------------------------------------------------------------

    def _save_context_buffer(self, df: pd.DataFrame) -> None:
        """
        Eğitim verisinin son satırlarını context buffer olarak saklar.

        predict() çağrısında lag ve rolling feature'larının NaN üretmemesi
        için tahmin verisinin önüne eklenen tarihsel bağlam penceresidir.

        Buffer boyutu = max(lags) + max(rolling_windows) satır.
        Grup sütunu varsa her grup için ayrı ayrı son N satır alınır;
        böylece farklı TM_ID'lerin geçmişleri birbirine karışmaz.

        Parameters
        ----------
        df : Ham eğitim DataFrame'i (feature engineering öncesi)
        """
        # Kaç satır geriye bakmalıyız?
        buffer_size = max(self.lags) + max(self.rolling_windows)

        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        if self.group_column and self.group_column in df.columns:
            # Her grup için son buffer_size satırı al, birleştir
            parts = []
            for _, grp in df.groupby(self.group_column):
                parts.append(
                    grp.sort_values(self.date_column).tail(buffer_size)
                )
            self.context_buffer_ = (
                pd.concat(parts, ignore_index=True)
                .sort_values([self.group_column, self.date_column])
                .reset_index(drop=True)
            )
        else:
            self.context_buffer_ = (
                df.sort_values(self.date_column)
                .tail(buffer_size)
                .reset_index(drop=True)
            )

        if self.logging_enabled:
            logger.info(
                f"💾 Context buffer kaydedildi: "
                f"{len(self.context_buffer_)} satır "
                f"(buffer_size={buffer_size} × "
                f"{df[self.group_column].nunique() if self.group_column and self.group_column in df.columns else 1} grup)"
            )

    def _prepend_context_buffer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tahmin verisinin önüne context buffer'ı ekler.

        predict() içinde feature engineering çalışmadan önce çağrılır.
        Buffer satırları feature engineering sonrasında çıkarılır;
        yalnızca lag/rolling hesaplamaları için geçici olarak eklenir.

        Buffer yoksa (model henüz fit edilmemiş veya buffer kaydedilmemiş)
        orijinal DataFrame'i değiştirmeden döndürür.

        Parameters
        ----------
        df : Ham tahmin DataFrame'i

        Returns
        -------
        pd.DataFrame
            Buffer + tahmin verisi birleşimi (tarih sıralamalı)
        """
        if self.context_buffer_ is None or self.context_buffer_.empty:
            if self.logging_enabled:
                logger.warning(
                    "⚠️  Context buffer yok — lag değerleri ffill/bfill ile doldurulacak."
                )
            return df.copy()

        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        # target_column tahmin verisinde NaN/eksik olabilir — buffer'daki
        # gerçek değerleri korumak için iki DataFrame'i birleştiriyoruz.
        # Buffer'da target_column varsa olduğu gibi bırak (lag hesabı için gerekli).
        combined = pd.concat(
            [self.context_buffer_, df],
            ignore_index=True
        )

        if self.group_column and self.group_column in combined.columns:
            combined = combined.sort_values(
                [self.group_column, self.date_column]
            ).reset_index(drop=True)
        else:
            combined = combined.sort_values(self.date_column).reset_index(drop=True)

        return combined

    # -----------------------------------------------------------------------
    # Self-Evaluation (fit sonrası)
    # -----------------------------------------------------------------------

    def _evaluate_on_test(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Test ve Train setleri üzerinde WAPE ve Decision Regret hesaplar,
        raporlama ve sunumlar için aşırı öğrenme (overfit) analizi basar.
        """
        # --- TEST SETİ DEĞERLENDİRMESİ ---
        test_pool = Pool(data=X_test, cat_features=self.cat_features_)
        q50_preds_test = self.model_.predict(test_pool)[:, 1]
        q50_preds_test = np.maximum(q50_preds_test, 0)
        y_true_test = y_test.values

        sum_true_test = np.sum(y_true_test)
        wape_test = (
            float(np.sum(np.abs(y_true_test - q50_preds_test)) / sum_true_test)
            if sum_true_test > 0 else 0.0
        )

        diff_test = y_true_test - q50_preds_test
        regret_test = np.where(
            diff_test > 0,
            diff_test * self.underestimation_penalty,
            np.abs(diff_test) * 1.0,
        )
        decision_regret_test = float(np.mean(regret_test))

        # Geriye uyumluluk için eski anahtarları koruyoruz (optimize.py kırılmasın diye)
        self.eval_results_: Dict[str, float] = {
            "WAPE":            round(wape_test, 6),
            "Decision_Regret": round(decision_regret_test, 4),
            "test_samples":    len(y_true_test),
        }

        # --- TRAIN SETİ DEĞERLENDİRMESİ (OVERFIT KONTROLÜ) ---
        wape_train = 0.0
        decision_regret_train = 0.0
        
        if X_train is not None and y_train is not None:
            train_pool = Pool(data=X_train, cat_features=self.cat_features_)
            q50_preds_train = self.model_.predict(train_pool)[:, 1]
            q50_preds_train = np.maximum(q50_preds_train, 0)
            y_true_train = y_train.values

            sum_true_train = np.sum(y_true_train)
            wape_train = (
                float(np.sum(np.abs(y_true_train - q50_preds_train)) / sum_true_train)
                if sum_true_train > 0 else 0.0
            )

            diff_train = y_true_train - q50_preds_train
            regret_train = np.where(
                diff_train > 0,
                diff_train * self.underestimation_penalty,
                np.abs(diff_train) * 1.0,
            )
            decision_regret_train = float(np.mean(regret_train))

            # Raporlama için yeni anahtarları ekle
            self.eval_results_["Train_WAPE"] = round(wape_train, 6)
            self.eval_results_["Train_Decision_Regret"] = round(decision_regret_train, 4)
            self.eval_results_["train_samples"] = len(y_true_train)

        # --- JÜRİ VE RAPORLAMA İÇİN ŞIK TABLO GÖSTERİMİ ---
        if self.logging_enabled:
            status = "✅ STABİL"
            if X_train is not None and (wape_test - wape_train) > 0.05:
                status = "⚠️ OVERFIT"

            log_table = (
                f"\n📊 MODEL PERFORMANS VE OVERFIT ANALİZİ (q50):\n"
                f"   ┌───────────────────┬──────────────┬──────────────┬──────────────┐\n"
                f"   │ Metrik            │ Train Seti   │ Test Seti    │ Durum        │\n"
                f"   ├───────────────────┼──────────────┼──────────────┼──────────────┤\n"
                f"   │ WAPE              │ {wape_train:<12.4%} │ {wape_test:<12.4%} │ {status:<12} │\n"
                f"   │ Decision Regret   │ {decision_regret_train:<12.2f} │ {decision_regret_test:<12.2f} │ {'-'*12} │\n"
                f"   │ Örnek Sayısı      │ {len(y_train) if y_train is not None else 0:<12,} │ {len(y_true_test):<12,} │ {'-'*12} │\n"
                f"   └───────────────────┴──────────────┴──────────────┴──────────────┘"
            )
            logger.info(log_table)

        return self.eval_results_

    # -----------------------------------------------------------------------
    # Yardımcılar
    # -----------------------------------------------------------------------

    def get_feature_importances(self) -> pd.DataFrame:
        """
        q50 modelinin feature importance değerlerini döndürür.

        Hangi özelliğin tahmini en çok etkilediğini gösterir.
        Feature selection ve debug için kullanılır.

        Returns
        -------
        pd.DataFrame
            feature_name ve importance sütunlarıyla sıralı tablo.
        """
        if not self.is_fitted_:
            raise ValueError("❌ Önce fit() çağırın!")

        importances = self.model_.get_feature_importance()
        return (
            pd.DataFrame({
                "feature_name": self.feature_names_,
                "importance": importances,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Sklearn uyumlu parametre sözlüğü."""
        base_params = super().get_params(deep=deep)
        base_params.update({
            "iterations":              self.iterations,
            "learning_rate":           self.learning_rate,
            "depth":                   self.depth,
            "lags":                    self.lags,
            "rolling_windows":         self.rolling_windows,
            "underestimation_penalty": self.underestimation_penalty,
            "outlier_clip_multiplier": self.outlier_clip_multiplier,
            "log_transform_enabled":   self.log_transform_enabled,
        })
        return base_params

    def summary(self) -> str:
        """İnsan okunabilir model özeti."""
        status = "✅ Eğitildi" if self.is_fitted_ else "⏳ Eğitilmedi"
        lines = [
            "=" * 55,
            "  DemandForecaster — Model Özeti",
            "=" * 55,
            f"  Durum           : {status}",
            f"  Hedef           : {self.target_column}",
            f"  Grup            : {self.group_column}",
            f"  Horizon         : {self.forecast_horizon} gün",
            f"  Iterations      : {self.iterations}",
            f"  Depth           : {self.depth}",
            f"  Lags            : {self.lags}",
            f"  Rolling         : {self.rolling_windows}",
            f"  Asimetrik Ceza  : {self.underestimation_penalty}x (q90)",
            f"  Outlier Clip    : IQR × {self.outlier_clip_multiplier} ({'kapalı' if self.outlier_clip_multiplier == 0 else 'açık'})",
            f"  Log Dönüşümü    : {'✅ log1p → expm1' if self.log_transform_enabled else '⬜ kapalı'}",
            f"  Kantiller       : q10 / q50 / q90",
            f"  Çıktı Formatı   : In-memory JSON (disk I/O yok)",
        ]
        if self.is_fitted_ and hasattr(self, "eval_results_"):
            lines += [
                "-" * 55,
                f"  Train WAPE      : {self.eval_results_.get('Train_WAPE', 0.0):.4%}",
                f"  Test WAPE       : {self.eval_results_.get('WAPE', 'N/A'):.4%}",
                f"  Decision Regret : {self.eval_results_.get('Decision_Regret', 'N/A'):.2f}",
            ]
        lines.append("=" * 55)
        return "\n".join(lines)
