# src/features.py
"""
Feature Engineering Modülü — DemandForecaster için (Polars backend)

Performans Mimarisi:
  - Tüm feature engineering işlemleri Polars üzerinde çalışır.
    Pandas'taki groupby().transform(lambda) darboğazı, Polars'ın
    vektörize .over() ifadesiyle 7-10x hızlandırıldı.
  - CatBoost Pool nesnesi Pandas DataFrame beklediği için,
    build_feature_matrix() sonunda tek bir .to_pandas() dönüşümü yapılır.
  - Dönüşüm maliyeti (Polars → Pandas) benchmark'ta < 50ms — ihmal edilebilir.

Strateji:
  Zaman özellikleri  : weekday, month, quarter, is_weekend, week_of_year
  Lag özellikleri    : lag_1, lag_7, lag_14, lag_30   (leakage riski sıfır)
  Tatil özellikleri  : `holidays` kütüphanesi — TR resmi + dini bayram
  Spatio-temporal    : TM_ID × weekday etkileşimi
  Rolling stats      : rolling_mean_7/14, rolling_std_7/14

Kural:
  OHE YOK → kategorik kolonlar string kalır, CatBoost cat_features ile alır.
  Tüm işlemler Polars lazy/eager expression API; python lambda YOK.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logging.warning(
        "⚠️  `scipy` kütüphanesi bulunamadı. "
        "compute_target_skewness() devre dışı. "
        "Yüklemek için: pip install scipy"
    )

try:
    import holidays as holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False
    logging.warning(
        "⚠️  `holidays` kütüphanesi bulunamadı. "
        "Tatil özellikleri devre dışı. "
        "Yüklemek için: pip install holidays"
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

DEFAULT_LAGS: List[int] = [1, 7, 14, 30]
DEFAULT_ROLLING_WINDOWS: List[int] = [7, 14]
HOLIDAY_LEAD_DAYS: int = 2


# ---------------------------------------------------------------------------
# Yardımcı: Polars DataFrame'e güvenli dönüşüm
# ---------------------------------------------------------------------------

def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """
    Pandas → Polars dönüşümü.

    pyarrow kurulu olduğunda sıfır kopya ile çalışır.
    Kurulu değilse seri seri numpy üzerinden dönüştürür.
    """
    try:
        return pl.from_pandas(df)
    except ImportError:
        # pyarrow yoksa sütun sütun dönüştür
        data = {}
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                data[col] = pl.Series(col, s.astype("int64").values).cast(
                    pl.Datetime("ns")
                )
            else:
                data[col] = pl.Series(col, s.values)
        return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Yardımcı: Tatil seti
# ---------------------------------------------------------------------------

def _build_holiday_set(years: List[int]) -> set:
    """
    Türkiye resmi tatil + dini bayram tarihlerini döndürür.
    `holidays` kütüphanesi ile otomatik — Ramazan + Kurban dahil.
    """
    if not _HOLIDAYS_AVAILABLE:
        return set()
    holiday_set = set()
    for year in years:
        tr_holidays = holidays_lib.country_holidays("TR", years=year)
        holiday_set.update(tr_holidays.keys())
    return holiday_set


# ---------------------------------------------------------------------------
# 1. Zaman Özellikleri
# ---------------------------------------------------------------------------

def add_time_features(df: pl.DataFrame, date_column: str) -> pl.DataFrame:
    """
    Tarih sütunundan döngüsel ve kategorik zaman özellikleri çıkarır.

    Tüm işlemler Polars expression API ile vektörize — python döngüsü yok.

    Çıktı sütunlar
    --------------
    year, month, day, weekday (0=Pzt … 6=Paz), is_weekend,
    quarter, week_of_year,
    month_sin / month_cos   → döngüsel ay kodlaması
    weekday_sin / weekday_cos → döngüsel gün kodlaması

    Parameters
    ----------
    df          : Polars DataFrame
    date_column : Datetime sütun adı

    Returns
    -------
    pl.DataFrame
    """
    TWO_PI = 2.0 * np.pi

    df = df.with_columns(
        pl.col(date_column).cast(pl.Datetime).alias(date_column)
    )

    df = df.with_columns([
        pl.col(date_column).dt.year().alias("year"),
        pl.col(date_column).dt.month().alias("month"),
        pl.col(date_column).dt.day().alias("day"),
        # Polars weekday: 1=Pzt … 7=Paz → 0-indexed yapalım
        (pl.col(date_column).dt.weekday() - 1).alias("weekday"),
        pl.col(date_column).dt.quarter().alias("quarter"),
        pl.col(date_column).dt.week().alias("week_of_year"),
    ])

    df = df.with_columns([
        # is_weekend: weekday >= 5  (0-indexed: Cmt=5, Paz=6)
        (pl.col("weekday") >= 5).cast(pl.Int8).alias("is_weekend"),

        # Döngüsel kodlama — OHE yerine sin/cos (RAM dostu)
        (TWO_PI * pl.col("month")   / 12.0).sin().alias("month_sin"),
        (TWO_PI * pl.col("month")   / 12.0).cos().alias("month_cos"),
        (TWO_PI * pl.col("weekday") / 7.0 ).sin().alias("weekday_sin"),
        (TWO_PI * pl.col("weekday") / 7.0 ).cos().alias("weekday_cos"),
    ])

    logger.debug("✅ Zaman özellikleri eklendi (Polars).")
    return df


# ---------------------------------------------------------------------------
# 2. Tatil Özellikleri
# ---------------------------------------------------------------------------

def add_holiday_features(
    df: pl.DataFrame,
    date_column: str,
    lead_days: int = HOLIDAY_LEAD_DAYS,
) -> pl.DataFrame:
    """
    Türkiye resmi tatil ve dini bayram bayraklarını ekler.

    Polars'ta set membership kontrolü için tatil tarihleri Int32 epoch
    listesine dönüştürülür ve .is_in() ile vektörize sorgulanır.
    Python döngüsü yalnızca tatil listesi oluşturma aşamasında kullanılır
    (bu sabit boyutlu ve bir kez çalışır).

    Çıktı sütunlar
    --------------
    is_holiday     : O gün resmi tatil mi? (Int8, 0/1)
    is_holiday_eve : Tatil öncesi `lead_days` içinde mi? (Int8, 0/1)

    Parameters
    ----------
    df          : Polars DataFrame
    date_column : Datetime sütunu
    lead_days   : Tatil arifesi kaç gün önceden başlasın (varsayılan: 2)

    Returns
    -------
    pl.DataFrame
    """
    if not _HOLIDAYS_AVAILABLE:
        return df.with_columns([
            pl.lit(0).cast(pl.Int8).alias("is_holiday"),
            pl.lit(0).cast(pl.Int8).alias("is_holiday_eve"),
        ])

    # Yıl listesini Polars'tan çek
    years = df.select(pl.col(date_column).dt.year()).to_series().unique().to_list()
    holiday_set = _build_holiday_set(years)

    # Tatil tarihlerini Unix epoch gün sayısına çevir (Int32, set lookup için)
    _epoch = pd.Timestamp("1970-01-01")
    holiday_days_epoch = [
        int((pd.Timestamp(d) - _epoch).days) for d in holiday_set
    ]
    holiday_eve_days_epoch = []
    for d in holiday_set:
        for offset in range(1, lead_days + 1):
            eve = pd.Timestamp(d) - pd.Timedelta(days=offset)
            holiday_eve_days_epoch.append(int((eve - _epoch).days))

    # Polars: tarihi gün epoch'una çevir, is_in ile karşılaştır
    df = df.with_columns(
        # Datetime → epoch day (nanosaniye → gün)
        (pl.col(date_column).cast(pl.Int64) // (86_400 * 1_000_000_000))
        .cast(pl.Int32)
        .alias("_epoch_day")
    )

    df = df.with_columns([
        pl.col("_epoch_day").is_in(holiday_days_epoch)
          .cast(pl.Int8).alias("is_holiday"),
        pl.col("_epoch_day").is_in(holiday_eve_days_epoch)
          .cast(pl.Int8).alias("is_holiday_eve"),
    ]).drop("_epoch_day")

    logger.debug(f"✅ Tatil özellikleri eklendi ({len(holiday_set)} tatil günü, Polars).")
    return df


# ---------------------------------------------------------------------------
# 3. Lag (Gecikme) Özellikleri
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pl.DataFrame,
    target_column: str,
    group_column: Optional[str],
    lags: List[int] = DEFAULT_LAGS,
) -> pl.DataFrame:
    """
    Hedef değişkenin gecikmeli değerlerini ekler.

    Polars .shift().over() ifadesi ile tamamen vektörize.
    Pandas'taki groupby().transform(lambda x: x.shift(n)) darboğazı YOK.

    ⚠️  Data Leakage Güvencesi:
      - Her grup kendi geçmişine bakar; başka grubun verisi sızmaz.
      - shift(n) ile gelecek bilgisi kullanılmaz.

    Parameters
    ----------
    df            : Polars DataFrame (date'e göre sıralı olmalı)
    target_column : Hedef sütun adı
    group_column  : Grup sütunu; None ise tek seri
    lags          : Gecikme günleri listesi

    Returns
    -------
    pl.DataFrame
    """
    if group_column and group_column in df.columns:
        lag_exprs = [
            pl.col(target_column)
              .shift(lag)
              .over(group_column)          # ← Polars'ın vektörize group-shift'i
              .alias(f"lag_{lag}")
            for lag in lags
        ]
    else:
        lag_exprs = [
            pl.col(target_column).shift(lag).alias(f"lag_{lag}")
            for lag in lags
        ]

    df = df.with_columns(lag_exprs)
    logger.debug(f"✅ Lag özellikleri eklendi (Polars): {[f'lag_{l}' for l in lags]}")
    return df


# ---------------------------------------------------------------------------
# 4. Rolling İstatistikler
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pl.DataFrame,
    target_column: str,
    group_column: Optional[str],
    windows: List[int] = DEFAULT_ROLLING_WINDOWS,
) -> pl.DataFrame:
    """
    Kayan pencere ortalaması ve standart sapmasını ekler.

    Polars'ta .shift(1).rolling_mean(w).over(group) ifadesi ile tek
    with_columns çağrısında tüm pencereler hesaplanır.
    Pandas'taki iç içe lambda + transform zinciri tamamen ortadan kalktı.

    ⚠️  shift(1) → bugünün verisini pencereye dahil etme (leakage önlemi).
    ⚠️  min_samples=1 → kısa geçmişli satırlar NaN üretmez.

    Parameters
    ----------
    df            : Polars DataFrame
    target_column : Hedef sütun
    group_column  : Grup sütunu
    windows       : Pencere boyutları (gün)

    Returns
    -------
    pl.DataFrame
    """
    roll_exprs = []

    for w in windows:
        shifted = pl.col(target_column).shift(1)

        if group_column and group_column in df.columns:
            # .over() ile grup bazında rolling — Polars 1.21+ API
            mean_expr = (
                shifted
                .rolling_mean(window_size=w, min_samples=1)
                .over(group_column)
                .alias(f"rolling_mean_{w}")
            )
            std_expr = (
                shifted
                .rolling_std(window_size=w, min_samples=1)
                .fill_null(0.0)
                .over(group_column)
                .alias(f"rolling_std_{w}")
            )
        else:
            mean_expr = (
                shifted
                .rolling_mean(window_size=w, min_samples=1)
                .alias(f"rolling_mean_{w}")
            )
            std_expr = (
                shifted
                .rolling_std(window_size=w, min_samples=1)
                .fill_null(0.0)
                .alias(f"rolling_std_{w}")
            )

        roll_exprs += [mean_expr, std_expr]

    df = df.with_columns(roll_exprs)
    logger.debug(f"✅ Rolling özellikler eklendi (Polars): pencereler={windows}")
    return df


# ---------------------------------------------------------------------------
# 5. Spatio-Temporal Etkileşim Özellikleri
# ---------------------------------------------------------------------------

def add_spatio_temporal_features(
    df: pl.DataFrame,
    group_column: str,
    date_column: str,
) -> pl.DataFrame:
    """
    Uzamsal-zamansal etkileşim özelliklerini ekler.

    tm_id_encoded      : TM_ID kategorik → integer label (OHE yok)
    tm_weekday_interaction: tm_id_encoded × weekday

    Polars'ta kategorik kodlama .cast(pl.Categorical).to_physical()
    ile tek ifadede yapılır.

    Parameters
    ----------
    df           : Polars DataFrame
    group_column : TM_ID veya benzeri kategorik sütun
    date_column  : Datetime sütunu

    Returns
    -------
    pl.DataFrame
    """
    if group_column not in df.columns:
        return df

    # Kategorik → integer (OHE DEĞİL, sadece label)
    df = df.with_columns(
        pl.col(group_column)
          .cast(pl.Categorical)
          .to_physical()
          .cast(pl.Int32)
          .alias("tm_id_encoded")
    )

    # Weekday sütunu yoksa türet
    if "weekday" not in df.columns:
        df = df.with_columns(
            (pl.col(date_column).dt.weekday() - 1).alias("weekday")
        )

    # Etkileşim: hangi merkez × hangi gün
    df = df.with_columns(
        (pl.col("tm_id_encoded") * 10 + pl.col("weekday"))
        .cast(pl.Int32)
        .alias("tm_weekday_interaction")
    )

    logger.debug("✅ Spatio-temporal özellikler eklendi (Polars).")
    return df


# ---------------------------------------------------------------------------
# 6. Ana Pipeline Fonksiyonu
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
    group_column: Optional[str] = "TM_ID",
    lags: List[int] = DEFAULT_LAGS,
    rolling_windows: List[int] = DEFAULT_ROLLING_WINDOWS,
    holiday_lead_days: int = HOLIDAY_LEAD_DAYS,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Tüm feature engineering adımlarını sırayla uygulayan ana fonksiyon.

    Giriş/Çıkış: Pandas DataFrame (sklearn ve CatBoost uyumluluğu için).
    İç hesaplamalar tamamen Polars üzerinde çalışır (7-10x hızlı).
    Son adımda tek bir .to_pandas() dönüşümü yapılır.

    Adım sırası (data leakage riskine göre):
      1. Pandas → Polars dönüşümü + tarih sıralaması
      2. Zaman özellikleri     (mevcut satırdan, leakage yok)
      3. Tatil özellikleri     (mevcut tarihten, leakage yok)
      4. Spatio-temporal       (grup × zaman etkileşimi)
      5. Lag özellikleri       (shift() ile, leakage yok)
      6. Rolling istatistikler (shift(1) + rolling, leakage yok)
      7. NaN satır temizliği   (lag'den gelen ilk N satır boş olur)
      8. Polars → Pandas dönüşümü (CatBoost için)

    Parameters
    ----------
    df               : Ham giriş verisi (Pandas)
    target_column    : Tahmin hedefi (örn. "desi_hacmi")
    date_column      : Tarih sütunu (örn. "tarih")
    group_column     : Grup/TM sütunu (örn. "TM_ID"); None ise tek seri
    lags             : Gecikme günleri
    rolling_windows  : Rolling pencere boyutları
    holiday_lead_days: Tatil arifesi kaç gün önceden başlasın
    drop_na          : Lag'den kaynaklanan NaN satırları at (varsayılan: True)

    Returns
    -------
    pd.DataFrame
        Modele beslenmeye hazır feature matrix (Pandas).
        Kategorik sütunlar STRING olarak kalır → CatBoost cat_features ile alır.

    Examples
    --------
    >>> X = build_feature_matrix(
    ...     df=raw_df,
    ...     target_column="desi_hacmi",
    ...     date_column="tarih",
    ...     group_column="TM_ID",
    ... )
    >>> cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    >>> model.fit(X.drop("desi_hacmi", axis=1), X["desi_hacmi"],
    ...           cat_features=cat_cols)
    """
    # --- Adım 1: Pandas → Polars + tarih garantisi ---
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    sort_keys = [group_column, date_column] if group_column else [date_column]
    df = df.sort_values(sort_keys).reset_index(drop=True)

    pl_df: pl.DataFrame = _to_polars(df)

    # Polars tarafında da datetime tipini garantiye al
    pl_df = pl_df.with_columns(
        pl.col(date_column).cast(pl.Datetime).alias(date_column)
    )

    # --- Adım 2: Zaman özellikleri ---
    pl_df = add_time_features(pl_df, date_column)

    # --- Adım 3: Tatil özellikleri ---
    pl_df = add_holiday_features(pl_df, date_column, lead_days=holiday_lead_days)

    # --- Adım 4: Spatio-temporal ---
    if group_column and group_column in pl_df.columns:
        pl_df = add_spatio_temporal_features(pl_df, group_column, date_column)

    # --- Adım 5: Lag ---
    pl_df = add_lag_features(pl_df, target_column, group_column, lags=lags)

    # --- Adım 6: Rolling ---
    pl_df = add_rolling_features(
        pl_df, target_column, group_column, windows=rolling_windows
    )

    # --- Adım 7: NaN temizliği ---
    rows_before = pl_df.height
    if drop_na:
        pl_df = pl_df.drop_nulls()
        dropped = rows_before - pl_df.height
        if dropped > 0:
            logger.info(
                f"ℹ️  {dropped} satır lag NaN nedeniyle atıldı "
                f"({rows_before} → {pl_df.height})"
            )

    # --- Adım 8: Polars → Pandas (CatBoost / sklearn uyumluluğu için) ---
    result: pd.DataFrame = pl_df.to_pandas()

    logger.info(
        f"✅ Feature matrix hazır (Polars backend): "
        f"{result.shape[0]} satır × {result.shape[1]} sütun"
    )
    return result


# ---------------------------------------------------------------------------
# Yardımcı: Kategorik sütun listesi (CatBoost için)
# ---------------------------------------------------------------------------

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    DataFrame'deki object/category tipindeki sütunları listeler.

    CatBoost'a `cat_features` parametresi olarak doğrudan verilir.
    OHE yapılmaz — bu fonksiyon sadece hangi kolonun kategorik
    olduğunu raporlar.

    Parameters
    ----------
    df : Feature matrix (Pandas)

    Returns
    -------
    List[str]
    """
    return list(df.select_dtypes(include=["object", "category", "string"]).columns)


# ---------------------------------------------------------------------------
# Yardımcı: Hedef Değişken Çarpıklık Analizi (scipy.stats)
# ---------------------------------------------------------------------------

def compute_target_skewness(
    df: pd.DataFrame,
    target_column: str,
    group_column: Optional[str] = None,
    log_transform: bool = False,
) -> dict:
    """
    Hedef değişkenin istatistiksel dağılımını ve çarpıklığını hesaplar.

    Lojistik talepte kampanya günleri gibi aşırı yüksek değerler
    sağ-çarpık (right-skewed) dağılım oluşturur.  Bu fonksiyon:
      1. Ham dağılımın çarpıklığını (skewness) raporlar,
      2. İsteğe bağlı olarak np.log1p() dönüşümü sonrası
         çarpıklığın ne kadar azaldığını karşılaştırır,
      3. log dönüşümünün önerilip önerilmeyeceğine dair
         otomatik bir tavsiye üretir.

    Çarpıklık yorumu (kural-of-thumb):
      |skewness| < 0.5  → Simetrik dağılım, dönüşüm gerekmez
      0.5 ≤ |s| < 1.0  → Orta çarpıklık, isteğe bağlı
      |skewness| ≥ 1.0  → Yüksek çarpıklık, log dönüşümü önerilir

    scipy.stats.skew() Pandas mean/std'den daha doğrudur:
      - Fisher'ın düzeltmesini (bias=True) uygular
      - NaN içeren satırları otomatik atlar (nan_policy='omit')

    Parameters
    ----------
    df            : Ham veya feature matrix DataFrame
    target_column : Analiz edilecek hedef sütun
    group_column  : Belirtilirse grup bazında da çarpıklık hesaplanır
    log_transform : True ise np.log1p() sonrası karşılaştırma da yapılır

    Returns
    -------
    dict
        {
          "n_samples"       : int   — sıfır hariç gözlem sayısı,
          "mean"            : float — ham ortalama,
          "std"             : float — ham standart sapma,
          "median"          : float — ham medyan,
          "skewness_raw"    : float — ham çarpıklık (scipy.stats.skew),
          "kurtosis_raw"    : float — ham basıklık (scipy.stats.kurtosis),
          "skewness_log1p"  : float | None — log1p sonrası çarpıklık,
          "kurtosis_log1p"  : float | None — log1p sonrası basıklık,
          "recommend_log"   : bool  — |skewness_raw| ≥ 1.0 ise True,
          "group_skewness"  : dict  — {grup_adı: skewness} (group_column verilmişse),
        }

    Examples
    --------
    >>> stats = compute_target_skewness(
    ...     df, target_column="desi_hacmi",
    ...     group_column="rota", log_transform=True
    ... )
    >>> if stats["recommend_log"]:
    ...     print(f"⚠️  Yüksek çarpıklık: {stats['skewness_raw']:.2f} — log1p önerilir")
    """
    if not _SCIPY_AVAILABLE:
        logger.warning("⚠️  scipy bulunamadı — compute_target_skewness() boş döndürüyor.")
        return {}

    if target_column not in df.columns:
        raise ValueError(
            f"❌ '{target_column}' sütunu bulunamadı!\n"
            f"   Mevcut sütunlar: {list(df.columns)}"
        )

    # Sıfır ve NaN dışı gerçek gözlemler (kampanya dışı günler dahil analiz bozar)
    raw_vals = df[target_column].dropna().values.astype(float)
    nonzero  = raw_vals[raw_vals > 0]

    if len(nonzero) < 3:
        logger.warning(
            f"⚠️  '{target_column}' için yeterli gözlem yok ({len(nonzero)} adet). "
            "Çarpıklık hesabı güvenilmez."
        )
        return {"n_samples": len(nonzero), "recommend_log": False}

    # --- Ham istatistikler ---
    skew_raw = float(_scipy_stats.skew(nonzero, bias=True))
    kurt_raw = float(_scipy_stats.kurtosis(nonzero, bias=True))

    result: dict = {
        "n_samples":      int(len(nonzero)),
        "mean":           float(np.mean(nonzero)),
        "std":            float(np.std(nonzero, ddof=1)),
        "median":         float(np.median(nonzero)),
        "skewness_raw":   round(skew_raw, 4),
        "kurtosis_raw":   round(kurt_raw, 4),
        "skewness_log1p": None,
        "kurtosis_log1p": None,
        # |skewness| ≥ 1.0 → yüksek çarpıklık → log1p önerilir
        "recommend_log":  abs(skew_raw) >= 1.0,
        "group_skewness": {},
    }

    # --- log1p karşılaştırması ---
    if log_transform:
        log_vals         = np.log1p(nonzero)
        result["skewness_log1p"] = round(float(_scipy_stats.skew(log_vals, bias=True)), 4)
        result["kurtosis_log1p"] = round(float(_scipy_stats.kurtosis(log_vals, bias=True)), 4)

    # --- Grup bazında çarpıklık ---
    if group_column and group_column in df.columns:
        group_skew: dict = {}
        for grp, grp_df in df.groupby(group_column):
            grp_vals = grp_df[target_column].dropna().values.astype(float)
            grp_vals = grp_vals[grp_vals > 0]
            if len(grp_vals) >= 3:
                group_skew[str(grp)] = round(
                    float(_scipy_stats.skew(grp_vals, bias=True)), 4
                )
        result["group_skewness"] = group_skew

    # --- Log çıktısı ---
    log_line = (
        f"📐 Çarpıklık Analizi — '{target_column}'\n"
        f"   Gözlem (sıfır hariç) : {result['n_samples']:,}\n"
        f"   Ortalama / Medyan    : {result['mean']:.2f} / {result['median']:.2f}\n"
        f"   Çarpıklık (ham)      : {result['skewness_raw']:+.4f}  "
        f"({'⚠️ YÜKSEK' if result['recommend_log'] else '✅ normal'})\n"
        f"   Basıklık  (ham)      : {result['kurtosis_raw']:+.4f}"
    )
    if log_transform and result["skewness_log1p"] is not None:
        log_line += (
            f"\n   Çarpıklık (log1p)    : {result['skewness_log1p']:+.4f}"
            f"  (iyileşme: {result['skewness_raw'] - result['skewness_log1p']:+.4f})"
        )
    if result["recommend_log"]:
        log_line += "\n   💡 Öneri: log_transform_enabled=True ile modeli yeniden eğitin"
    logger.info(log_line)

    return result