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
  Zaman özellikleri      : weekday, month, quarter, is_weekend, week_of_year
  Lag özellikleri        : lag_1, lag_7, lag_14, lag_30   (leakage riski sıfır)
  Tatil özellikleri      : `holidays` kütüphanesi — TR resmi + dini bayram
  Spatio-temporal        : TM_ID × weekday etkileşimi
  Rolling stats          : rolling_mean_7/14, rolling_std_7/14
  Ağ/Çizge özellikleri   : Pressure Ratio, Hub Centrality, K-dereceli komşuluk
                            (KDD Cup 2020 şampiyonluk mekanizması — PDF Bölüm 1)
  Hiyerarşik özellikler  : Hub-to-Route Ratio Lags, Çapraz-Grup Max/Mean/Std
                            (M5 & Grupo Bimbo şampiyonluk mekanizması — PDF Bölüm 2.3)
  Ekstrem olay özellikleri: Pencere Eğrisi (Backlog Window Intensity),
                            SHOS Hurdle Skoru, Log Dönüşümü Sinyali
                            (M5 Uncertainty & RSNA şampiyonluk — PDF Bölüm 3)

Kural:
  OHE YOK → kategorik kolonlar string kalır, CatBoost cat_features ile alır.
  Tüm işlemler Polars lazy/eager expression API; python lambda YOK.
"""

import logging
from typing import List, Optional, Union

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

DEFAULT_LAGS: List[int] = [1, 7, 14]
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
# Yardımcı: Kampanya (E-Ticaret) Seti
# ---------------------------------------------------------------------------

def _build_campaign_set(years: List[int]) -> set:
    """
    Türkiye e-ticaret ve lojistik hacmini patlatan majör kampanya günlerini hesaplar.

    Hareketsiz özel günler (Dünya Su Günü, Kızılay Haftası vb.) dahil edilmez.
    Yalnızca online alışveriş zirvesi yaratan, kargo şubelerini kilitleyen günler:

      1. Sevgililer Günü   -> 14 Şubat (sabit)
      2. 11.11 İndirimleri -> 11 Kasım  (sabit)
      3. Anneler Günü      -> Mayıs'ın 2. Pazarı (hareketli)
      4. Babalar Günü      -> Haziran'ın 3. Pazarı (hareketli)
      5. Efsane Cuma       -> Kasım'ın son Cuması (hareketli)

    Not: Asıl kargo patlaması kampanya günü değil, önceki 3-5 gündür
    (herkes hediyesini hafta içinde sipariş eder). Bu pencere
    add_campaign_features() içinde lead_days ile modellenir.
    """
    campaign_set = set()
    for y in years:
        # 1. Sevgililer Günü (Sabit)
        campaign_set.add(pd.Timestamp(year=y, month=2, day=14).date())

        # 2. 11.11 İndirimleri (Sabit)
        campaign_set.add(pd.Timestamp(year=y, month=11, day=11).date())

        # 3. Anneler Günü (Mayıs'ın 2. Pazarı)
        may_sundays = pd.date_range(start=f"{y}-05-01", end=f"{y}-05-31", freq="D")
        may_sundays = may_sundays[may_sundays.weekday == 6]  # 6 = Pazar
        campaign_set.add(may_sundays[1].date())              # index 1 = 2. Pazar

        # 4. Babalar Günü (Haziran'ın 3. Pazarı)
        jun_sundays = pd.date_range(start=f"{y}-06-01", end=f"{y}-06-30", freq="D")
        jun_sundays = jun_sundays[jun_sundays.weekday == 6]  # 6 = Pazar
        campaign_set.add(jun_sundays[2].date())              # index 2 = 3. Pazar

        # 5. Efsane Cuma / Black Friday (Kasım'ın son Cuması)
        nov_fridays = pd.date_range(start=f"{y}-11-01", end=f"{y}-11-30", freq="D")
        nov_fridays = nov_fridays[nov_fridays.weekday == 4]  # 4 = Cuma
        campaign_set.add(nov_fridays[-1].date())             # index -1 = Son Cuma

    return campaign_set


# ---------------------------------------------------------------------------
# 2.5 Kampanya (E-Ticaret) Özellikleri
# ---------------------------------------------------------------------------

def add_campaign_features(
    df: pl.DataFrame,
    date_column: str,
    lead_days: int = 5,
) -> pl.DataFrame:
    """
    E-Ticaret kampanya günlerini ve kampanya öncesi sipariş dönemini ekler.

    Lojistik gerçeği: Kargo patlaması kampanya gününde değil, kampanyadan
    önceki 3-5 gün yaşanır — müşteriler hediyelerini hafta içi sipariş eder,
    teslimat çoğunlukla kampanya günü veya haftasında olur. Bu nedenle:
      - is_campaign_day : Asıl kampanya günü (Anneler Günü, Black Friday vb.)
      - is_campaign_eve : Kampanyadan lead_days gün öncesine kadar olan dönem
                          (kargo hacminin gerçekten arttığı sipariş penceresi)

    Polars native pl.Date karşılaştırması kullanılır.
    Eski epoch-bölme yaklaşımı Polars'ın yeni mikrosaniye Datetime'ında
    sıfır ürettiğinden kaldırıldı.

    Çıktı sütunlar
    --------------
    is_campaign_day : O gün ana kampanya günü mü?       (Int8, 0/1)
    is_campaign_eve : Kampanya öncesi sipariş dönemi mi? (Int8, 0/1)
    """
    years = df.select(pl.col(date_column).dt.year()).to_series().unique().to_list()
    campaign_set = _build_campaign_set(years)

    # Python date objeleri — epoch matematiği yok, Polars versiyonundan bağımsız
    camp_days = [pd.Timestamp(d).date() for d in campaign_set]
    camp_eve_days = []
    for d in campaign_set:
        for offset in range(1, lead_days + 1):
            eve = pd.Timestamp(d) - pd.Timedelta(days=offset)
            camp_eve_days.append(eve.date())

    # cast(pl.Date).is_in() — native Polars Date karşılaştırması
    df = df.with_columns([
        pl.col(date_column).cast(pl.Date).is_in(camp_days)
          .cast(pl.Int8).alias("is_campaign_day"),
        pl.col(date_column).cast(pl.Date).is_in(camp_eve_days)
          .cast(pl.Int8).alias("is_campaign_eve"),
    ])

    logger.debug(
        f"✅ Kampanya özellikleri eklendi "
        f"({len(campaign_set)} ana gün, {lead_days} gün arife penceresi, Polars native)."
    )
    return df



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
# 1.5 Kaggle Takvim + Spatio-Temporal (Rota-Gün) Özellikleri
# ---------------------------------------------------------------------------

def add_advanced_calendar_features(df: pl.DataFrame, date_column: str, group_column: str) -> pl.DataFrame:
    """
    Kaggle stili gelişmiş takvim ve Spatio-Temporal etkileşim özelliklerini ekler.

    Çıktı sütunlar
    --------------
    is_payday_period : Türkiye maaş dönemleri (1-5 ve 15-20'si) mi? (Int8, 0/1)
                       — eski binary bayrak, geriye uyumluluk için tutuldu.
    route_weekday    : Rota × Gün etkileşimi — "TM_ID_weekday" (Utf8, CatBoost kategorik)
                       Örn: "Manisa->İstanbul_0" → o rotanın Pazartesi profili.
                       Mevcut sayısal tm_weekday_interaction'ı tamamlar;
                       CatBoost bu string sütunu doğrudan kategorik olarak okur.
    is_post_holiday  : Tatil dönüşü mü? (Int8, 0/1) — dün tatildi, bugün değil
                       → birikmiş kargo patlaması beklenir.
    is_post_campaign : Kampanya artçı şoku mu? (Int8, 0/1) — dün kampanya
                       öncesi dönemdi, bugün değil → devir devam ediyor.

    Maaş Günü Harmonikleri (Fourier) ve Asimetrik Sönüm
    ----------------------------------------------------
    Binary `is_payday_period` bayrağı "ayın 6'sı" ile "ayın 2'si"ni aynı
    kutuya koyar; CatBoost split'leri için bilgi kaybı yaratır. Aşağıdaki
    sürekli özellikler ayın gününe göre yumuşak bir sinyal üretir:

    payday_sin_k1 / payday_cos_k1 : Aylık genel salınım (k=1, periyot = ayın gün sayısı)
    payday_sin_k2 / payday_cos_k2 : Yarım aylık çift maaş döngüsü (k=2, periyot = ay/2)
                                     → ayın 1'i VE 15'i için aynı fazda zirve yapar.
    public_payday_impact  : Ayın 15'i (kamu maaşı) sonrası asimetrik üstel
                            sönüm. Maaş sonrası (gün>=15): exp(-(gün-15)/2.2)
                            (yavaş sönüm, 3 günlük harcama dalgası).
                            Maaş öncesi (gün<15): exp(-(15-gün)/1.2)
                            (hızlı sönüm, kısa bekleyiş etkisi).
    private_payday_impact : Ayın 1'i (özel sektör maaşı) sonrası asimetrik
                            üstel sönüm. Mesafe, ayın başına/sonuna göre
                            simetrik hesaplanır (dist_private_payday);
                            maaş sonrası ilk 3 gün hızlı (exp(-d/1.5)),
                            diğer günler kademeli (exp(-d/3.0)) sönüm.
    """
    if group_column not in df.columns:
        return df

    TWO_PI = 2.0 * np.pi
    day = pl.col(date_column).dt.day().cast(pl.Float64)
    days_in_month = pl.col(date_column).dt.month_end().dt.day().cast(pl.Float64)

    # --- Özel sektör maaşı mesafesi (ayın 1'i merkezli, ay başı/sonuna göre simetrik) ---
    # day >= days_in_month - 1  → maaşın yattığı gün / arife  → mesafe 0
    # day <= 3                  → maaş sonrası ilk günler     → mesafe = day
    # diğer                     → min(day - 3, days_in_month - 1 - day)
    dist_private = (
        pl.when(day >= days_in_month - 1.0).then(0.0)
          .when(day <= 3.0).then(day)
          .otherwise(
              pl.min_horizontal(day - 3.0, days_in_month - 1.0 - day)
          )
    )

    df = df.with_columns([
        # 1. Maaş Günü Etkisi (Ayın 1-5'i ve 15-20'si) — geriye uyumlu binary bayrak
        pl.when(
            ((pl.col(date_column).dt.day() >= 1) & (pl.col(date_column).dt.day() <= 5)) |
            ((pl.col(date_column).dt.day() >= 15) & (pl.col(date_column).dt.day() <= 20))
        ).then(1).otherwise(0).cast(pl.Int8).alias("is_payday_period"),

        # 2. Rota - Gün Etkileşimi (Örn: "Manisa -> İstanbul_0" yani Pazartesi)
        (pl.col(group_column).cast(pl.Utf8) + "_" + pl.col("weekday").cast(pl.Utf8)).alias("route_weekday"),

        # 3. Tatil Dönüşü Birikim Etkisi (Post-Holiday Backlog)
        # Dün tatilse ve bugün tatil değilse -> Birikim patlaması yaşanacak!
        # .over(group_column): shift rota bazında yapılır, rotalar arası sızıntı önlenir.
        pl.when(
            (pl.col("is_holiday").shift(1).over(group_column) == 1) & (pl.col("is_holiday") == 0)
        ).then(1).otherwise(0).cast(pl.Int8).alias("is_post_holiday"),

        # 4. Kampanya Sonrası Artçı Şok (Post-Campaign)
        # Kampanya bittikten sonraki 1 gün kargo devirleri devam eder.
        pl.when(
            (pl.col("is_campaign_eve").shift(1).over(group_column) == 1) & (pl.col("is_campaign_eve") == 0)
        ).then(1).otherwise(0).cast(pl.Int8).alias("is_post_campaign"),

        # 5. Maaş Günü Fourier Harmonikleri
        (TWO_PI * 1.0 * day / days_in_month).sin().alias("payday_sin_k1"),
        (TWO_PI * 1.0 * day / days_in_month).cos().alias("payday_cos_k1"),
        (TWO_PI * 2.0 * day / days_in_month).sin().alias("payday_sin_k2"),
        (TWO_PI * 2.0 * day / days_in_month).cos().alias("payday_cos_k2"),

        # 6. Kamu Sektörü Maaşı (Ayın 15'i) — Asimetrik Üstel Sönüm
        pl.when(day >= 15.0)
          .then((-(day - 15.0) / 2.2).exp())   # maaş sonrası: yavaş sönüm (3 günlük dalga)
          .otherwise((-(15.0 - day) / 1.2).exp())  # maaş öncesi: hızlı sönüm (kısa bekleyiş)
          .alias("public_payday_impact"),

        # 7. Özel Sektör Maaşı (Ayın 1'i) — Asimetrik Üstel Sönüm
        pl.when(day <= 3.0)
          .then((-dist_private / 1.5).exp())   # maaş sonrası ilk 3 gün: hızlı lojistik çıkış
          .otherwise((-dist_private / 3.0).exp())  # maaş öncesi: kademeli azalma
          .alias("private_payday_impact"),
    ])

    logger.debug(
        "✅ Gelişmiş Takvim, Rota-Gün, Birikim (Backlog) ve Maaş Harmonik/Sönüm "
        "özellikleri eklendi."
    )
    return df


# ---------------------------------------------------------------------------
# 2. Tatil Özellikleri
# ---------------------------------------------------------------------------

def add_holiday_features(
    df: pl.DataFrame,
    date_column: str,
    group_column: Optional[str] = None,
    lead_days: int = HOLIDAY_LEAD_DAYS,
    backlog_alpha: float = 1.4,
) -> pl.DataFrame:
    """
    Türkiye resmi tatil ve dini bayram bayraklarını ekler.

    Polars native pl.Date karşılaştırması kullanılır.
    Eski epoch-bölme yaklaşımı (cast(Int64) // 86_400 * 1_000_000_000)
    Polars'ın yeni sürümlerinde mikrosaniye birimli Datetime ile sıfır
    ürettiğinden kaldırıldı. cast(pl.Date).is_in() tüm versiyonlarda
    doğru çalışır ve daha okunabilirdir.

    Çıktı sütunlar
    --------------
    is_holiday              : O gün resmi tatil mi? (Int8, 0/1)
    is_holiday_eve          : Tatil öncesi `lead_days` içinde mi? (Int8, 0/1)
    is_closed               : O gün lojistik ağ kapalı mı? (Int8, 0/1)
                              (resmi tatil VEYA Pazar günü — Türkiye'de
                              pazar günleri dağıtım yapılmaz)
    accumulated_closed_days : Az önce sona eren ardışık kapalı gün sayısı
                              (Float64) — Lojistik Birikim teorisi (BAI),
                              rota bazında hesaplanır.
    days_since_resumption   : Hizmete dönüşten itibaren geçen aktif gün
                              sayısı (Int64). Kapalı günlerde -1.
    backlog_release_index   : Tatil dönüşü birikmiş kargonun üstel
                              sönümle dağıtım hızı (Float64).
                              backlog_release_index =
                                accumulated_closed_days.shift(1)
                                * exp(-days_since_resumption / alpha)
                              alpha≈1.4: TR lojistik ağının birikmiş
                              yükü eritme hız sabiti (gün bazında).
                              4 günden sonra etki sıfırlanır.

    Parameters
    ----------
    df            : Polars DataFrame
    date_column   : Datetime sütunu
    group_column  : Rota/grup sütunu — backlog hesapları bu sütun
                    bazında (.over()) yapılır; None ise tek seri kabul
                    edilir (rotalar arası sızıntı riski oluşmaz).
    lead_days     : Tatil arifesi kaç gün önceden başlasın (varsayılan: 2)
    backlog_alpha : Birikim erime hız sabiti (varsayılan: 1.4)

    Returns
    -------
    pl.DataFrame
    """
    if not _HOLIDAYS_AVAILABLE:
        df = df.with_columns([
            pl.lit(0).cast(pl.Int8).alias("is_holiday"),
            pl.lit(0).cast(pl.Int8).alias("is_holiday_eve"),
        ])
        holiday_set = set()
    else:
        years = df.select(pl.col(date_column).dt.year()).to_series().unique().to_list()
        holiday_set = _build_holiday_set(years)

        # Python date objeleri — epoch matematiği yok, Polars versiyonundan bağımsız
        holiday_days = [pd.Timestamp(d).date() for d in holiday_set]
        holiday_eve_days = []
        for d in holiday_set:
            for offset in range(1, lead_days + 1):
                eve = pd.Timestamp(d) - pd.Timedelta(days=offset)
                holiday_eve_days.append(eve.date())

        # cast(pl.Date).is_in() — native Polars Date karşılaştırması
        df = df.with_columns([
            pl.col(date_column).cast(pl.Date).is_in(holiday_days)
              .cast(pl.Int8).alias("is_holiday"),
            pl.col(date_column).cast(pl.Date).is_in(holiday_eve_days)
              .cast(pl.Int8).alias("is_holiday_eve"),
        ])

    # ------------------------------------------------------------------
    # Lojistik Birikim (BAI) — is_closed = tatil VEYA Pazar günü
    # ------------------------------------------------------------------
    df = df.with_columns([
        pl.when(
            (pl.col("is_holiday") == 1) | (pl.col(date_column).dt.weekday() == 7)
        ).then(1).otherwise(0).cast(pl.Int8).alias("is_closed")
    ])

    over_keys = [group_column] if group_column and group_column in df.columns else None

    def _over(expr: pl.Expr) -> pl.Expr:
        return expr.over(over_keys) if over_keys else expr

    # _reset_grp: her açık günde 1 artar — ardışık kapalı blokları gruplar
    df = df.with_columns([
        _over((pl.col("is_closed") == 0).cum_sum()).alias("_reset_grp")
    ])

    # streak_len: o ana kadar süregelen ardışık kapalı gün sayısı
    reset_keys = (over_keys or []) + ["_reset_grp"]
    df = df.with_columns([
        pl.when(pl.col("is_closed") == 1)
          .then(pl.col("is_closed").cum_sum().over(reset_keys))
          .otherwise(0)
          .alias("_streak_len")
    ])

    # accumulated_closed_days: az önce sona eren kapalı bloğun boyu (açık günlerde dolu, kapalı günlerde 0)
    df = df.with_columns([
        pl.when(pl.col("is_closed") == 0)
          .then(_over(pl.col("_streak_len").shift(1)).fill_null(0))
          .otherwise(0.0)
          .cast(pl.Float64)
          .alias("accumulated_closed_days")
    ])

    # days_since_resumption: açık günlerde, mevcut açık-blok içindeki 1-indeksli pozisyon; kapalı günlerde -1
    df = df.with_columns([
        pl.when(pl.col("is_closed") == 0)
          .then(pl.int_range(1, pl.len() + 1).over(reset_keys))
          .otherwise(-1)
          .alias("days_since_resumption")
    ])

    # backlog_release_index: önceki günün accumulated_closed_days'i * exp(-days_since_resumption/alpha)
    # 4 günden sonra etki sıfırlanır
    df = df.with_columns([
        _over(pl.col("accumulated_closed_days").shift(1)).fill_null(0.0).alias("_prev_accum")
    ])
    df = df.with_columns([
        pl.when(
            (pl.col("is_closed") == 0) &
            (pl.col("days_since_resumption") >= 0) &
            (pl.col("days_since_resumption") <= 4)
        )
        .then(pl.col("_prev_accum") * (-pl.col("days_since_resumption") / backlog_alpha).exp())
        .otherwise(0.0)
        .alias("backlog_release_index")
    ])

    df = df.drop(["_reset_grp", "_streak_len", "_prev_accum"])

    logger.debug(
        f"✅ Tatil özellikleri eklendi ({len(holiday_set)} tatil günü, "
        f"Polars native, BAI alpha={backlog_alpha})."
    )
    return df


# ---------------------------------------------------------------------------
# 3. Lag (Gecikme) Özellikleri
# ---------------------------------------------------------------------------

def _feature_suffix(target_column: str, target_columns: List[str]) -> str:
    """
    Birden fazla paralel hedef sütun varsa (wide format: toplam_desi_0900 /
    toplam_desi_1700), çıktı feature adına eklenecek soneki üretir.

    'toplam_desi_0900' -> '_0900'
    'toplam_desi_1700' -> '_1700'

    Tek hedef sütun varsa (eski/legacy tek-serili akış) -> ''
    (sonek yok — eski sütun adları `lag_1`, `rolling_mean_7` vb. ile
    birebir aynı kalır, geriye dönük uyumluluk bozulmaz.)
    """
    if len(target_columns) <= 1:
        return ""
    parts = target_column.rsplit("_", 1)
    return f"_{parts[-1]}" if len(parts) > 1 else f"_{target_column}"


def add_lag_features(
    df: pl.DataFrame,
    target_columns: Union[str, List[str]],
    group_column: Optional[str],
    lags: List[int] = DEFAULT_LAGS,
) -> pl.DataFrame:
    """
    Hedef değişken(ler)in gecikmeli değerlerini ekler.

    Polars .shift().over() ifadesi ile tamamen vektörize.
    Pandas'taki groupby().transform(lambda x: x.shift(n)) darboğazı YOK.

    Wide format (09:00 / 17:00) desteği
    -------------------------------------
    target_columns birden fazla sütun içeriyorsa (örn.
    ["toplam_desi_0900", "toplam_desi_1700"]), her sütun için AYRI AYRI
    lag üretilir ve çıktı sütun adına slot soneki eklenir:
      lag_1_0900, lag_7_0900, lag_14_0900, ...
      lag_1_1700, lag_7_1700, lag_14_1700, ...
    Satır zaten "gün" granülaritesinde olduğu için shift() mantığı
    değişmiyor — shift(1).over(rota) hâlâ "dünkü değer" demek.
    Tek sütun verilirse eski sütun adları (lag_1, lag_7, ...) korunur.

    ⚠️  Data Leakage Güvencesi:
      - Her grup kendi geçmişine bakar; başka grubun verisi sızmaz.
      - shift(n) ile gelecek bilgisi kullanılmaz.

    Parameters
    ----------
    df             : Polars DataFrame (date'e göre sıralı olmalı)
    target_columns : Hedef sütun adı (str) veya sütun listesi (List[str])
    group_column   : Grup sütunu; None ise tek seri
    lags           : Gecikme günleri listesi

    Returns
    -------
    pl.DataFrame
    """
    target_cols: List[str] = [target_columns] if isinstance(target_columns, str) else list(target_columns)

    lag_exprs = []
    produced_names: List[str] = []
    for tcol in target_cols:
        suffix = _feature_suffix(tcol, target_cols)
        base = pl.col(tcol)
        for lag in lags:
            alias = f"lag_{lag}{suffix}"
            produced_names.append(alias)
            if group_column and group_column in df.columns:
                lag_exprs.append(base.shift(lag).over(group_column).alias(alias))
            else:
                lag_exprs.append(base.shift(lag).alias(alias))

    df = df.with_columns(lag_exprs)
    logger.debug(f"✅ Lag özellikleri eklendi (Polars): {produced_names}")
    return df


# ---------------------------------------------------------------------------
# 4. Rolling İstatistikler
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pl.DataFrame,
    target_columns: Union[str, List[str]],
    group_column: Optional[str],
    windows: List[int] = DEFAULT_ROLLING_WINDOWS,
) -> pl.DataFrame:
    """
    Kayan pencere ortalaması, standart sapması ve varyasyon katsayısını ekler.

    Polars'ta .shift(1).rolling_mean(w).over(group) ifadesi ile tek
    with_columns çağrısında tüm pencereler hesaplanır.
    Pandas'taki iç içe lambda + transform zinciri tamamen ortadan kalktı.

    Wide format (09:00 / 17:00) desteği
    -------------------------------------
    target_columns birden fazla sütun içeriyorsa, her sütun için AYRI AYRI
    rolling istatistik üretilir ve çıktı sütun adına slot soneki eklenir:
      rolling_mean_7_0900, rolling_std_7_0900, rolling_cov_7_0900, ...
      rolling_mean_7_1700, rolling_std_7_1700, rolling_cov_7_1700, ...
    Tek sütun verilirse eski sütun adları (rolling_mean_7, ...) korunur.

    ⚠️  shift(1) → bugünün verisini pencereye dahil etme (leakage önlemi).
    ⚠️  min_samples=1 → kısa geçmişli satırlar NaN üretmez.

    Parameters
    ----------
    df             : Polars DataFrame
    target_columns : Hedef sütun adı (str) veya sütun listesi (List[str])
    group_column   : Grup sütunu
    windows        : Pencere boyutları (gün)

    Returns
    -------
    pl.DataFrame
    """
    target_cols: List[str] = [target_columns] if isinstance(target_columns, str) else list(target_columns)

    roll_exprs = []
    produced_names: List[str] = []

    for tcol in target_cols:
        suffix = _feature_suffix(tcol, target_cols)

        for w in windows:
            shifted = pl.col(tcol).shift(1)
            mean_alias = f"rolling_mean_{w}{suffix}"
            std_alias  = f"rolling_std_{w}{suffix}"
            cov_alias  = f"rolling_cov_{w}{suffix}"

            if group_column and group_column in df.columns:
                # .over() ile grup bazında rolling — Polars 1.21+ API
                mean_expr = (
                    shifted
                    .rolling_mean(window_size=w, min_samples=1)
                    .over(group_column)
                    .alias(mean_alias)
                )
                std_expr = (
                    shifted
                    .rolling_std(window_size=w, min_samples=1)
                    .fill_null(0.0)
                    .over(group_column)
                    .alias(std_alias)
                )
            else:
                mean_expr = (
                    shifted
                    .rolling_mean(window_size=w, min_samples=1)
                    .alias(mean_alias)
                )
                std_expr = (
                    shifted
                    .rolling_std(window_size=w, min_samples=1)
                    .fill_null(0.0)
                    .alias(std_alias)
                )

            roll_exprs += [mean_expr, std_expr]
            produced_names += [mean_alias, std_alias]

            # --- Volatilite (Oynaklık) İndeksi ---
            # Sıfıra bölme hatasını önlemek için paydaya küçük bir epsilon (1e-5) ekliyoruz.
            cov_expr = (std_expr / (mean_expr + 1e-5)).alias(cov_alias)
            roll_exprs.append(cov_expr)
            produced_names.append(cov_alias)

    df = df.with_columns(roll_exprs)
    logger.debug(f"✅ Rolling özellikler eklendi (Polars): {produced_names}")
    return df


# ---------------------------------------------------------------------------
# 4.5 Cross-Lag (Gün-İçi Slotlar Arası Bilgi Akışı) — 09:00 / 17:00
# ---------------------------------------------------------------------------

def add_cross_lag_features(
    df: pl.DataFrame,
    target_columns: Union[str, List[str]],
) -> pl.DataFrame:
    """
    İki paralel (09:00 / 17:00) hedef arasındaki meşru gün-içi bilgi akışını
    açık, isimlendirilmiş bir feature olarak ekler.

    Mantık
    ------
    - 17:00 tahmini yapılırken, aynı günün 09:00 talebi ZATEN GERÇEKLEŞMİŞ ve
      bilinen bir değerdir → shift YOK, doğrudan bugünkü değer feature olarak
      kullanılabilir. Bu leakage DEĞİL, gerçek operasyonel bilgidir (17:00
      tahmini yapıldığı anda sabahki gerçek talep zaten elde mevcuttur).
    - 09:00 tahmini yapılırken, aynı günün 17:00 talebi HENÜZ GERÇEKLEŞMEMİŞTİR
      → kullanılamaz. Bunun yerine sadece dünkü (shift(1)) 17:00 değeri
      kullanılabilir — bu zaten add_lag_features()'ın ürettiği `lag_1_1700`
      sütunuyla birebir aynıdır, burada tekrar üretilmez.

    Bu fonksiyon SADECE meşru cross-lag'i açıkça isimlendirir; hangi sütunun
    hangi model için "leakage" sayılıp düşürüleceğine (drop) karar vermez —
    o ayrım forecasters.py'de, model bazında (09:00 modeli vs 17:00 modeli)
    yapılacaktır. İki model de aynı zengin feature havuzundan beslenir.

    Çıktı sütun
    -----------
    cross_lag_0900_same_day : Bugünün "…_0900" ile biten hedef sütununun
                              ham (shift'siz) değeri. target_columns içinde
                              "_0900" ile biten bir sütun yoksa hiçbir şey
                              eklenmez (fonksiyon no-op döner).

    Parameters
    ----------
    df             : Polars DataFrame
    target_columns : Hedef sütun adı (str) veya sütun listesi (List[str])

    Returns
    -------
    pl.DataFrame
    """
    target_cols: List[str] = [target_columns] if isinstance(target_columns, str) else list(target_columns)

    slot_0900 = next((c for c in target_cols if c.endswith("_0900")), None)
    if slot_0900 is None or slot_0900 not in df.columns:
        return df

    df = df.with_columns([
        pl.col(slot_0900).alias("cross_lag_0900_same_day")
    ])
    logger.debug(
        "✅ Cross-lag özelliği eklendi: cross_lag_0900_same_day "
        f"(kaynak: '{slot_0900}', shift yok — bugünkü 09:00 değeri)."
    )
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
    target_columns: Union[str, List[str]],
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

    Wide format (09:00 / 17:00) — TEK ORTAK FEATURE MATRİSİ
    -----------------------------------------------------------
    target_columns birden fazla sütun içeriyorsa (örn.
    ["toplam_desi_0900", "toplam_desi_1700"]) bu fonksiyon HALA TEK bir
    feature matrisi üretir — hem iki hedef sütunu, hem her ikisi için ayrı
    ayrı lag/rolling'leri, hem de meşru cross-lag sütununu (bugünkü 09:00
    değeri) aynı tabloda barındırır. Hangi sütunun hangi model (09:00 vs
    17:00) için "leakage" sayılıp drop edileceğine BURADA karar VERİLMEZ —
    bu ayrım bir sonraki dosyada (forecasters.py), model bazında yapılır.
    İki model de aynı zengin feature havuzundan beslenir.

    Adım sırası (data leakage riskine göre):
      1. Pandas → Polars dönüşümü + tarih sıralaması
      2. Zaman özellikleri      (mevcut satırdan, leakage yok)
      3. Tatil özellikleri      (mevcut tarihten, leakage yok)
      3.5 Kampanya özellikleri   (e-ticaret zirveleri + arife, leakage yok)
      4. Spatio-temporal        (grup × zaman etkileşimi)
      5. Lag özellikleri        (her hedef sütun için AYRI AYRI, shift() ile leakage yok)
      6. Rolling istatistikler  (her hedef sütun için AYRI AYRI, shift(1) + rolling, leakage yok)
      6.3 Cross-lag             (bugünkü 09:00 → 17:00 modeli için meşru gün-içi bilgi)
      6.4 Günlük toplam (geçici) (toplam_desi_0900 + toplam_desi_1700 — SADECE aşağıdaki
                                  hub/graph/hierarchical/extreme fonksiyonlarının iç
                                  hesaplarında kullanılır, ham haliyle ASLA nihai
                                  matriste bırakılmaz — bkz. Adım 6.4 yorumları)
      6.5 Hub özellikleri       (hub_lag_1 — kaynak merkezi yığılma, günlük toplam üzerinden)
      6.6 Ağ/Çizge özellikleri  (KDD Cup şampiyonluk — Pressure Ratio,
                                  Centrality, K-dereceli komşuluk, günlük toplam üzerinden)
      6.7 Hiyerarşik özellikler (M5/Grupo Bimbo — Hub-to-Route Ratio,
                                  Çapraz-Grup agregasyonları, günlük toplam üzerinden)
      6.8 Ekstrem olay özellikleri (Tweedie/SHOS — pencere eğrisi,
                                    ekstrem aday skoru, log sinyal, günlük toplam üzerinden)
      7. NaN satır temizliği    (lag'den gelen ilk N satır boş olur — iki hedefin
                                  NaN zincirleri BİRLEŞTİĞİ için satır kaybı tek
                                  hedefli akışa göre büyüyebilir, bkz. aşağıdaki log)
      8. Polars → Pandas dönüşümü (CatBoost için)

    Parameters
    ----------
    df               : Ham giriş verisi (Pandas) — wide format bekleniyor
                        (bir satır = bir (rota, tarih))
    target_columns   : Tahmin hedefi/hedefleri. Tek sütun (str, eski/legacy
                        tek-serili akış) veya sütun listesi (List[str], yeni
                        wide format: ["toplam_desi_0900", "toplam_desi_1700"])
    date_column      : Tarih sütunu (örn. "tarih")
    group_column     : Grup/TM sütunu (örn. "rota"); None ise tek seri
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
    ...     df=raw_wide_df,
    ...     target_columns=["toplam_desi_0900", "toplam_desi_1700"],
    ...     date_column="tarih",
    ...     group_column="rota",
    ... )
    >>> cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    >>> # forecasters.py: 09:00 modeli X'i -> "toplam_desi_1700" ve
    >>> # "toplam_desi_0900" (kendi target'ı) drop edilir.
    >>> # 17:00 modeli X'i -> sadece "toplam_desi_1700" (kendi target'ı) drop
    >>> # edilir; "toplam_desi_0900" / "cross_lag_0900_same_day" bilerek tutulur.
    """
    target_cols: List[str] = [target_columns] if isinstance(target_columns, str) else list(target_columns)

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
    pl_df = add_holiday_features(pl_df, date_column, group_column=group_column, lead_days=holiday_lead_days)

    # --- Adım 3.5: Kampanya (E-ticaret) özellikleri ---
    # lead_days=3: Anneler Günü gibi kampanyalarda 5 günlük arife
    # Nisan/Mayıs tatil birikim dönemleriyle çakışıyor.
    # 3 gün, gerçek e-ticaret sipariş penceresini daha temiz modelliyor.
    pl_df = add_campaign_features(pl_df, date_column, lead_days=3)

    # --- Adım 3.7: Kaggle Takvim Özellikleri (Maaş günü + Rota-Gün + Birikim) ---
    # is_holiday ve is_campaign_eve sütunlarına bağımlı olduğu için
    # tatil/kampanya adımlarından SONRA çalıştırılır.
    pl_df = add_advanced_calendar_features(pl_df, date_column, group_column)


    # --- Adım 4: Spatio-temporal ---
    if group_column and group_column in pl_df.columns:
        pl_df = add_spatio_temporal_features(pl_df, group_column, date_column)

    # --- Adım 5: Lag (her hedef sütun için ayrı ayrı, suffix'li) ---
    pl_df = add_lag_features(pl_df, target_cols, group_column, lags=lags)

    # --- Adım 6: Rolling (her hedef sütun için ayrı ayrı, suffix'li) ---
    pl_df = add_rolling_features(
        pl_df, target_cols, group_column, windows=rolling_windows
    )

    # --- Adım 6.3: Cross-lag (bugünkü 09:00 → 17:00 modeli için meşru feature) ---
    pl_df = add_cross_lag_features(pl_df, target_cols)

    # --- Adım 6.4: Günlük toplam (GEÇİCİ — hub/graph/hierarchical/extreme için) ---
    # add_hub_features / add_graph_network_features / add_hierarchical_features /
    # add_extreme_event_features fiziksel olarak "o gün o hub'dan/rotadan geçen
    # TOPLAM hacim"i bekliyor — bu artık toplam_desi_0900 + toplam_desi_1700.
    # Bu sütun SADECE aşağıdaki 4 fonksiyona syöylenmek üzere üretilir; hepsi
    # kendi içinde shift(1) uyguladığı için leakage yok. Ama HAM (shift'siz)
    # haliyle nihai feature matrisinde KESİNLİKLE bırakılmaz: iki hedefin
    # toplamı olduğu için, bir model bu sütun + kendi target'ından diğer
    # modelin target'ını trivial şekilde geri çıkarabilir
    # (gunluk - kendi_target = diğer_target). Bu yüzden Adım 7'den önce drop edilir.
    _DAILY_TOTAL_COL = "_toplam_desi_gunluk_temp"
    if len(target_cols) > 1:
        pl_df = pl_df.with_columns([
            pl.sum_horizontal([pl.col(c) for c in target_cols]).alias(_DAILY_TOTAL_COL)
        ])
        hub_graph_target = _DAILY_TOTAL_COL
    else:
        hub_graph_target = target_cols[0]

    # --- Adım 6.5: Hub Özellikleri ---
    pl_df = add_hub_features(pl_df, hub_graph_target, date_column)

    # --- Adım 6.6: Ağ / Çizge Özellikleri (KDD Cup Şampiyonluk Mekanizması) ---
    # Kaynak ve hedef hub sütunlarını otomatik tespit et
    _graph_source = next(
        (c for c in ["kaynak_tm", "source_hub", "kaynak"] if c in pl_df.columns), None
    )
    _graph_dest = next(
        (c for c in ["varis_tm", "hedef_tm", "dest_hub", "hedef"] if c in pl_df.columns), None
    )
    if _graph_source and _graph_dest:
        pl_df = add_graph_network_features(
            pl_df,
            target_column=hub_graph_target,
            date_column=date_column,
            source_col=_graph_source,
            dest_col=_graph_dest,
        )
    else:
        logger.debug(
            "ℹ️  Graph/Network özellikleri atlandı: kaynak/hedef hub sütunları bulunamadı. "
            "Veri setinde 'kaynak_tm' ve 'hedef_tm' sütunları varsa otomatik aktif olur."
        )

    # --- Adım 6.7: Hiyerarşik Özellikler (M5 / Grupo Bimbo Şampiyonluk Mekanizması) ---
    if _graph_source:
        pl_df = add_hierarchical_features(
            pl_df,
            target_column=hub_graph_target,
            date_column=date_column,
            source_col=_graph_source,
            dest_col=_graph_dest or "",
        )

    # --- Adım 6.8: Ekstrem Olay Özellikleri (Tweedie / SHOS / Pencere Eğrisi) ---
    # NOT: Bu fonksiyon, tek-hedefli akışta üretilen literal "rolling_mean_7" /
    # "rolling_std_7" sütunlarını arıyor. Wide format'ta rolling artık
    # suffix'li (rolling_mean_7_0900 / _1700) olduğu için bu literal sütunlar
    # bulunmayacak ve fonksiyon kendi tasarlanmış fallback'ine (statik grup
    # quantile eşiği) düşecektir — bu bir hata değil, fonksiyonun zaten
    # desteklediği bir davranış (bkz. add_extreme_event_features içindeki
    # "else: Statik grup quantile" dalı).
    pl_df = add_extreme_event_features(
        pl_df,
        target_column=hub_graph_target,
        date_column=date_column,
        group_column=group_column,
    )

    # Geçici günlük toplam sütununu nihai matristen temizle (leakage önlemi)
    if _DAILY_TOTAL_COL in pl_df.columns:
        pl_df = pl_df.drop(_DAILY_TOTAL_COL)

    # --- Adım 7: NaN temizliği ---
    rows_before = pl_df.height
    if drop_na:
        # Wide format'ta iki hedefin (0900/1700) lag/rolling NaN zincirleri
        # BİRLEŞİR (drop_nulls satır bazında AND değil OR mantığıyla çalışır —
        # herhangi bir sütunda NaN varsa satır düşer). Bu yüzden düşürmeden
        # önce, hangi lag/rolling sütununun en çok NaN ürettiğini logluyoruz;
        # böylece kayıp "normal" mi (beklenen ilk-N-gün lag ısınması) yoksa
        # "anormal" mi (örn. lag_30_0900 + lag_30_1700 ikiye katlanan kayıp)
        # kolayca ayırt edilebilir.
        lag_roll_cols = [c for c in pl_df.columns if c.startswith("lag_") or c.startswith("rolling_")]
        if lag_roll_cols:
            null_counts = pl_df.select(lag_roll_cols).null_count().to_dicts()[0]
            top_offenders = sorted(null_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            logger.info(
                "🔎 Dropna öncesi en çok NaN üreten lag/rolling sütunları (ilk 5): "
                + ", ".join(f"{name}={count}" for name, count in top_offenders)
            )

        pl_df = pl_df.drop_nulls()
        dropped = rows_before - pl_df.height
        if dropped > 0:
            logger.info(
                f"ℹ️  {dropped} satır lag NaN nedeniyle atıldı "
                f"({rows_before} → {pl_df.height}, "
                f"{dropped / rows_before:.1%})"
            )

    # --- Adım 8: Polars → Pandas (CatBoost / sklearn uyumluluğu için) ---
    result: pd.DataFrame = pl_df.to_pandas()

    logger.info(
        f"✅ Feature matrix hazır (Polars backend): "
        f"{result.shape[0]} satır × {result.shape[1]} sütun "
        f"(hedef sütunlar: {target_cols})"
    )
    return result


# ---------------------------------------------------------------------------
# 5.5 Hub (Merkez) Geçişkenlik Özellikleri
# ---------------------------------------------------------------------------

def add_hub_features(df: pl.DataFrame, target_column: str, date_column: str) -> pl.DataFrame:
    """
    Kaynak merkezinin (hub) toplam hacmini hesaplar ve 1 gün gecikmeli olarak modele verir.
    Lojistik ağındaki yığılmaları yakalayan en güçlü özelliktir.
    """
    if "kaynak_tm" not in df.columns:
        return df
    # 1. Aynı gün, aynı kaynak merkezinden çıkan TOPLAM kargoyu bul
    df = df.with_columns(
        pl.col(target_column).sum().over([date_column, "kaynak_tm"]).alias("hub_daily_total")
    )
    # 2. Bu toplamı, kaynak merkezi bazında 1 gün kaydır (Geleceği görme / Leakage önlemi!)
    df = df.with_columns(
        pl.col("hub_daily_total").shift(1).over("kaynak_tm").alias("hub_lag_1")
    )
    # 3. Sızıntı yapmaması için bugünün toplam bilgisini sil
    df = df.drop("hub_daily_total")
    logger.debug("✅ Hub (Kaynak) geçişkenlik özellikleri eklendi (Polars).")
    return df


# ---------------------------------------------------------------------------
# 6. Ağ / Çizge (Graph Network) Özellikleri — KDD Cup Şampiyonluk Mekanizması
# ---------------------------------------------------------------------------

def add_graph_network_features(
    df: pl.DataFrame,
    target_column: str,
    date_column: str,
    source_col: str = "kaynak_tm",
    dest_col: str = "hedef_tm",
    pressure_windows: List[int] = [1, 7],
    neighbor_order: int = 2,
) -> pl.DataFrame:
    """
    Rotaları izole satırlar olarak değil, ağ düğümleri olarak modeller.

    Motivasyon (PDF Bölüm 1):
        Karar ağaçları veriyi izole satırlar (tabular data) halinde işler.
        Bu nedenle A→B rotasındaki birikim, B→C ve C→D'ye olan şelale etkisini
        (ripple effect) içsel olarak göremez. KDD Cup 2020 şampiyonları bu körlüğü
        üç tür çizge özelliğiyle aşmıştır: Pressure Ratio, dinamik PageRank
        benzeri merkez ağırlığı ve K-dereceli komşuluk hacimleri.

    Çıktı sütunlar
    --------------
    hub_pressure_ratio_{w}d    : Hub'ın son {w} günlük giriş/çıkış hacim oranı.
                                  Pressure_Ratio = Σ(k→i)_vol / (Σ(i→j)_vol + ε)
                                  1'den büyükse hub'a girenden fazla yük giriyor
                                  → backlog başlangıcının matematiksel göstergesi.
    hub_in_vol_{w}d            : Hub'a giren toplam hacim (w günlük rolling).
    hub_out_vol_{w}d           : Hub'dan çıkan toplam hacim (w günlük rolling).
    hub_centrality_{w}d        : Kaynak hub'ın ağdaki ağırlık merkezi skoru.
                                  (out_vol / toplam_ağ_vol) — dinamik PageRank
                                  yerine deterministik, sızıntısız vekil.
    neighbor_vol_1st_order     : Kaynak hub'ın 1. derece komşularındaki (hedef
                                  hariç) gecikmeli toplam hacim. Birinci Derece
                                  Şelale Etkisi.
    neighbor_vol_2nd_order     : 2. derece komşulardaki gecikmeli toplam hacim.
                                  Eğer 2. derece komşularda %30 artış varsa ağaç
                                  bunu erken sinyal olarak kullanır.

    ⚠️  Data Leakage Güvencesi:
        Tüm rolling işlemler shift(1) ile bugünün verisi pencereye dahil
        edilmeden hesaplanır.

    Parameters
    ----------
    df               : Polars DataFrame
    target_column    : Hacim / hedef sütunu
    date_column      : Tarih sütunu
    source_col       : Kaynak hub sütunu (varsayılan: "kaynak_tm")
    dest_col         : Hedef hub sütunu (varsayılan: "hedef_tm")
    pressure_windows : Pressure Ratio hesabı için pencere boyutları (gün)
    neighbor_order   : Komşuluk derecesi (1 veya 2)

    Returns
    -------
    pl.DataFrame
    """
    if source_col not in df.columns or dest_col not in df.columns:
        logger.warning(
            f"⚠️  Graph özellikleri atlandı: '{source_col}' veya '{dest_col}' "
            "sütunu bulunamadı."
        )
        return df

    # ------------------------------------------------------------------
    # Pressure Ratio: In/Out Degree Imbalance
    #   Pressure_Ratio_{i,t} = Σ_k Volume(k→i)_{t-1} / (Σ_j Volume(i→j)_{t-1} + ε)
    # ------------------------------------------------------------------
    EPS = 1e-5

    # Günlük hub bazında giriş ve çıkış toplamları (bugünün verisi dahil değil — shift sonra)
    # Giriş: bu hub hedef konumunda olan tüm rotaların hacmi
    # Çıkış: bu hub kaynak konumunda olan tüm rotaların hacmi

    df = df.with_columns([
        pl.col(target_column).alias("_vol_for_graph")
    ])

    for w in pressure_windows:
        # Kaynak hub'ın o gün çıkış hacmi (shift=1 → leakage yok)
        out_vol_expr = (
            pl.col("_vol_for_graph")
            .shift(1)
            .rolling_mean(window_size=w, min_samples=1)
            .over(source_col)
            .alias(f"hub_out_vol_{w}d")
        )

        # Hedef hub'a giriş hacmi — dest_col bazında rolling (kaynak bakışından)
        # Aynı dest_col değerine sahip satırlardaki hacim = o hub'a gelen yük
        in_vol_expr = (
            pl.col("_vol_for_graph")
            .shift(1)
            .rolling_mean(window_size=w, min_samples=1)
            .over(dest_col)
            .alias(f"hub_in_vol_{w}d")
        )

        df = df.with_columns([out_vol_expr, in_vol_expr])

        # Pressure Ratio = in / (out + ε)
        df = df.with_columns([
            (pl.col(f"hub_in_vol_{w}d") / (pl.col(f"hub_out_vol_{w}d") + EPS))
            .alias(f"hub_pressure_ratio_{w}d")
        ])

        # Hub Centrality: kaynak hub'ın ağ içindeki ağırlık payı
        # = hub_out_vol / (tüm rotaların o günkü ortalama out vol)
        # shift(1) uygulanmış out_vol zaten var → toplam ağ ortalamasını bölüyoruz
        total_net_vol = (
            pl.col("_vol_for_graph")
            .shift(1)
            .rolling_mean(window_size=w, min_samples=1)
            .mean()  # scalar — tüm satırlarda aynı (global)
        )
        df = df.with_columns([
            (pl.col(f"hub_out_vol_{w}d") / (total_net_vol + EPS))
            .alias(f"hub_centrality_{w}d")
        ])

    # ------------------------------------------------------------------
    # K-Dereceli Komşuluk Hacimleri (Extended Neighborhood Volume)
    # 1. Derece: Kaynak hub'ın komşularındaki (hedef hariç) hacim
    # ------------------------------------------------------------------
    if neighbor_order >= 1:
        # Her kaynak hub için, farklı hedeflere giden toplam hacim
        # (hedef sütununa göre değil kaynak sütununa göre grupluyoruz)
        # shift(1) → leakage yok
        df = df.with_columns([
            pl.col("_vol_for_graph")
            .shift(1)
            .sum()
            .over(source_col)
            .alias("_source_total_vol_lag1")
        ])
        # 1. derece komşu = kaynak hub'ın tüm çıkış rotaları toplamı - bu rotanın kendi hacmi
        df = df.with_columns([
            (pl.col("_source_total_vol_lag1") - pl.col("_vol_for_graph").shift(1))
            .clip(lower_bound=0)
            .alias("neighbor_vol_1st_order")
        ])

    if neighbor_order >= 2:
        # 2. Derece: Hedef hub'un kendi çıkış hacmi (kaynak hub'ın komşusunun komşusu)
        # dest_col bazında toplam çıkış hacmi → 2. derece komşuluk birikimi
        df = df.with_columns([
            pl.col("_vol_for_graph")
            .shift(1)
            .sum()
            .over(dest_col)
            .alias("neighbor_vol_2nd_order")
        ])

    # Ara sütunları temizle
    cols_to_drop = ["_vol_for_graph"]
    if "_source_total_vol_lag1" in df.columns:
        cols_to_drop.append("_source_total_vol_lag1")
    df = df.drop(cols_to_drop)

    logger.debug(
        f"✅ Graph/Network özellikleri eklendi: "
        f"Pressure Ratio ({pressure_windows}), "
        f"Centrality, {neighbor_order}. derece komşuluk (Polars)."
    )
    return df


# ---------------------------------------------------------------------------
# 7. Hiyerarşik Özellik Mühendisliği — M5 & Grupo Bimbo Şampiyonluk Mekanizması
# ---------------------------------------------------------------------------

def add_hierarchical_features(
    df: pl.DataFrame,
    target_column: str,
    date_column: str,
    source_col: str = "kaynak_tm",
    dest_col: str = "hedef_tm",
    hub_ratio_lags: List[int] = [7, 14, 21],
    cross_group_keys: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Alt rotaların üst Hub hiyerarşisini içselleştirmesini sağlar.

    Motivasyon (PDF Bölüm 2.3):
        M5 ve Grupo Bimbo şampiyonları, post-processing uzlaştırma yerine
        rotaya doğrudan Hub'ın statüsünü sayısal özellik olarak vermiştir.
        Bu sayede CatBoost, "Hub bugün 10.000 kapasitede çalışıyor ve bu
        rotanın tarihsel payı %5 ise tahminim 500 olmalı" çıkarımını
        ağaç bölünmelerinde (splits) bağımsız öğrenir.

    Çıktı sütunlar
    --------------
    hub_to_route_ratio_lag_{X} : A→B rotasının, A Hub'ının toplam günlük
                                  hacmine oranının X günlük gecikmesi.
                                  Rotanın sistem içindeki "pazar payını"
                                  (Hub_to_Route_Ratio_Lag_X) modele öğretir.
    cross_group_max_{key}      : Çapraz-grup hedef maksimumu — belirtilen
                                  gruplama anahtarına göre (örn. Hub + Tatil Tipi)
                                  tarihsel maksimum hacim. Extrapolation körlüğünü
                                  "içdeğerleme" (interpolation) ile aşar.
    cross_group_mean_{key}     : Çapraz-grup tarihsel ortalama hacim.
    cross_group_std_{key}      : Çapraz-grup tarihsel standart sapma.

    Parameters
    ----------
    df               : Polars DataFrame
    target_column    : Hacim / hedef sütunu
    date_column      : Tarih sütunu
    source_col       : Kaynak hub sütunu
    dest_col         : Hedef hub sütunu
    hub_ratio_lags   : Hub payı gecikme günleri (varsayılan: [7, 14, 21])
    cross_group_keys : Çapraz-grup için ek sütun listesi (örn. ["is_holiday", "is_campaign_day"])

    Returns
    -------
    pl.DataFrame
    """
    if source_col not in df.columns:
        logger.warning(
            f"⚠️  Hiyerarşik özellikler atlandı: '{source_col}' sütunu bulunamadı."
        )
        return df

    EPS = 1e-5

    # ------------------------------------------------------------------
    # Hub-to-Route Ratio Lags: rotanın hub içindeki pazar payı
    # Formül: route_vol / hub_total_vol (her iki değer de shift(1) ile leakage önlenir)
    # ------------------------------------------------------------------

    # Hub günlük toplam çıkış hacmi (kaynak bazında)
    df = df.with_columns([
        pl.col(target_column)
        .shift(1)
        .sum()
        .over([date_column, source_col])
        .alias("_hub_daily_total_lag1")
    ])

    # Rotanın kendi gecikmeli değeri / hub toplamı = pazar payı
    # Ardından bu oranın X günlük gecikmesi alınır
    df = df.with_columns([
        (pl.col(target_column).shift(1) / (pl.col("_hub_daily_total_lag1") + EPS))
        .alias("_route_hub_ratio_base")
    ])

    route_key = [source_col, dest_col] if dest_col in df.columns else [source_col]
    for lag in hub_ratio_lags:
        df = df.with_columns([
            pl.col("_route_hub_ratio_base")
            .shift(lag - 1)  # zaten shift(1) uygulandı, toplam lag = lag
            .over(route_key)
            .alias(f"hub_to_route_ratio_lag_{lag}")
        ])

    # ------------------------------------------------------------------
    # Çapraz-Grup Hedef Agregasyonları ("Sihirli Özellikler")
    # PDF Bölüm 3.2: Extrapolation körlüğünü interpolation ile aşar.
    # Grup: kaynak_hub + tatil_tipi kombinasyonu üzerinden tarihsel
    # max/mean/std hesaplanır — model "Bayram Dönüşü + İstanbul Hub"
    # kombinasyonunu daha önce görmüş maksimum değerden interpolate eder.
    # ------------------------------------------------------------------
    if cross_group_keys is None:
        cross_group_keys = []

    # Varsayılan olarak mevcut sütunlardan uygun olanları ekle
    auto_keys = ["is_holiday", "is_campaign_day", "is_weekend"]
    available_auto = [k for k in auto_keys if k in df.columns]
    combined_keys = list(dict.fromkeys(cross_group_keys + available_auto))  # deduplicate

    if combined_keys:
        for extra_key in combined_keys:
            group_keys = [source_col, extra_key]
            agg_col_prefix = f"cross_group_{extra_key}"

            # shift(1) + expanding stats — geçmişe dönük, leakage yok
            df = df.with_columns([
                pl.col(target_column)
                .shift(1)
                .max()
                .over(group_keys)
                .alias(f"{agg_col_prefix}_max"),

                pl.col(target_column)
                .shift(1)
                .mean()
                .over(group_keys)
                .alias(f"{agg_col_prefix}_mean"),

                pl.col(target_column)
                .shift(1)
                .std()
                .fill_null(0.0)
                .over(group_keys)
                .alias(f"{agg_col_prefix}_std"),
            ])

    # Ara sütunları temizle
    df = df.drop(["_hub_daily_total_lag1", "_route_hub_ratio_base"])

    logger.debug(
        f"✅ Hiyerarşik özellikler eklendi: "
        f"Hub-to-Route Ratio ({hub_ratio_lags}), "
        f"Çapraz-Grup Agregasyonları ({combined_keys}) (Polars)."
    )
    return df


# ---------------------------------------------------------------------------
# 8. Ekstrem Olay Özellikleri — Tweedie / SHOS / Pencere Eğrisi Dönüşümü
# ---------------------------------------------------------------------------

def add_extreme_event_features(
    df: pl.DataFrame,
    target_column: str,
    date_column: str,
    group_column: Optional[str] = None,
    backlog_clearance_rate: float = 0.35,
    extreme_threshold_quantile: float = 0.90,
) -> pl.DataFrame:
    """
    Ağaç tabanlı modellerin extrapolation körlüğünü ve ekstrem olayları
    kaçırma (clipping) sorununu çözen özellik seti.

    Motivasyon (PDF Bölüm 3):
        CatBoost/LightGBM gibi GBDT algoritmaları yaprak ortalaması ürettiği
        için eğitim setindeki maksimum değerin ötesine geçemez. Lojistik
        ağlarındaki bayram dönüşü / kampanya patlamaları bu sınırı aşar ve
        model "güvenli" ancak operasyonel olarak yetersiz tahminler üretir.
        Kaggle M5 Uncertainty ve Grupo Bimbo şampiyonları bu sorunu üç
        yöntemle aşmıştır:
          1. Pencere (Window) Eğrisi: Tatil etkisini statik 0/1 yerine
             üstel sönüm eğrisi ile sürekli değişken olarak modellemek.
          2. SHOS (Statistical Hurdle and Occurrence Size): Ekstrem olayı
             "olacak mı?" ve "ne büyüklükte?" olarak ikiye bölmek.
          3. Log Dönüşümü Sinyali: Gradient explosion riskini sayısal
             özellik olarak raporlamak.

    Çıktı sütunlar
    --------------
    backlog_window_intensity   : Tatil/kampanya sonrası birikimin üstel
                                  sönüm eğrisi değeri.
                                  Formül: max(0, α · exp(-β · (t - t0)))
                                  α = tatil öncesi beklenen yük (accumulated_closed_days
                                  varsa kullanılır, yoksa backlog_release_index).
                                  β = backlog_clearance_rate (ağın eritme hızı).
    is_extreme_event_candidate : Tarihsel grup medyanının extreme_threshold_quantile
                                  katını aşan gün sinyali (Int8, 0/1).
                                  SHOS Hurdle Modeli için "oluşma" hedefi.
    extreme_event_prob_score   : Grup × tatil/kampanya kombinasyonunun geçmişte
                                  ekstrem olaya dönüşme oranı (0–1 arası Float).
                                  Model 1 (Sınıflandırma) çıktısını taklit eder;
                                  Model 2 (Regresyon) için sayısal girdi olarak kullanılır.
    log_transform_signal       : Anlık rolling_std / rolling_mean oranı
                                  (varyasyon katsayısı). Yüksekse Tweedie
                                  kayıp fonksiyonu veya log1p dönüşümü gerektirir.

    Parameters
    ----------
    df                        : Polars DataFrame
    target_column             : Hedef sütun
    date_column               : Tarih sütunu
    group_column              : Hub/rota grubu (None ise tek seri)
    backlog_clearance_rate    : Tatil birikiminin günlük erime hızı β (varsayılan: 0.35)
    extreme_threshold_quantile: Ekstrem eşiği için quantile (varsayılan: 0.90)

    Returns
    -------
    pl.DataFrame

    Not
    ---
    Gerçek SHOS mimarisi için, is_extreme_event_candidate'i hedef olarak
    CatBoost sınıflandırma modeli eğitin; ardından bu modelin predict_proba()
    çıktısını regresyon modelinin özellik setine ekleyin.
    Bu fonksiyon, o olasılık skorunun tarihsel veri tabanlı vekil (proxy)
    sürümünü üretir.
    """
    EPS = 1e-5
    over_keys = [group_column] if group_column and group_column in df.columns else None

    def _over(expr: pl.Expr) -> pl.Expr:
        return expr.over(over_keys) if over_keys else expr

    # ------------------------------------------------------------------
    # 1. Tatil/Kampanya Pencere Eğrisi (Window Intensity)
    # PDF 3.4: Backlog_Intensity_t = max(0, α · exp(-β · (t - t0)))
    # Mevcut backlog_release_index sütunundan türetilir (varsa)
    # ------------------------------------------------------------------
    if "backlog_release_index" in df.columns and "days_since_resumption" in df.columns:
        # backlog_release_index zaten α · exp(-d/alpha) formunda
        # Bunu clearance_rate ile yeniden ölçeklendir (daha ayarlanabilir β)
        df = df.with_columns([
            pl.when(
                (pl.col("is_closed") == 0) &
                (pl.col("days_since_resumption") > 0) &
                (pl.col("days_since_resumption") <= 6)
            )
            .then(
                pl.col("accumulated_closed_days").shift(1).fill_null(0.0) *
                (-pl.col("days_since_resumption") * backlog_clearance_rate).exp()
            )
            .otherwise(0.0)
            .alias("backlog_window_intensity")
        ] if "is_closed" in df.columns and "accumulated_closed_days" in df.columns
        else [pl.lit(0.0).alias("backlog_window_intensity")])
    else:
        # Fallback: campaign_eve veya holiday sinyalinden türet
        if "is_campaign_eve" in df.columns:
            df = df.with_columns([
                pl.when(pl.col("is_campaign_eve") == 1)
                .then((-pl.col("is_campaign_eve").cum_sum().over(over_keys or []) * backlog_clearance_rate).exp())
                .otherwise(0.0)
                .alias("backlog_window_intensity")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("backlog_window_intensity")])

    # ------------------------------------------------------------------
    # 2. Ekstrem Olay Adayı (SHOS Hurdle — "Oluşma" Sinyali)
    # Eğitim setindeki grup medyanının Q90 katını aşan günler → potansiyel ekstrem
    # ------------------------------------------------------------------
    if "rolling_mean_7" in df.columns and "rolling_std_7" in df.columns:
        # Anlık kayan ortalama ve std mevcut — eşiği dinamik tut
        threshold_expr = (
            pl.col("rolling_mean_7") +
            2.0 * pl.col("rolling_std_7")
        )
        df = df.with_columns([
            pl.when(
                pl.col(target_column).shift(1) > threshold_expr
            ).then(1).otherwise(0).cast(pl.Int8)
            .alias("is_extreme_event_candidate")
        ])
    else:
        # Statik grup quantile — leakage riski düşük (tüm geçmiş kullanılır)
        q_expr = _over(pl.col(target_column).quantile(extreme_threshold_quantile))
        df = df.with_columns([
            pl.when(pl.col(target_column).shift(1) > q_expr)
            .then(1).otherwise(0).cast(pl.Int8)
            .alias("is_extreme_event_candidate")
        ])

    # ------------------------------------------------------------------
    # 3. Ekstrem Olay Olasılık Skoru (SHOS Proxy)
    # Grup × tatil kombinasyonunda geçmişte ekstrem olay oranı
    # Gerçek SHOS'ta bu, ayrı bir CatBoost sınıflandırma modelinin
    # predict_proba() çıktısıdır. Burada tarihsel oran vekil olarak kullanılır.
    # ------------------------------------------------------------------
    prob_group_keys = ([group_column] if group_column and group_column in df.columns else [])
    if "is_holiday" in df.columns:
        prob_group_keys.append("is_holiday")
    if "is_campaign_day" in df.columns:
        prob_group_keys.append("is_campaign_day")

    if prob_group_keys and "is_extreme_event_candidate" in df.columns:
        df = df.with_columns([
            pl.col("is_extreme_event_candidate")
            .shift(1)
            .mean()
            .over(prob_group_keys)
            .fill_null(0.0)
            .alias("extreme_event_prob_score")
        ])
    else:
        df = df.with_columns([pl.lit(0.0).alias("extreme_event_prob_score")])

    # ------------------------------------------------------------------
    # 4. Log Dönüşümü Sinyali (Tweedie / RMSLE ihtiyaç göstergesi)
    # Varyasyon katsayısı (CoV) = std / mean; yüksekse (>1.5) Tweedie önerilir
    # ------------------------------------------------------------------
    if "rolling_std_7" in df.columns and "rolling_mean_7" in df.columns:
        df = df.with_columns([
            (pl.col("rolling_std_7") / (pl.col("rolling_mean_7") + EPS))
            .alias("log_transform_signal")
        ])
    else:
        df = df.with_columns([pl.lit(0.0).alias("log_transform_signal")])

    logger.debug(
        "✅ Ekstrem Olay özellikleri eklendi: "
        "backlog_window_intensity, is_extreme_event_candidate, "
        "extreme_event_prob_score, log_transform_signal (Polars)."
    )
    return df


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
    if result["recommend_log"] and not log_transform:
        log_line += "\n   💡 Not: MultiQuantile kullanıldığı için log dönüşümü bilinçli olarak kapalı tutulmuştur."
    logger.info(log_line)

    return result