"""
run_forecast.py — Teknofest 2026 Tahmin Çalıştırıcı
=====================================================
Kullanım:
    python run_forecast.py

Giriş  : Desi_talep.xlsx
Çıkış  : ALNS motoruna in-memory JSON (List[Dict])
         İsteğe bağlı: alns_payload.json (debug için)

Mimari:
    DemandForecaster.fit()   → Tek CatBoost Modeli (MultiQuantile: q10/q50/q90)
    DemandForecaster.predict() → List[Dict] (tarih, kaynak_tm, varis_tm, q10, q50, q90)
    UncertaintyBand.to_alns_payload() → ALNS formatı (risk_class, safety_buffer, ...)
"""

import json
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Proje modülleri — src/ altında
sys.path.insert(0, str(Path(__file__).parent))
from src.forecasters import DemandForecaster
from src.uncertainty import UncertaintyBand

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

# Proje kökü: run_forecast.py — src/predict_model/ içinde
_HERE          = Path(__file__).resolve().parent          # src/predict_model/
_PROJECT_ROOT  = _HERE.parent.parent                      # routetech_lojistik/

DATA_PATH      = str(_PROJECT_ROOT / "data" / "raw" / "Desi_talep.xlsx")
PREDICT_START  = "2026-05-11"
PREDICT_END    = "2026-05-17"
OUTPUT_JSON    = str(_PROJECT_ROOT / "alns_payload.json")  # debug için; ALNS motoru RAM'den alır

TARGET_COL  = "desi_hacmi"
DATE_COL    = "tarih"
GROUP_COL   = "rota"          # kaynak_tm → varış_tm kombinasyonu
KAYNAK_COL  = "kaynak_tm"
VARIS_COL   = "varis_tm"


# ---------------------------------------------------------------------------
# 1. Veri Hazırlama
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """
    Excel → DemandForecaster'ın beklediği formata dönüştür.

    DemandForecaster group_column olarak tek bir sütun bekliyor.
    Dataset A'da grup = kaynak_tm + varış_tm kombinasyonu → 'rota' sütunu.

    Eksik gün × rota kombinasyonları 0 ile doldurulur:
      - Lag/rolling feature'lar için gerekli (süreksizlik = NaN zinciri)
      - Model "o gün o rotada taşıma yok" örüntüsünü öğrenir

    Sütun eşleştirme stratejisi (Dataset B'ye dayanıklılık):
      1. Bilinen Türkçe/İngilizce sütun adlarına isim bazlı bak
      2. Bulunamazsa pozisyon bazlı fallback (uyarı logla)
      Böylece Dataset B farklı sütun adı veya sırası gelse de çalışır.
    """
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Bilinen olası sütun adları (Dataset A Türkçe + Dataset B İngilizce varyantlar)
    _KAYNAK_CANDIDATES = ["Çıkış Transfer Merkezi", "kaynak_tm", "source", "origin", "from"]
    _VARIS_CANDIDATES  = ["Varış Transfer Merkezi", "varis_tm",  "destination", "dest", "to"]
    _DATE_CANDIDATES   = ["Tarih", "tarih", "date", "Date", "tarih_"]
    _TARGET_CANDIDATES = ["Toplam Desi", "desi_hacmi", "desi", "demand", "talep"]

    def _find_col(candidates: list, col_idx: int, label: str) -> str:
        """Sütun adını isim bazlı ara, bulamazsan pozisyon bazlı fallback."""
        cols_lower = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        # Fallback: pozisyon bazlı
        fallback = df.columns[col_idx]
        logger.warning(
            f"⚠️  '{label}' sütunu isim bazlı bulunamadı → "
            f"pozisyon [{col_idx}] kullanılıyor: '{fallback}'\n"
            f"   Beklenen adlardan biri: {candidates}"
        )
        return fallback

    kaynak_col_raw = _find_col(_KAYNAK_CANDIDATES, 0, "kaynak_tm")
    varis_col_raw  = _find_col(_VARIS_CANDIDATES,  1, "varis_tm")
    date_col_raw   = _find_col(_DATE_CANDIDATES,   2, "tarih")
    target_col_raw = _find_col(_TARGET_CANDIDATES, 3, "desi_hacmi")

    df = df.rename(columns={
        kaynak_col_raw: KAYNAK_COL,
        varis_col_raw:  VARIS_COL,
        date_col_raw:   DATE_COL,
        target_col_raw: TARGET_COL,
    })
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[GROUP_COL] = df[KAYNAK_COL] + " → " + df[VARIS_COL]

    # Tam grid: tüm rota × tüm gün (eksikler 0)
    all_dates  = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    all_routes = df[GROUP_COL].unique()
    rota_map   = df[[GROUP_COL, KAYNAK_COL, VARIS_COL]].drop_duplicates()

    idx    = pd.MultiIndex.from_product([all_routes, all_dates], names=[GROUP_COL, DATE_COL])
    full   = pd.DataFrame(index=idx).reset_index()
    full   = full.merge(rota_map, on=GROUP_COL, how="left")
    full   = full.merge(df[[GROUP_COL, DATE_COL, TARGET_COL]], on=[GROUP_COL, DATE_COL], how="left")
    full[TARGET_COL] = full[TARGET_COL].fillna(0.0)
    full   = full.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    logger.info(
        f"✅ Veri hazır: {len(df):,} gerçek kayıt | "
        f"{full[GROUP_COL].nunique()} rota | {full[DATE_COL].nunique()} gün\n"
        f"   {full[DATE_COL].min().date()} → {full[DATE_COL].max().date()}"
    )
    return full


def build_predict_grid(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    11-17 Mayıs için boş tahmin grid'i oluştur.
    Context buffer (son 35 gün) eklenerek lag/rolling doğru hesaplanır.
    """
    target_dates = pd.date_range(PREDICT_START, PREDICT_END, freq="D")
    all_routes   = full_df[GROUP_COL].unique()

    rows = []
    for route in all_routes:
        info = full_df[full_df[GROUP_COL] == route][[KAYNAK_COL, VARIS_COL]].iloc[0]
        for d in target_dates:
            rows.append({
                GROUP_COL:  route,
                DATE_COL:   d,
                KAYNAK_COL: info[KAYNAK_COL],
                VARIS_COL:  info[VARIS_COL],
                TARGET_COL: np.nan,
            })

    pred_df   = pd.DataFrame(rows)
    buf_start = pd.Timestamp(PREDICT_START) - pd.Timedelta(days=35)
    buffer    = full_df[full_df[DATE_COL] >= buf_start].copy()
    combined  = pd.concat([buffer, pred_df], ignore_index=True)
    combined  = combined.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    # NaN → 0 (lag kaynağı için), tahmin haftasında model bu değerleri üretecek
    combined[TARGET_COL] = combined[TARGET_COL].fillna(0.0)
    return combined


# ---------------------------------------------------------------------------
# 2. Tahmin + ALNS Payload
# ---------------------------------------------------------------------------

def run(save_json: bool = True) -> Dict[str, Any]:
    """
    Ana akış:
      1. Veri yükle
      2. DemandForecaster.fit() — 3 ayrı CatBoost
      3. DemandForecaster.predict() — List[Dict] (q10/q50/q90)
      4. UncertaintyBand.to_alns_payload() — ALNS formatı
      5. İsteğe bağlı JSON kaydet (debug)
    """
    logger.info("=" * 60)
    logger.info("🚀 Teknofest 2026 — Tahmin Motoru")
    logger.info("=" * 60)

    # --- 1. Veri ---
    full_df = load_dataset(DATA_PATH)

    # --- 2. Fit ---
    forecaster = DemandForecaster(
        target_column    = TARGET_COL,
        date_column      = DATE_COL,
        group_column     = GROUP_COL,
        train_test_split = 0.85,
        forecast_horizon = 7,
        lags             = [1, 7, 14, 21, 30],  # lag_30 geri eklendi: aylık sezonsallık
        rolling_windows  = [7, 14],
        logging_enabled  = True,
        random_state     = 42,
    )
    forecaster.fit(full_df)

    # --- Feature Importance ---
    importances = forecaster.get_feature_importances()
    print("\n\U0001f525 En Önemli 10 Feature:")
    print(importances.head(10))

    # --- 3. Predict grid ---
    predict_grid = build_predict_grid(full_df)
    all_preds: List[Dict[str, Any]] = forecaster.predict(predict_grid)

    # Buffer satırlarını çıkar — sadece PREDICT_START/END aralığı kalsın
    target_dates = set(
        pd.date_range(PREDICT_START, PREDICT_END, freq="D").strftime("%Y-%m-%d")
    )
    raw_preds = [r for r in all_preds if str(r[DATE_COL])[:10] in target_dates]

    logger.info(
        f"\n✅ Tahmin tamamlandı: {len(raw_preds)} kayıt "
        f"({PREDICT_START} → {PREDICT_END}, "
        f"{full_df[GROUP_COL].nunique()} rota × 7 gün)"
    )

    # --- 4. OR-Tools Payload (DataFrame Dönüşümü) ---
    band = UncertaintyBand(buffer_ratio=0.5, logging_enabled=True)

    df_ortools = band.to_ortools_dataframe(
        predictions = raw_preds,
        date_key    = DATE_COL,
        group_key   = GROUP_COL,
    )
    # --- 5. CSV Kaydet (OR-Tools doğrudan bu dosyayı okuyacak) ---
    payload = band.to_alns_payload()
    if save_json:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"ALNS payload kaydedildi: {OUTPUT_JSON}")

    OUTPUT_CSV = str(_PROJECT_ROOT / "ortools_payload.csv")
    df_ortools.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"💾 OR-Tools payload kaydedildi: {OUTPUT_CSV}")
    logger.info(
        f"\n{'='*60}\n"
        f"✅ Tamamlandı!\n"
        f"   Tahmin sayısı  : {len(df_ortools)}\n"
        f"   Tarih aralığı  : {PREDICT_START} → {PREDICT_END}\n"
        f"============================================================"
    )
    print("\n📋 OR-Tools Payload Örnek (İlk 5 Satır):")
    print(df_ortools.head().to_string(index=False))

    return df_ortools


# ---------------------------------------------------------------------------
# Örnek payload çıktısı
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run(save_json=True)