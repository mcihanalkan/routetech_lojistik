"""
optimize.py — Hiperparametre Optimizasyon Scripti
===================================================
Ayrı çalıştırılır (run_forecast.py ile değil).
Gerçek veri üzerinde Optuna çalıştırır, en iyi parametreleri
hyperparams_map.json'a yazar.

Kullanım:
    python optimize.py                          # varsayılan: 50 trial, 900s timeout
    python optimize.py --data baska_veri.xlsx  # farklı veriyle
    python optimize.py --trials 50 --timeout 1200
    python optimize.py --trials 50 --timeout 0  # timeout yok, sadece trial sayısı sınırı

Her çalıştırmada JSON'daki ilgili bucket güncellenir.
Yeni bir veri boyutu (örn. 500K satır) gelirse otomatik yeni bucket eklenir.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).parent))
from src.features import build_feature_matrix, get_categorical_columns

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("❌ Optuna bulunamadı: pip install optuna")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# scripts/ içinden çalışıyorsa bir üst = proje kökü
_PROJECT_ROOT    = Path(__file__).parent
HYPERPARAMS_PATH = _PROJECT_ROOT / "hyperparams_map.json"
TARGET_COL  = "desi_hacmi"
DATE_COL    = "tarih"
GROUP_COL   = "rota"
KAYNAK_COL  = "kaynak_tm"
VARIS_COL   = "varis_tm"


# ---------------------------------------------------------------------------
# Veri boyutuna göre Optuna arama uzayı
# ---------------------------------------------------------------------------

def _search_space(n_rows: int) -> dict:
    """
    Veri boyutuna göre Optuna arama uzayı.
    Küçük veri → sığ/regularize, büyük veri → derin/kapasiteli.

    small bucket değişiklikleri (overfit düzeltmesi v2):
      - depth  : (3,6) → (3,4)    — derinlik daha da kısıtlandı, ezber azalır
      - iter   : (400,1200) → (300,600) — daha düşük üst sınır, early_stopping
                                          ile birlikte kapasiteyi sınırlar
      - lr     : (0.02,0.12) → (0.02,0.08) — düşük depth ile dengeyi korur
      - l2     : (5.0,20.0) → (10.0,30.0)  — regularizasyon tabanı yükseltildi
    """
    if n_rows < 5_000:
        return dict(iter=(200, 600, 100), depth=(3, 5), lr=(0.02, 0.15), l2=(5.0, 15.0), bag=(0.1, 0.5))
    if n_rows < 30_000:
        return dict(iter=(300, 600, 50), depth=(3, 4), lr=(0.02, 0.08), l2=(10.0, 30.0), bag=(0.2, 0.6))
    if n_rows < 100_000:
        return dict(iter=(600, 1500, 100), depth=(5, 7), lr=(0.005, 0.08), l2=(2.0, 10.0), bag=(0.1, 0.6))
    if n_rows < 500_000:
        return dict(iter=(800, 2000, 100), depth=(6, 8), lr=(0.003, 0.05), l2=(0.5, 6.0),  bag=(0.1, 0.7))
    return dict(iter=(1000, 3000, 200), depth=(7, 9), lr=(0.001, 0.03), l2=(0.1, 4.0),  bag=(0.1, 0.8))


def _bucket_name(n_rows: int) -> str:
    """Veri boyutuna göre JSON bucket ismi."""
    if n_rows < 5_000:    return "xs"
    if n_rows < 30_000:   return "small"
    if n_rows < 100_000:  return "medium"
    if n_rows < 500_000:  return "large"
    return "xlarge"


# ---------------------------------------------------------------------------
# Veri yükleme (run_forecast.py ile aynı mantık)
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # run_forecast.py ile aynı isim bazlı eşleştirme mantığı
    _KAYNAK_CANDIDATES = ["Çıkış Transfer Merkezi", "kaynak_tm", "source", "origin", "from"]
    _VARIS_CANDIDATES  = ["Varış Transfer Merkezi",  "varis_tm",  "destination", "dest", "to"]
    _DATE_CANDIDATES   = ["Tarih", "tarih", "date", "Date"]
    _TARGET_CANDIDATES = ["Toplam Desi", "desi_hacmi", "desi", "demand", "talep"]

    def _find_col(candidates, col_idx, label):
        cols_lower = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        fallback = df.columns[col_idx]
        logger.warning(f"⚠️  '{label}' isim bazlı bulunamadı → pozisyon [{col_idx}]: '{fallback}'")
        return fallback

    df = df.rename(columns={
        _find_col(_KAYNAK_CANDIDATES, 0, "kaynak_tm"): KAYNAK_COL,
        _find_col(_VARIS_CANDIDATES,  1, "varis_tm"):  VARIS_COL,
        _find_col(_DATE_CANDIDATES,   2, "tarih"):     DATE_COL,
        _find_col(_TARGET_CANDIDATES, 3, "desi_hacmi"): TARGET_COL,
    })
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[GROUP_COL] = df[KAYNAK_COL] + " → " + df[VARIS_COL]

    all_dates  = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    all_routes = df[GROUP_COL].unique()
    rota_map   = df[[GROUP_COL, KAYNAK_COL, VARIS_COL]].drop_duplicates()

    idx  = pd.MultiIndex.from_product([all_routes, all_dates], names=[GROUP_COL, DATE_COL])
    full = pd.DataFrame(index=idx).reset_index()
    full = full.merge(rota_map, on=GROUP_COL, how="left")
    full = full.merge(df[[GROUP_COL, DATE_COL, TARGET_COL]], on=[GROUP_COL, DATE_COL], how="left")
    full[TARGET_COL] = full[TARGET_COL].fillna(0.0)
    full = full.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)
    logger.info(f"✅ Veri: {len(df):,} kayıt | {full[GROUP_COL].nunique()} rota")
    return full


# ---------------------------------------------------------------------------
# Optuna Optimizasyonu
# ---------------------------------------------------------------------------

def optimize(
    data_path: str,
    n_trials: int = 30,
    timeout: int  = 180,
) -> dict:
    """
    q50 WAPE'yi minimize eden hiperparametreleri bul.
    q50 üzerinde optimize → q10/q90 aynı yapıyı kullanır.
    Walk-forward split: son %20 validation (anormal haftalar dışlanır).
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("🔍 Optuna Hiperparametre Optimizasyonu")
    logger.info("=" * 60)

    # Veri
    full_df = load_data(data_path)
    n_rows  = len(full_df[full_df[TARGET_COL] > 0])  # sadece gerçek kayıtlar
    bucket  = _bucket_name(n_rows)
    space   = _search_space(n_rows)

    logger.info(f"   Gerçek kayıt: {n_rows:,} | Bucket: {bucket}")

    # Feature engineering
    feat_df = build_feature_matrix(
        df             = full_df,
        target_column  = TARGET_COL,
        date_column    = DATE_COL,
        group_column   = GROUP_COL,
        lags           = [1, 7, 14, 21],  # lag_2,3 negatif otokor. | lag_30 kaldırıldı → +801 satır eğitim
        rolling_windows= [7, 14],
        drop_na        = True,
    )

    # Anormal haftaları tespit et (tatil/birikim → validation'dan dışla)
    weekly_avg  = full_df[full_df[TARGET_COL] > 0].copy()
    weekly_avg["week"] = weekly_avg[DATE_COL].dt.isocalendar().week.astype(int)
    weekly_avg["year"] = weekly_avg[DATE_COL].dt.year
    wk_means    = weekly_avg.groupby(["year", "week"])[TARGET_COL].mean()
    threshold   = wk_means.mean() * 1.4
    abnormal_wk = set(wk_means[wk_means > threshold].index)
    if abnormal_wk:
        logger.info(f"⚠️  Anormal haftalar dışlandı: {abnormal_wk}")

    # Walk-forward split: son %20 validation — tarih bazlı (TM_ID sızıntısı yok)
    unique_dates = feat_df[DATE_COL].sort_values().unique()
    split_date   = unique_dates[min(int(len(unique_dates) * 0.80), len(unique_dates) - 1)]
    train_df     = feat_df[feat_df[DATE_COL] <  split_date]
    val_df       = feat_df[feat_df[DATE_COL] >= split_date]

    # Sadece hedef ve tarih düşürülür; rota/kaynak_tm/varis_tm cat_features olarak modele girer.
    drop_cols    = [TARGET_COL, DATE_COL]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    cat_features = get_categorical_columns(train_df[feature_cols])

    # Validation: anormal haftaları WAPE'den çıkar
    val_df = val_df.copy()
    val_df["_week"] = val_df[DATE_COL].dt.isocalendar().week.astype(int)
    val_df["_year"] = val_df[DATE_COL].dt.year
    normal_val = val_df[~val_df.apply(
        lambda r: (r["_year"], r["_week"]) in abnormal_wk, axis=1
    )]

    X_tr  = train_df[feature_cols]
    y_tr  = train_df[TARGET_COL]
    X_val = normal_val[feature_cols]
    y_val = normal_val[TARGET_COL].values

    logger.info(
        f"\n📦 Train: {len(X_tr):,} | Validation: {len(X_val):,} (normal günler)\n"
        f"   {len(feature_cols)} feature | Kategorik: {cat_features}\n"
        f"   {n_trials} trial, {timeout}s timeout"
    )

    # Objective
    def objective(trial):
        s = space
        params = {
            "iterations":          trial.suggest_int("iterations", s["iter"][0], s["iter"][1], step=s["iter"][2]),
            "depth":               trial.suggest_int("depth", s["depth"][0], s["depth"][1]),
            "learning_rate":       trial.suggest_float("learning_rate", s["lr"][0], s["lr"][1], log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", s["l2"][0], s["l2"][1]),
            "bagging_temperature": trial.suggest_float("bagging_temperature", s["bag"][0], s["bag"][1]),
            "loss_function":       "Quantile:alpha=0.5",
            "random_seed":         42,
            "verbose":             False,
            "allow_writing_files": False,
            "thread_count":        -1,
        }
        model = CatBoostRegressor(**params)
        model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features),
            eval_set=Pool(X_val, y_val, cat_features=cat_features),
            early_stopping_rounds=50,
        )
        # early_stopping'in gerçekte durduğu iteration'ı kaydet
        trial.set_user_attr("best_iteration", model.get_best_iteration())
        preds = np.maximum(model.predict(X_val), 0)
        wape  = float(np.sum(np.abs(y_val - preds)) / np.sum(y_val)) if np.sum(y_val) > 0 else 1.0
        return wape

    study = optuna.create_study(direction="minimize", study_name=f"rtopt_{bucket}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best    = study.best_params
    elapsed = time.time() - t0

    best_iter_actual = study.best_trial.user_attrs.get("best_iteration", best["iterations"])

    logger.info(
        f"\n✅ Optimizasyon tamamlandı ({elapsed/60:.1f} dakika)\n"
        f"   Best WAPE      : {study.best_value:.4%}\n"
        f"   iterations     : {best['iterations']} → erken durak: {best_iter_actual}\n"
        f"   depth          : {best['depth']}\n"
        f"   learning_rate  : {best['learning_rate']:.6f}\n"
        f"   l2_leaf_reg    : {best['l2_leaf_reg']:.3f}\n"
        f"   bagging_temp   : {best['bagging_temperature']:.3f}"
    )

    # JSON güncelle
    # best_iteration: early_stopping'in gerçekte durduğu adım.
    # Bu değer forecasters.py'de iterations olarak kullanılır — arama uzayındaki
    # ham sayı değil, validation'da en iyi performansı veren gerçek adım sayısı.
    result_entry = {
        "row_count":                n_rows,
        "best_wape":                round(study.best_value, 6),
        "optimization_time_minutes": round(elapsed / 60, 2),
        "params": {
            "iterations":          best_iter_actual,   # ← early_stopping gerçek dur noktası
            "depth":               best["depth"],
            "learning_rate":       best["learning_rate"],
            "l2_leaf_reg":         best["l2_leaf_reg"],
            "bagging_temperature": best["bagging_temperature"],
        }
    }

    # Mevcut JSON'u yükle, bucket'ı güncelle
    if HYPERPARAMS_PATH.exists():
        with open(HYPERPARAMS_PATH) as f:
            hmap = json.load(f)
    else:
        hmap = {}

    hmap[bucket] = result_entry

    HYPERPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HYPERPARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(hmap, f, indent=4, ensure_ascii=False)

    logger.info(f"\n💾 hyperparams_map.json güncellendi → bucket='{bucket}'")
    logger.info("=" * 60)

    return result_entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hiperparametre Optimizasyonu")
    parser.add_argument("--data",    default="data/Desi_talep.xlsx", help="Veri dosyası")
    parser.add_argument("--trials",  type=int, default=50,      help="Optuna trial sayısı")
    parser.add_argument("--timeout", type=int, default=900,     help="Timeout (saniye) — 0 = sınırsız (sadece trials sayısı geçerli)")
    args = parser.parse_args()

    timeout = args.timeout if args.timeout > 0 else None
    result = optimize(args.data, n_trials=args.trials, timeout=timeout)
    print(f"\nJSON'a yazılan parametreler:\n{json.dumps(result, indent=2)}")
