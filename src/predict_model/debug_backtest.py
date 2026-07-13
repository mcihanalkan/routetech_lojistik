"""
debug_backtest.py — predict_sequential() ile gerçek bir pencereyi "tahmin
edermiş gibi" çalıştırıp gerçek değerlerle karşılaştırır.

ZEMİN GERÇEĞİ (forecasters.py::DemandForecaster.fit() içinden BİREBİR alındı,
tahmin edilmedi):

    FOLD_DATES = [
        ("Fold 1", "2026-05-31", "2026-06-06"),
        ("Fold 2", "2026-06-07", "2026-06-13"),
        ("Fold 3", "2026-06-14", "2026-06-20"),
        ("Fold 4", "2026-06-21", "2026-06-27"),
    ]
    Her fold: train = df[date_column < val_start] (walk-forward, KÜMÜLATİF).
    Yani Fold 2 train'i Fold 1'in val haftasını da İÇERİR, Fold 3 train'i
    Fold 1+2'nin val haftalarını içerir, Fold 4 train'i Fold 1+2+3'ünkileri
    içerir. use_best_model=False olduğu için eval_set (val) ağırlıklara
    KARIŞMAZ — yani bir fold'un KENDİ val haftası o fold'un ağırlıkları için
    gerçekten görülmemiştir, ama SONRAKİ fold'ların TRAIN'ine girer.

    SONUÇ: Veri setindeki (2026-01-01 → 2026-06-28) hemen hemen her tarih,
    ensemble'daki en az bir fold modelinin (özellikle Fold 4'ün) eğitim
    setinde. Ensemble genelinde (4 fold'un HİÇBİRİNİN ağırlıkça görmediği)
    TEK temiz pencere:

        2026-06-21 → 2026-06-27   (Fold 4'ün kendi val haftası)
        2026-06-28                (veride var ama hiçbir fold'a train/val
                                    olarak hiç girmemiş — en temiz tek gün)

    ⚠️  BUNUN DA BİR SINIRI VAR: 06-21→06-27 penceresi zaten fit() içinde
    self-evaluation (Test WAPE / Decision Regret) olarak raporlanıyor. Bu
    script'i bu pencerede çalıştırmak size 30 Haziran'daki day-2/Salı
    düşüşü hakkında YENİ bir genelleme sinyali VERMEZ — sadece bu backtest
    pipeline'ının fit()-zamanı metrikleriyle TUTARLI sonuç üretip
    üretmediğini doğrular (yani bir "pipeline sanity check", bir
    "genelleme testi" değil).
    GERÇEKTEN yeni/temiz bir pencerede test etmek için iki gerçek seçenek:
      (A) Üretim modelinden BAĞIMSIZ, ayrı bir "holdout" DemandForecaster
          eğitin; bu modelin FOLD_DATES'ini erkene çekip 06-21→06-28'i (ya da
          30 Haziran'a benzer bir Salı içeren bir pencereyi) HİÇBİR fold'a
          train/val olarak vermeyin. Sadece diagnostik amaçlı, üretime
          çıkmaz.
      (B) 06-28'den sonrasının gerçek verisi elinize geçtikçe (canlı/gerçek
          desi kayıtları), o günleri bu script ile karşılaştırın — bu,
          modelin hâlâ hiç görmediği tek gerçek gelecek.
    Bu script şu an sadece (mevcut sınır dahilinde) pipeline'ı doğrulamak
    için 06-21→06-27 penceresini varsayılan olarak kullanır ve yukarıdaki
    sınırı her çalıştırmada açıkça hatırlatır.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_forecast import (
    load_dataset, DATA_PATH, DATE_COL, GROUP_COL,
    TARGET_COL_0900, TARGET_COL_1700,
    MODEL_FILE_PATH_0900, MODEL_FILE_PATH_1700,
)
from src.forecasters import DemandForecaster
from src.metrics import decision_regret

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("debug_backtest")

# ⚠️  forecasters.py::fit() içindeki fold_dates ile BİREBİR AYNI TUTULMALI.
# fold_dates orada self'e kaydedilmiyor (yalnızca fit() içinde local), bu
# yüzden yüklenen modelden introspect EDİLEMİYOR — burada elle senkron
# tutmak zorundayız. fit() değişirse burayı da güncelleyin.
FOLD_DATES = [
    ("Fold 1", "2026-05-31", "2026-06-06"),
    ("Fold 2", "2026-06-07", "2026-06-13"),
    ("Fold 3", "2026-06-14", "2026-06-20"),
    ("Fold 4", "2026-06-21", "2026-06-27"),
]

# Ensemble genelinde HİÇBİR fold'un ağırlıkça görmediği pencerenin başlangıcı
# = en geç fold'un val_start'ı (bkz. modül docstring'i).
_LAST_FOLD_VAL_START = max(pd.Timestamp(vs) for _, vs, _ in FOLD_DATES)

DEFAULT_START  = str(_LAST_FOLD_VAL_START.date())        # 2026-06-21
DEFAULT_END    = FOLD_DATES[-1][2]                        # 2026-06-27
DEFAULT_CUTOFF = str((_LAST_FOLD_VAL_START - pd.Timedelta(days=1)).date())  # 2026-06-20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="Context buffer'ın kesileceği tarih")
    p.add_argument("--start", default=DEFAULT_START, help="Backtest penceresi başlangıcı")
    p.add_argument("--end", default=DEFAULT_END, help="Backtest penceresi bitişi")
    p.add_argument(
        "--force", action="store_true",
        help="Pencere en az bir fold'un TRAIN setiyle çakışsa bile devam et (RİSKLİ — sonuç sızıntılı olur)",
    )
    p.add_argument(
        "--model-0900", default=None,
        help="09:00 modeli için MODEL_FILE_PATH_0900 yerine kullanılacak .joblib yolu "
             "(ör. ablate_payday.py çıktısı — production model dosyasıyla A/B karşılaştırma için)",
    )
    p.add_argument(
        "--model-1700", default=None,
        help="17:00 modeli için MODEL_FILE_PATH_1700 yerine kullanılacak .joblib yolu",
    )
    return p.parse_args()


def guard_no_leakage(start: str, end: str, data_max_date, force: bool):
    """
    Test penceresinin ensemble'daki HERHANGİ bir fold'un TRAIN setiyle
    (ağırlıkça görülmüş veri) çakışıp çakışmadığını FOLD_DATES üzerinden
    kesin olarak hesaplar (bkz. modül docstring'i — walk-forward kümülatif
    train mantığı). Çakışma varsa hata fırlatıp durdurur.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    if start_ts < _LAST_FOLD_VAL_START:
        msg = (
            f"test penceresi {start_ts.date()}'da başlıyor ama en geç fold "
            f"(Fold 4) {_LAST_FOLD_VAL_START.date()}'dan ÖNCEKİ HER TARİHİ "
            f"eğitiminde görmüş (walk-forward kümülatif train — bkz. modül "
            f"docstring'i). Bu pencere sızıntılı olur.\n"
            f"   Ensemble genelinde temiz olan TEK aralık: "
            f"{_LAST_FOLD_VAL_START.date()} → veri sonu."
        )
        if not force:
            raise RuntimeError("❌ DATA LEAKAGE: " + msg + "\n   Yine de devam etmek için (bilerek, sızıntılı): --force")
        log.warning(
            f"⚠️  --force verildi: {start_ts.date()} penceresi en az bir fold'un TRAIN "
            f"setiyle çakışıyor. Sonuçlar in-sample sızıntılı — gerçek genelleme hatası DEĞİL."
        )

    if end_ts > data_max_date:
        raise RuntimeError(
            f"❌ Test penceresi {end_ts.date()}'a kadar gidiyor ama veri sadece "
            f"{data_max_date.date()}'a kadar var — bu aralıkta karşılaştırılacak "
            f"gerçek değer YOK (gerçek=0 görünür, bu da yanıltıcı büyük bir "
            f"'fark %' üretir). --end değerini {data_max_date.date()} veya öncesine çekin."
        )

    if not force and start_ts == _LAST_FOLD_VAL_START and end_ts <= pd.Timestamp(FOLD_DATES[-1][2]):
        log.warning(
            "ℹ️  NOT: Bu pencere (Fold 4'ün val haftası) zaten fit() içinde "
            "self-evaluation olarak raporlanıyor. Bu backtest yeni bir "
            "genelleme sinyali değil, pipeline sanity-check'idir — bkz. modül "
            "docstring'indeki (A)/(B) seçenekleri gerçek out-of-sample test için."
        )


def build_prediction_frame(full_df, target_dates):
    all_routes = full_df[GROUP_COL].unique()
    rows = []
    for route in all_routes:
        info = full_df[full_df[GROUP_COL] == route][["kaynak_tm", "varis_tm"]].iloc[0]
        for d in target_dates:
            rows.append({
                GROUP_COL: route, DATE_COL: d,
                "kaynak_tm": info["kaynak_tm"], "varis_tm": info["varis_tm"],
                TARGET_COL_0900: np.nan, TARGET_COL_1700: np.nan,
            })
    pred_df = pd.DataFrame(rows)
    for c in (TARGET_COL_0900, TARGET_COL_1700):
        pred_df[c] = pred_df[c].fillna(0.0)
    return pred_df


def run_one_target(label, target_col, model_path, full_df, cutoff, start, end):
    fc = DemandForecaster.load_model(model_path)
    fc.surge_calibration_factor_ = 1.5

    # Context buffer'ı CUTOFF'a göre GEÇİCİ olarak yeniden hesapla
    # (fit-zamanı buffer'ı değiştirmiyoruz, sadece bu backtest için)
    hist_df = full_df[full_df[DATE_COL] <= pd.Timestamp(cutoff)].copy()
    fc._save_context_buffer(hist_df)

    target_dates = pd.date_range(start, end, freq="D")
    pred_df = build_prediction_frame(full_df, target_dates)

    preds = fc.predict_sequential(pred_df)
    preds_df = pd.DataFrame(preds).assign(tarih=lambda d: d["tarih"].astype(str).str[:10])

    pred_daily = preds_df.groupby("tarih")["q50"].sum()

    actual_daily = (
        full_df[(full_df[DATE_COL] >= start) & (full_df[DATE_COL] <= end)]
        .assign(tarih=lambda d: d[DATE_COL].astype(str).str[:10])
        .groupby("tarih")[target_col].sum()
    )

    # v16: decision_regret — MAPE değil, asıl önemsediğimiz metrik bu.
    # MAPE eksik/fazla tahmini simetrik cezalandırır; decision_regret ise
    # spot_multiplier=9.0 ile eksik tahmini 9x ağırlıklandırır (metrics.py
    # ile aynı hesap, optimize.py'nin raporladığı sayıyla karşılaştırılabilir
    # olsun diye). Rota-gün granülerliğinde hesaplanır (günlük TOPLAM üzerinden
    # değil) çünkü decision_regret'in asıl anlamı rota bazlı spot-araç kararı —
    # günlük toplamda hesaplarsak rotalar arası +/- hatalar birbirini
    # götürüp gerçek regret'i olduğundan düşük gösterir.
    actual_route_day = (
        full_df[(full_df[DATE_COL] >= start) & (full_df[DATE_COL] <= end)]
        .assign(tarih=lambda d: d[DATE_COL].astype(str).str[:10])
        [[GROUP_COL, "tarih", target_col]]
        .rename(columns={target_col: "y_true"})
    )
    merged = preds_df.merge(actual_route_day, on=[GROUP_COL, "tarih"], how="inner")

    print(f"\n🔬 [{label}] Backtest {start}→{end} (cutoff={cutoff}) — Tahmin vs Gerçek:")
    abs_pct_errors = []
    daily_regret_q50 = {}
    for d in sorted(set(pred_daily.index) | set(actual_daily.index)):
        p = pred_daily.get(d, 0.0)
        a = actual_daily.get(d, 0.0)
        wd = pd.Timestamp(d).day_name()
        diff_pct = (p - a) / a * 100 if a > 0 else float("nan")
        if a > 0:
            abs_pct_errors.append(abs(diff_pct))
        day_rows = merged[merged["tarih"] == d]
        regret_d = decision_regret(day_rows["y_true"].values, day_rows["q50"].values, spot_multiplier=9.0) if len(day_rows) else float("nan")
        daily_regret_q50[d] = regret_d
        print(f"   {d} ({wd:<9}) | tahmin={p:>12,.0f} | gerçek={a:>12,.0f} | fark={diff_pct:+6.1f}% | regret(q50)={regret_d:>8.2f}")

    if abs_pct_errors:
        print(f"   → Ortalama mutlak yüzde hata (MAPE): {np.mean(abs_pct_errors):.1f}%")

    if len(merged):
        regret_q50_week = decision_regret(merged["y_true"].values, merged["q50"].values, spot_multiplier=9.0)
        regret_q90_week = decision_regret(merged["y_true"].values, merged["q90"].values, spot_multiplier=9.0)
        print(f"   → Decision Regret (q50, rota-gün, hafta ort.) : {regret_q50_week:.2f}")
        print(f"   → Decision Regret (q90, rota-gün, hafta ort.) : {regret_q90_week:.2f}  (güvenli/üretim stratejisi)")




def main():
    args = parse_args()
    full_df = load_dataset(DATA_PATH)
    data_max_date = pd.Timestamp(full_df[DATE_COL].max())

    guard_no_leakage(args.start, args.end, data_max_date, args.force)

    for label, target_col, default_model_path, override in [
        ("09:00", TARGET_COL_0900, MODEL_FILE_PATH_0900, args.model_0900),
        ("17:00", TARGET_COL_1700, MODEL_FILE_PATH_1700, args.model_1700),
    ]:
        model_path = override or default_model_path
        if override:
            log.info(f"ℹ️  [{label}] Model override kullanılıyor: {model_path}")
        run_one_target(
            label, target_col, model_path, full_df,
            cutoff=args.cutoff, start=args.start, end=args.end,
        )


if __name__ == "__main__":
    main()