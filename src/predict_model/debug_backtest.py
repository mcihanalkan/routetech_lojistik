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
import json
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
from src.metrics import decision_regret, wape

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
    p.add_argument(
        "--surge-cap-alpha", type=float, default=None,
        help="Faz 2 Relative Cap: surge correction'ı baseline (düzeltme öncesi) "
             "hacmin bu oranıyla sınırla (ör. 0.20, 0.25, 0.30, 0.40). "
             "Belirtilmezse retrain'de kaydedilmiş değer (varsayılan None=kapalı) kullanılır. "
             "Retrain gerekmez — fc.surge_relative_cap_alpha_ çalışma zamanında elle atanır. "
             "Her iki hedefe (09:00 + 17:00) birden uygulanır.",
    )
    p.add_argument(
        "--surge-segment-scale", type=str, default=None,
        help="Faz 2b Segment Scale (SADECE 09:00 hedefine uygulanır — 09:00'daki "
             "overprediction deseni monoton değil, orta hacimde tetikleniyor; "
             "17:00 için --surge-cap-alpha yeterli). JSON string olarak "
             "[[low, high, scale], ...] listesi verin, ör: "
             "'[[0,60,1.0],[60,1200,0.4],[1200,999999,1.0]]'. "
             "Retrain gerekmez — fc.surge_segment_scale_ çalışma zamanında elle atanır.",
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


def run_one_target(label, target_col, model_path, full_df, cutoff, start, end,
                    surge_cap_alpha=None, surge_segment_scale=None):
    fc = DemandForecaster.load_model(model_path)
    fc.surge_calibration_factor_ = 1.5
    if surge_cap_alpha is not None:
        fc.surge_relative_cap_alpha_ = surge_cap_alpha
        log.info(f"ℹ️  [{label}] Faz 2 Relative Cap aktif: alpha={surge_cap_alpha}")
    if surge_segment_scale is not None:
        fc.surge_segment_scale_ = surge_segment_scale
        log.info(f"ℹ️  [{label}] Faz 2b Segment Scale aktif: segmentler={surge_segment_scale}")

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

    # --- Teşhis: Gerçek vs Tahmin — İzlenen (şüpheli) Rotalar ---
    # run_forecast.py'deki lag_1 çöküş şüphesiyle aynı üç rota; burada
    # backtest penceresindeki gerçek y_true/q50/q90 karşılaştırması ile
    # aynı desenin (sert uçurum vs kademeli düşüş) burada da olup
    # olmadığına bakıyoruz.
    _izlenen_rotalar = ["Yalova → İstanbul", "Kocaeli → İstanbul", "İstanbul → Yalova"]
    print(f"\n🔬 [{label}] Gerçek vs Tahmin — İzlenen Rotalar:")
    print(
        merged[merged[GROUP_COL].isin(_izlenen_rotalar)]
        .assign(hata_pct=lambda d: (d["q50"] - d["y_true"]) / d["y_true"].replace(0, np.nan) * 100)
        [[GROUP_COL, "tarih", "y_true", "q50", "q90", "hata_pct"]]
        .sort_values([GROUP_COL, "tarih"])
        .to_string(index=False)
    )

    # --- Faz 1 Teşhis: Volume vs Overprediction (Adım 5 öncesi) ---
    diag = merged[merged["y_true"] >= 20].copy()   # çok küçük gerçek değerleri filtrele (% patlamasını önler)
    diag["overpred_pct"] = (diag["q50"] - diag["y_true"]) / diag["y_true"] * 100
    diag["abs_overpred"] = np.maximum(diag["q50"] - diag["y_true"], 0.0)
    diag["volume_decile"] = pd.qcut(diag["q50"], 10, labels=False, duplicates="drop")
    diag["weekday"] = pd.to_datetime(diag["tarih"]).dt.day_name()

    # --- Faz 1b: Baseline (düzeltme öncesi, q50_base) vs Düzeltilmiş (q50) ---
    # q50_base = surge/weekday düzeltmesinden ÖNCEKİ ham Model-1 çıktısı
    # (forecasters.py::predict() içinde eklendi). Bu, overprediction'ın
    # ne kadarının Model-1'in kendisinden, ne kadarının düzeltme
    # katmanlarından (Model 2 + weekday bias) geldiğini ayrıştırır.
    if "q50_base" in diag.columns:
        diag["base_overpred_pct"] = (diag["q50_base"] - diag["y_true"]) / diag["y_true"] * 100
        diag["correction_contrib"] = diag["q50"] - diag["q50_base"]   # Model 2'nin eklediği miktar

    print(f"\n📊 [{label}] Hacim-Bias Teşhisi (düzeltilmiş, y_true>=20 filtreli):")
    agg_spec = dict(
        n=("q50", "size"),
        ort_hacim=("q50", "mean"),
        medyan_overpred_pct=("overpred_pct", "median"),   # ortalama yerine medyan — outlier'a dayanıklı
        toplam_abs_overpred=("abs_overpred", "sum"),        # gerçek desi cinsinden boyut
    )
    if "q50_base" in diag.columns:
        agg_spec["medyan_base_overpred_pct"] = ("base_overpred_pct", "median")
        agg_spec["ort_correction_contrib"] = ("correction_contrib", "mean")
    decile_stats = diag.groupby("volume_decile").agg(**agg_spec).round(1)
    print(decile_stats.to_string())

    # --- Faz 1e: Rota Bazlı WAPE + Ufuk (Horizon) Bazlı Hata Büyümesi ---
    merged["horizon"] = (pd.to_datetime(merged["tarih"]) - pd.Timestamp(cutoff)).dt.days

    # (A) Ufuk ilerledikçe WAPE gerçekten büyüyor mu? (PDF'in "hata birikimi" iddiasının testi)
    horizon_stats = merged.groupby("horizon").apply(
        lambda g: pd.Series({
            "n": len(g),
            "toplam_hacim": g["y_true"].sum(),
            "WAPE": wape(g["y_true"].values, g["q50"].values),
            "decision_regret": decision_regret(g["y_true"].values, g["q50"].values, spot_multiplier=9.0),
        })
    ).round(4)
    print(f"\n📈 [{label}] Ufuk (Horizon) Bazlı WAPE / Regret Büyümesi:")
    print(horizon_stats.to_string())

    # (B) Rota bazlı WAPE — düşük hacimli gürültü ile gerçek sorunlu hatları ayırt et
    route_stats = merged.groupby(GROUP_COL).apply(
        lambda g: pd.Series({
            "n": len(g),
            "ort_hacim": g["y_true"].mean(),
            "toplam_hacim": g["y_true"].sum(),
            "WAPE": wape(g["y_true"].values, g["q50"].values),
            "decision_regret": decision_regret(g["y_true"].values, g["q50"].values, spot_multiplier=9.0),
        })
    ).round(3)

    # Sadece anlamlı hacimli rotalara odaklan (örn. ort_hacim >= 50) — aksi halde
    # WAPE zaten payda küçük olduğu için yapısal olarak şişer, teşhisi bozar.
    route_stats_relevant = route_stats[route_stats["ort_hacim"] >= 50].sort_values("decision_regret", ascending=False)
    print(f"\n🚩 [{label}] En Kötü 20 Rota (ort_hacim>=50, decision_regret'e göre sıralı):")
    print(route_stats_relevant.head(20).to_string())

    print(f"\n📊 [{label}] Rota Segmentasyonu Özeti:")
    print(f"   Toplam rota sayısı           : {len(route_stats)}")
    print(f"   ort_hacim>=50 rota sayısı    : {len(route_stats_relevant)}")
    print(f"   Bunların medyan WAPE'i       : {route_stats_relevant['WAPE'].median():.3f}")
    print(f"   Bunların WAPE>0.5 olan sayısı: {(route_stats_relevant['WAPE'] > 0.5).sum()}")

    # --- Faz 1c: Orta Hacim Bandı (60-1200) — Hafta Günü Kırılımı ---
    # Decile'lar hacim ekseninde, bu ise hacmi 60-1200 bandına sabitleyip
    # hatanın hafta içinde HANGİ güne yoğunlaştığını gösterir (ör. Salı
    # düşüşü gibi day-2 etkilerini decile ortalamasının maskeleyebileceği
    # durumlar için).
    orta_hacim = diag[(diag["q50"] >= 60) & (diag["q50"] < 1200)].copy()
    # abs_overpred'in aynası, ters yönde: eksik tahmin edilen miktar (gerçek desi cinsinden).
    orta_hacim["abs_eksik"] = np.maximum(orta_hacim["y_true"] - orta_hacim["q50"], 0.0)

    weekday_stats = orta_hacim.groupby("weekday").agg(
        n=("q50", "size"),
        medyan_overpred_pct=("overpred_pct", "median"),
        n_eksik=("overpred_pct", lambda s: (s < 0).sum()),
        n_fazla=("overpred_pct", lambda s: (s >= 0).sum()),
        toplam_abs_overpred=("abs_overpred", "sum"),
        toplam_abs_eksik=("abs_eksik", "sum"),
    ).round(1)

    # decision_regret, y_true VE q50'yi BİRLİKTE gerektirdiği için tek sütun
    # bazlı .agg() içine sığmıyor — her weekday grubu için ayrı groupby.apply()
    # ile hesaplanıp weekday_stats'a sütun olarak eklenir. Bu, o gün/banttaki
    # GERÇEK spot-araç maliyetini (9x asimetrik ceza dahil) yansıtır —
    # toplam_abs_overpred/eksik gibi ham hacim farkları değil.
    weekday_stats["decision_regret"] = orta_hacim.groupby("weekday").apply(
        lambda g: decision_regret(g["y_true"].values, g["q50"].values, spot_multiplier=9.0)
    ).round(2)

    weekday_stats = weekday_stats.reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    print(f"\n🗓️  [{label}] Orta Hacim Bandı (60-1200) — Hafta Günü Kırılımı:")
    print(weekday_stats.to_string())

    # --- Faz 1d: Orta Hacim Bandı — Model 2 Correction Büyüklüğüne Göre Kırılım ---
    # weekday'deki 7 kategoriden farklı olarak correction_contrib sürekli bir
    # değişken, bu yüzden decile yerine 5 bine ayırıyoruz (örnek sayısı az).
    if "correction_contrib" in orta_hacim.columns:
        orta_hacim["correction_bin"] = pd.qcut(
            orta_hacim["correction_contrib"], 5, labels=False, duplicates="drop"
        )

        correction_stats = orta_hacim.groupby("correction_bin").agg(
            n=("q50", "size"),
            ort_correction=("correction_contrib", "mean"),
            medyan_overpred_pct=("overpred_pct", "median"),
            n_eksik=("overpred_pct", lambda s: (s < 0).sum()),
            n_fazla=("overpred_pct", lambda s: (s >= 0).sum()),
            toplam_abs_eksik=("abs_eksik", "sum"),
            toplam_abs_overpred=("abs_overpred", "sum"),
        ).round(1)
        correction_stats["decision_regret"] = (
            (9 * correction_stats["toplam_abs_eksik"] + correction_stats["toplam_abs_overpred"])
            / correction_stats["n"]
        ).round(2)

        print(f"\n📐 [{label}] Orta Hacim Bandı — Model 2 Correction Büyüklüğüne Göre Kırılım:")
        print(correction_stats.to_string())

        bin0 = orta_hacim[orta_hacim["correction_bin"] == 0].copy()
        bin0_weekday = bin0.groupby("weekday").agg(
            n=("q50", "size"),
            n_eksik=("overpred_pct", lambda s: (s < 0).sum()),
            toplam_abs_eksik=("abs_eksik", "sum"),
            ort_correction=("correction_contrib", "mean"),
        ).round(1)
        bin0_weekday = bin0_weekday.reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        print(f"\n🎯 [{label}] Bin 0 (En Küçük Correction) — Hafta Günü Dağılımı:")
        print(bin0_weekday.to_string())

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

    # --surge-segment-scale SADECE 09:00 hedefine uygulanır (bkz. --help metni):
    # 09:00'daki overprediction deseni monoton değil, orta hacim aralığında
    # tetikleniyor — 17:00'da desen monoton olduğu için --surge-cap-alpha
    # (her iki hedefe de uygulanan, aşağıdaki döngü) yeterli.
    surge_segment_scale = None
    if args.surge_segment_scale is not None:
        try:
            parsed = json.loads(args.surge_segment_scale)
            surge_segment_scale = [tuple(seg) for seg in parsed]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise RuntimeError(
                f"❌ --surge-segment-scale JSON olarak parse edilemedi: {e}\n"
                f"   Beklenen format: '[[low, high, scale], ...]', ör. "
                f"'[[0,60,1.0],[60,1200,0.4],[1200,999999,1.0]]'"
            )

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
            surge_cap_alpha=args.surge_cap_alpha,
            # segment_scale sadece 09:00 için — 17:00 çağrısında None kalır
            surge_segment_scale=surge_segment_scale if label == "09:00" else None,
        )


if __name__ == "__main__":
    main()