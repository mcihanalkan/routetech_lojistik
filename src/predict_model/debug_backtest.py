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
from catboost import Pool

sys.path.insert(0, str(Path(__file__).parent))

from run_forecast import (
    load_dataset, DATA_PATH, DATE_COL, GROUP_COL,
    TARGET_COL_0900, TARGET_COL_1700,
    MODEL_FILE_PATH_0900, MODEL_FILE_PATH_1700,
)
from src.forecasters import DemandForecaster
from src.forecasters import (
    SURGE_BINARY_TRIGGER_COLUMNS,
    SURGE_CONTINUOUS_TRIGGER_COLUMNS,
    BUCKET_EVENT_CONTINUOUS_THRESHOLD,
    suggest_bucket_event_threshold,
)
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
        "--surge-vol-damping-k", type=float, default=None,
        help="Hacim Tabanlı Lojistik Sönümleme (S_vol) eğim parametresi k. "
             "S_vol = 2/(1+exp(-k·V/v_crit))-1 ≡ tanh(k·V/(2·v_crit)) olduğundan "
             "%50 bastırma noktası v_crit DEĞİL, v_crit·ln(3)/k'dir — eski "
             "varsayılan k=3.0, v_crit=150 ile bu nokta ≈55 desi'ye düşüyordu "
             "(docstring'in varsaydığı 150 yerine). k=1.0986 (ln(3)) verirse "
             "v_crit GERÇEKTEN %50 noktası olur. Belirtilmezse modelin kayıtlı "
             "değeri (varsayılan 3.0) kullanılır. Retrain gerekmez — "
             "fc.surge_volume_damping_k_ çalışma zamanında elle atanır.",
    )
    p.add_argument(
        "--surge-calibration-factor", type=float, default=None,
        help="Surge/Residual (Model 2) pozitif düzeltmelerine uygulanan çarpan "
             "(bkz. forecasters.py::surge_calibration_factor_, satır ~2722: "
             "SADECE max(residual,0) kısmını çarpar). Belirtilmezse modelin "
             "KENDİ kayıtlı/production değeri kullanılır (varsayılan 1.0x) — "
             "yani override YAPILMAZ. Önceki sürümde bu script'te sabit 1.5x "
             "olarak (açıklamasız) hardcode edilmişti; şimdi opt-in.",
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
    p.add_argument(
        "--surge-predict-threshold", type=float, default=None,
        help="Tahmin-zamanına özel, DAHA SIKI sürekli-tetikleyici eşiği "
             "(SURGE_CONTINUOUS_TRIGGER_THRESHOLD=0.05 yerine). Kök neden: "
             "backlog_release_index her Pazar kapanışından sonra üretiliyor "
             "ve eşiğin üzerinde haftanın neredeyse tamamı boyunca kalabiliyor "
             "(bkz. BUCKET_EVENT_CONTINUOUS_THRESHOLD yorumu, forecasters.py) "
             "— Model 2 rutin haftalık kapanışları bile 'surge' sanıyor. "
             "Bu değer set edilirse SADECE predict()'te hangi satırlara "
             "correction uygulanacağı süzülür; Model 2'nin eğitimi/ağırlıkları "
             "etkilenmez, retrain gerekmez. Her iki hedefe (09:00 + 17:00) "
             "birden uygulanır. ⚠️ v18 UYARISI: features.py'nin backlog_"
             "severity_ratio/campaign_release_index entegrasyonundan sonra "
             "backlog_release_index/campaign_release_index artık ORAN bazlı "
             "(gün-sayısı × decay DEĞİL) — eski '0.35 ilk deneme' önerisi bu "
             "yeni ölçekte YANILTICIDIR, kullanmayın. Bunun yerine bu script'i "
             "çalıştırıp konsoldaki '📏 v18 Sonrası Eşik Kalibrasyon Raporu'nu "
             "(bkz. _print_bucket_threshold_calibration_diagnosis) okuyun ve "
             "oradaki 'normal' bucket'ı sıfırlamayan/'event'i haftanın tamamına "
             "eşitlemeyen bir eşik seçin.",
    )
    p.add_argument(
        "--censor-require-persistence", type=str, default=None,
        choices=["true", "false"],
        help="[Adım 1 — ucuz doğrulama testi] unconstrain_censored_demand() "
             "weekday-persistence gate'ini AÇ/KAPAT (bkz. features.py). "
             "'false' → v1 (eski) tek-seferlik tespit davranışına döner: "
             "aday bir gün SADECE kendi 14-günlük yerel tavanına değdiği için "
             "sansürlü sayılır (AI bulgusu: bu, güçlü haftalık mevsimselliği "
             "yanlışlıkla kapasite sansürü sanıp 189 rotayı 'FAZLA' tahmine "
             "sürüklüyordu). 'true' (varsayılan/model default'u) → v2: aynı "
             "hafta gününün son N tekrarından en azı persistence_min_hits "
             "kadarının da aday olması şartı aranır. Belirtilmezse modelin "
             "kendi kayıtlı/varsayılan değeri kullanılır. Retrain GEREKMEZ — "
             "sadece backtest-zamanı geriye-dönük feature hesaplamasını "
             "etkiler (predict_sequential() içindeki _engineer_features "
             "her fold'da yeniden çalışır).",
    )
    p.add_argument(
        "--censor-inflation-factor", type=float, default=None,
        help="[Adım 1 — ucuz doğrulama testi] Sansürlü işaretlenen günlere "
             "uygulanan şişirme çarpanı (varsayılan/model default'u 1.05). "
             "1.0 verilirse unconstraining fiilen TAMAMEN KAPANIR (hiçbir "
             "gün şişirilmez) — hipotezi en hızlı doğrulama yolu: bunu 1.0 "
             "yapıp 189 FAZLA / 17 EKSİK oranının ne kadar düştüğüne bakın.",
    )
    p.add_argument(
        "--censor-persistence-occurrences", type=int, default=None,
        help="[Kalıcı düzeltme kalibrasyonu] Weekday-persistence penceresi: "
             "aynı hafta gününün geriye dönük kaç tekrarına bakılacağı "
             "(varsayılan/model default'u 3, ör. 'son 3 Pazartesi').",
    )
    p.add_argument(
        "--censor-persistence-min-hits", type=int, default=None,
        help="[Kalıcı düzeltme kalibrasyonu] --censor-persistence-occurrences "
             "tekrarı içinde en az kaçının da 'aday' (tavana değmiş) olması "
             "gerektiği (varsayılan/model default'u 2, ör. '2/3').",
    )
    p.add_argument(
        "--censor-capacity-file", type=str, default=None,
        help="[Adım 2 — kalıcı düzeltme, GERÇEK kapasite verisiyle gate'leme] "
             "Ellecleme-kapasite.xlsx (veya aynı şemaya sahip herhangi bir "
             "dosya) yolu — 'transfer_merkezi' ve 'ellecleme_kapasite' "
             "sütunlarını bekler. Verilirse, unconstrain_censored_demand() "
             "istatistiksel proxy'yi (persistence gate) BIRAKIP her kaynak "
             "TM'nin GERÇEK günlük elleçleme kapasitesine göre karar verir "
             "(bkz. features.py Mantık 3.5). Verilmezse bu adım atlanır, "
             "sadece --censor-require-persistence ile kontrol edilen "
             "istatistiksel proxy kullanılır. Retrain GEREKMEZ.",
    )
    p.add_argument(
        "--censor-source-tm-column", type=str, default="kaynak_tm",
        help="--censor-capacity-file verildiğinde full_df içinde kaynak "
             "transfer merkezini tutan sütun adı (varsayılan: 'kaynak_tm', "
             "bkz. run_forecast.py::KAYNAK_COL).",
    )
    p.add_argument(
        "--censor-real-capacity-ratio", type=float, default=None,
        help="[Kalıcı düzeltme kalibrasyonu] TM×gün gerçekleşen toplam hacim "
             "/ TM'nin kayıtlı elleçleme kapasitesi bu oranı geçerse o gün "
             "gerçekten kapasiteye dayanmış sayılır (varsayılan/model "
             "default'u 0.90 → %90). Sadece --censor-capacity-file ile "
             "birlikte anlamlıdır.",
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


def _print_bucket_correction_diagnosis(fc, tracked_routes, label):
    """
    Yalova kümesi teşhisi (Adım 5) — SADECE OKUMA, model/fc üzerinde hiçbir
    değişiklik yapmaz, retrain gerektirmez.

    İki soruya cevap arar:
    1. Öneri A'nın (rota,bucket) düzeltmesi izlenen rotalar için GERÇEKTEN
       öğrenilmiş mi (yoksa eff_n yetersiz kalıp flat rota-bazlı/1.0'a mı
       düşüyor)? Öğrenilmişse, kendi bucket'ının cap sınırına (event için
       [0.5, 2.5], bkz. forecasters.py::_learn_route_bucket_bias_correction
       çağrısındaki sabit değerler) YAPIŞMIŞ mı — yani "elimizdeki en
       agresif düzeltme bile yetmiyor" durumu mu?
    2. backlog_release_index / campaign_release_index gerçekten
       predict_sequential() sırasında (sequential/gerçek üretim akışında)
       ufuk (h) ilerledikçe SÖNÜYOR mu — yani gerçek talep platosu ile
       feature'ın öngördüğü sönme eğrisi arasında bir uyumsuzluk var mı?
       Bu, fc.capture_debug_features_=True ile zaten toplanmış olan
       feat_backlog_release_index / feat_campaign_release_index
       sütunlarından (varsa) okunur — SHAP teşhisiyle AYNI kaynak veri,
       ek bir predict çağrısı YOK.
    """
    # --- 1. Öğrenilmiş (rota,bucket) oranları + flat fallback ---
    bucket_map = getattr(fc, "route_bucket_bias_correction_", None) or {}
    flat_map = getattr(fc, "route_bias_correction_", None) or {}
    # ⚠️ Bu cap değerleri forecasters.py::_learn_route_bucket_bias_correction
    # ÇAĞRISINDAKİ (fit() içi) sabit kwarg'lardan elle kopyalanmıştır — o
    # çağrı değişirse burası da güncellenmeli (introspect edilemiyor, aynı
    # FOLD_DATES senkron notu gibi).
    _CAPS = {"sunday": (0.30, 1.2), "event": (0.5, 2.5), "normal": (0.7, 1.8)}

    print(f"\n🔎 [{label}] Öneri A (rota,bucket) Düzeltme Teşhisi — {tracked_routes}:")
    for route in tracked_routes:
        for bucket in ("sunday", "event", "normal"):
            key = (route, bucket)
            if key in bucket_map:
                ratio = bucket_map[key]
                lo, hi = _CAPS[bucket]
                at_cap = ""
                if ratio >= hi - 1e-6:
                    at_cap = "  ⚠️ ÜST CAP'E YAPIŞMIŞ (daha fazlası mümkün değil, mevcut mekanizmayla)"
                elif ratio <= lo + 1e-6:
                    at_cap = "  ⚠️ ALT CAP'E YAPIŞMIŞ"
                print(f"   {route:22s} | bucket={bucket:7s} | öğrenilmiş çarpan={ratio:.3f}x{at_cap}")
            else:
                flat_val = flat_map.get(route)
                fallback_desc = f"flat rota-bazlı={flat_val:.3f}x" if flat_val is not None else "flat de yok → 1.0x (no-op)"
                print(
                    f"   {route:22s} | bucket={bucket:7s} | (rota,bucket) YOK "
                    f"(eff_n yetersiz kaldı) → {fallback_desc}"
                )

    # --- 2. backlog_release_index / campaign_release_index sönme eğrisi ---
    cap_rows = getattr(fc, "debug_captured_rows_", None)
    if not cap_rows:
        print(f"   ⚠️ [{label}] Sönme-eğrisi teşhisi atlandı: debug_captured_rows_ boş.")
        return
    cap_df = pd.DataFrame(cap_rows)
    cap_df = cap_df[cap_df[GROUP_COL].isin(tracked_routes)].copy()
    if cap_df.empty:
        return

    decay_cols = [
        c for c in ("feat_backlog_release_index", "feat_campaign_release_index", "feat_weekday")
        if c in cap_df.columns
    ]
    if not decay_cols:
        print(f"   ℹ️ [{label}] backlog_release_index/campaign_release_index yakalanan feature'larda yok "
              f"(feat_ önekli sütun adı beklenenden farklı olabilir).")
        return

    print(f"\n📉 [{label}] backlog/campaign_release_index — Ufuk (h) Boyunca Gerçek Sönme Eğrisi:")
    show_cols = [GROUP_COL, fc.date_column, "_h_idx"] + decay_cols
    show_cols = [c for c in show_cols if c in cap_df.columns]
    for route in tracked_routes:
        sub = cap_df[cap_df[GROUP_COL] == route].sort_values("_h_idx") if "_h_idx" in cap_df.columns else cap_df[cap_df[GROUP_COL] == route]
        if sub.empty:
            continue
        print(f"   -- {route} --")
        print(sub[show_cols].to_string(index=False))


def _print_oof_bucket_raw_diagnosis(fc, tracked_routes, label, half_life_days: float = 21.0):
    """
    Adım 6 — SALT OKUMA. _print_bucket_correction_diagnosis()'in ("öğrenilmiş
    çarpan=1.118x" gibi) NİHAİ sonucunu değil, o sonucun ARKASINDAKİ ham OOF
    hesaplamayı (fc._oof_X_/_oof_residual_/_oof_base_pred_/_oof_dates_) açar.
    forecasters.py::_learn_route_bucket_bias_correction() ile BİREBİR aynı
    eff_n / Laplace-smoothing / cap adımlarını burada tekrar hesaplayıp her
    birini ayrı satırda yazdırır — retrain yok, fc üzerinde hiçbir şey
    değiştirilmiyor.
    """
    X_oof = getattr(fc, "_oof_X_", None)
    if X_oof is None or len(X_oof) == 0 or fc.group_column not in X_oof.columns:
        print(f"   ⚠️ [{label}] OOF ham veri teşhisi atlandı: _oof_X_ boş/uygun değil.")
        return

    groups    = X_oof[fc.group_column].to_numpy()
    residual  = np.asarray(fc._oof_residual_, dtype=float)
    base_pred = np.asarray(fc._oof_base_pred_, dtype=float)
    oof_dates = np.asarray(getattr(fc, "_oof_dates_", None))
    if not (len(groups) == len(residual) == len(base_pred) == len(oof_dates)):
        print(f"   ⚠️ [{label}] OOF ham veri teşhisi atlandı: uzunluklar uyuşmuyor.")
        return

    y_true = residual + base_pred
    oof_dates_dt = pd.to_datetime(oof_dates)
    max_date = oof_dates_dt.max()
    weights = 0.5 ** (((max_date - oof_dates_dt).values / np.timedelta64(1, "D")) / half_life_days)

    weekday_vals = (
        pd.to_numeric(X_oof["weekday"], errors="coerce").fillna(-1).to_numpy()
        if "weekday" in X_oof.columns else np.full(len(X_oof), -1)
    )
    is_sunday = (weekday_vals == 6)
    event_mask = np.zeros(len(X_oof), dtype=bool)
    for col in SURGE_BINARY_TRIGGER_COLUMNS:
        if col in X_oof.columns:
            event_mask |= (pd.to_numeric(X_oof[col], errors="coerce").fillna(0.0) > 0).to_numpy()
    for col in SURGE_CONTINUOUS_TRIGGER_COLUMNS:
        if col in X_oof.columns:
            event_mask |= (
                pd.to_numeric(X_oof[col], errors="coerce").fillna(0.0) > BUCKET_EVENT_CONTINUOUS_THRESHOLD
            ).to_numpy()
    bucket = np.where(is_sunday, "sunday", np.where(event_mask, "event", "normal"))

    w_pred = weights * base_pred
    w_true = weights * y_true
    global_w_mean_pred = float(np.sum(w_pred) / np.sum(weights)) if np.sum(weights) > 0 else 0.0

    # ⚠️ fit() çağrısındaki (forecasters.py::_learn_route_bucket_bias_correction
    # varsayılan kwarg'ları) İLE BİREBİR SENKRON tutulmalı.
    MIN_EFF_N = {"sunday": 1.8, "event": 4.0, "normal": 4.0}
    CAP = {"sunday": (0.30, 1.2), "event": (0.5, 2.5), "normal": (0.7, 1.8)}
    SMOOTHING = {"sunday": 0.4, "event": 6.0, "normal": 6.0}

    print(f"\n🔬 [{label}] OOF Ham Veri Teşhisi (Adım 6) — global_w_mean_pred={global_w_mean_pred:,.2f}:")
    for route in tracked_routes:
        route_mask = (groups == route)
        if not route_mask.any():
            print(f"   {route:22s} | OOF'ta bu rota YOK")
            continue
        for b in ("sunday", "event", "normal"):
            combo_mask = route_mask & (bucket == b)
            n_rows = int(combo_mask.sum())
            if n_rows == 0:
                print(f"   {route:22s} | bucket={b:7s} | OOF satırı YOK")
                continue
            eff_n = float(weights[combo_mask].sum())
            sum_pred_c = float(w_pred[combo_mask].sum())
            sum_true_c = float(w_true[combo_mask].sum())
            sm = SMOOTHING[b]
            smoothed_pred = sum_pred_c + global_w_mean_pred * sm
            smoothed_true = sum_true_c + global_w_mean_pred * sm
            lo, hi = CAP[b]
            raw_ratio = smoothed_true / smoothed_pred if smoothed_pred > 0 else float("nan")
            ratio = max(lo, min(raw_ratio, hi)) if smoothed_pred > 0 else float("nan")
            ham_oran = (sum_true_c / sum_pred_c) if sum_pred_c > 0 else float("nan")
            print(
                f"   {route:22s} | bucket={b:7s} | n={n_rows:3d} | eff_n={eff_n:6.2f} "
                f"(min={MIN_EFF_N[b]}) | ham_oran(smoothing'siz)={ham_oran:.3f}x | "
                f"smoothing×{sm}×global_mean → raw_ratio(cap öncesi)={raw_ratio:.3f}x | "
                f"NİHAİ={ratio:.3f}x"
            )


def _print_bucket_threshold_calibration_diagnosis(fc, label):
    """
    v18 SONRASI YENİDEN KALİBRASYON TEŞHİSİ — SALT OKUMA, retrain yok.

    features.py'nin backlog_severity_ratio/campaign_severity_ratio
    entegrasyonundan (v18) sonra BUCKET_EVENT_CONTINUOUS_THRESHOLD (ve
    isteğe bağlı SURGE_CONTINUOUS_TRIGGER_THRESHOLD) ESKİ (v17, gün-sayısı
    × decay) ölçeğine göre kalibre edilmiş kalır — bkz. forecasters.py
    başındaki "v18 ÖLÇEK UYARISI" yorumu. Bu fonksiyon forecasters.py::
    suggest_bucket_event_threshold()'ı gerçek OOF verisi (fc._oof_X_)
    üzerinde çalıştırıp:
      1. backlog_release_index/campaign_release_index'in gerçek
         dağılımını (p50/p75/p90/p95/max),
      2. bir aday eşik grid'i için "sunday/event/normal" bucket
         dağılımını
    yazdırır — nihai BUCKET_EVENT_CONTINUOUS_THRESHOLD seçimi (forecasters.py
    içinde elle güncellenir) bu rapora bakılarak yapılır; bu fonksiyon
    otomatik atama yapmaz, sadece ölçer.
    """
    X_oof = getattr(fc, "_oof_X_", None)
    if X_oof is None or len(X_oof) == 0:
        print(f"   ⚠️ [{label}] Eşik kalibrasyon teşhisi atlandı: _oof_X_ boş/uygun değil.")
        return

    report = suggest_bucket_event_threshold(X_oof, columns=SURGE_CONTINUOUS_TRIGGER_COLUMNS)
    if not report["distribution"]:
        print(f"   ⚠️ [{label}] Eşik kalibrasyon teşhisi atlandı: sürekli tetikleyici sütunlar OOF'ta yok.")
        return

    print(f"\n📏 [{label}] v18 Sonrası Eşik Kalibrasyon Raporu (mevcut BUCKET_EVENT_CONTINUOUS_THRESHOLD={BUCKET_EVENT_CONTINUOUS_THRESHOLD}):")
    for col, stats in report["distribution"].items():
        print(
            f"   {col:24s} | p50={stats['p50']:.3f} p75={stats['p75']:.3f} "
            f"p90={stats['p90']:.3f} p95={stats['p95']:.3f} max={stats['max']:.3f}"
        )
    print(f"   {'eşik':>8s} | {'sunday':>8s} | {'event':>8s} | {'normal':>8s}")
    for th, counts in report["bucket_counts_by_threshold"].items():
        marker = "  ← mevcut sabit" if abs(th - BUCKET_EVENT_CONTINUOUS_THRESHOLD) < 1e-9 else ""
        print(f"   {th:8.2f} | {counts['sunday']:8d} | {counts['event']:8d} | {counts['normal']:8d}{marker}")
    print(
        "   ℹ️ Sağlıklı bir eşik: 'normal' bucket'ı sıfırlanmamalı VE 'event' "
        "haftanın neredeyse tamamına eşdeğer olmamalı (bkz. forecasters.py "
        "BUCKET_EVENT_CONTINUOUS_THRESHOLD yorumundaki orijinal 'sunday=0, "
        "event=289, normal=0' vakası)."
    )


def _print_shap_diagnosis(fc, tracked_routes, label):
    """
    Adım 4 — Model 1'in (taban) q50_base'i ufuk (h) arttıkça neden düşüyor?
    fc.capture_debug_features_=True iken predict_sequential()'ın topladığı
    TAM (feat_<col> önekli) feature anlık görüntülerini kullanıp, CatBoost'un
    yerleşik SHAP değerleriyle (get_feature_importance(type="ShapValues"))
    her gün için "hangi feature q50_base'i ne kadar aşağı/yukarı çekti"
    sorusuna cevap verir.

    NOT: Bu, predict_sequential() SIRASINDA üretilen SEQUENTIAL/recursive
    feature değerlerini kullanır (rolling_context'in o ana kadar biriktirdiği
    lag/rolling'ler dahil) — yani "gerçek üretim akışında hangi feature'lar
    modeli aşağı çekiyor" sorusuna cevap verir, statik bir tek-seferlik
    feature setine değil.
    """
    if not getattr(fc, "debug_captured_rows_", None):
        print(
            f"⚠️ [{label}] SHAP teşhisi atlandı: debug_captured_rows_ boş "
            f"(fc.capture_debug_features_ = True set edilmiş mi kontrol edin)."
        )
        return

    cap_df = pd.DataFrame(fc.debug_captured_rows_)
    cap_df = cap_df[cap_df[GROUP_COL].isin(tracked_routes)].copy()
    if cap_df.empty:
        print(f"⚠️ [{label}] SHAP teşhisi atlandı: izlenen rotalar yakalanan satırlarda bulunamadı.")
        return

    feat_cols = [c for c in cap_df.columns if c.startswith("feat_")]
    orig_names = [c[len("feat_"):] for c in feat_cols]
    X = cap_df[feat_cols].copy()
    X.columns = orig_names
    # feature_names_ sırasıyla hizala — CatBoost sütun sırasına duyarlı.
    ordered = [c for c in fc.feature_names_ if c in X.columns]
    X = X[ordered]

    cat_idx = [X.columns.get_loc(c) for c in fc.cat_features_ if c in X.columns]
    pool = Pool(data=X, cat_features=cat_idx)
    shap_values = fc.model_.get_feature_importance(pool, type="ShapValues")
    # Model 1 MultiQuantile (q10/q50/q90 TEK modelde) ile eğitildiği için
    # SHAP çıktısı (n_satır, n_kantil, n_feature+1) — 3 BOYUTLU gelir, düz
    # bir regresördeki gibi (n_satır, n_feature+1) DEĞİL. İlk denemede bunu
    # gözden kaçırıp ikinci ekseni (kantil ekseni) feature ekseni sanıp
    # dilimlemiştim ("Length of values (2)" hatası buradan geldi — 3
    # kantilden [:-1] ile 2 tanesi kalmıştı). q50 her zaman index=1'de
    # (alpha sırası: 0.1, q50_alpha, 0.9 — bkz. forecasters.py loss_fn).
    if shap_values.ndim == 3:
        shap_values = shap_values[:, 1, :]   # q50 boyutunu seç → (n_satır, n_feature+1)
    base_value = shap_values[:, -1]     # son sütun: beklenen değer (bias)
    shap_only = shap_values[:, :-1]

    print(
        f"\n🧪 [{label}] SHAP Teşhisi — Model 1'in q50_base'ini {tracked_routes} "
        f"için ufuk arttıkça neyin çektiğine bakıyoruz:"
    )
    cap_df_reset = cap_df.reset_index(drop=True)
    for i, row in cap_df_reset.iterrows():
        h = row.get("_h_idx", "?")
        rota = row.get(GROUP_COL, "?")
        tarih = row.get(fc.date_column, "?")
        base_pred = row.get("q50_base", row.get("q50", float("nan")))
        contribs = pd.Series(shap_only[i], index=X.columns)
        top = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(8)
        print(f"\n  {rota} | h={h} | tarih={tarih} | q50_base≈{base_pred:.1f} | bias/taban={base_value[i]:.1f}")
        for feat, val in top.items():
            raw_val = X.iloc[i][feat]
            print(f"      {feat:35s} değer={str(raw_val):>14} SHAP={val:+9.2f}")


def run_one_target(label, target_col, model_path, full_df, cutoff, start, end,
                    surge_cap_alpha=None, surge_segment_scale=None,
                    surge_vol_damping_k=None,
                    surge_calibration_factor=None, surge_predict_threshold=None,
                    censor_require_persistence=None, censor_inflation_factor=None,
                    censor_persistence_occurrences=None, censor_persistence_min_hits=None,
                    censor_capacity_df=None, censor_source_tm_column=None,
                    censor_real_capacity_ratio=None):
    fc = DemandForecaster.load_model(model_path)
    # ⚠️ ÖNEMLİ (Öneri A): route_bias_correction_enabled_ bir INSTANCE
    # attribute'u — joblib.dump/load (pickle) __init__()'i tekrar ÇALIŞTIRMAZ,
    # sadece pickle ANINDAKİ değeri saklar. run_forecast.py'nin
    # forecaster_0900/1700.route_bias_correction_enabled_ = True satırı SADECE
    # o script'in kendi bellek-içi nesnesini etkiler; buraya (ayrı bir
    # DemandForecaster.load_model() çağrısı) YANSIMAZ. Bu satır olmadan
    # debug_backtest.py HER ZAMAN flag=False bir kopya analiz eder — Öneri A
    # (rota×gün-türü OOF bias düzeltmesi) hiç uygulanmamış gibi görünür,
    # sonuçlar model gerçekte değişmiş olsa bile eskisiyle birebir aynı çıkar.
    fc.route_bias_correction_enabled_ = True
    # ⚠️ DÜZELTME (bkz. Claude teşhis notu): burada önceden HİÇBİR açıklama
    # olmadan `fc.surge_calibration_factor_ = 1.5` sabit atanıyordu. Bu,
    # modelin joblib'e gömülü/production değerini (varsayılan 1.0x) SESSİZCE
    # eziyordu — yani bu script'in ürettiği TÜM Worst-20/gün-gün backtest
    # tabloları, run_forecast.py'nin gerçekte kullandığından %50 daha
    # agresif bir Surge/Residual düzeltmesiyle üretiliyordu. Şimdi opt-in:
    # override SADECE --surge-calibration-factor açıkça verildiyse uygulanır.
    if surge_calibration_factor is not None:
        fc.surge_calibration_factor_ = surge_calibration_factor
        log.info(
            f"ℹ️  [{label}] surge_calibration_factor_ override: "
            f"{surge_calibration_factor}x (production/model varsayılanı eziliyor)"
        )
    else:
        log.info(
            f"ℹ️  [{label}] surge_calibration_factor_ override YOK — modelin "
            f"kendi kayıtlı değeri kullanılıyor: "
            f"{getattr(fc, 'surge_calibration_factor_', 1.0)}x"
        )
    if surge_cap_alpha is not None:
        fc.surge_relative_cap_alpha_ = surge_cap_alpha
        log.info(f"ℹ️  [{label}] Faz 2 Relative Cap aktif: alpha={surge_cap_alpha}")
    if surge_vol_damping_k is not None:
        old_k = getattr(fc, "surge_volume_damping_k_", 3.0)
        v_crit = getattr(fc, "surge_volume_damping_v_crit_", 150.0)
        fc.surge_volume_damping_k_ = surge_vol_damping_k
        midpoint = v_crit * 1.0986 / max(surge_vol_damping_k, 1e-9)
        log.info(
            f"ℹ️  [{label}] S_vol k override: {old_k}→{surge_vol_damping_k} "
            f"(v_crit={v_crit}, gerçek %50 bastırma noktası ≈{midpoint:.1f} desi "
            f"— bkz. run_forecast.py'deki S_vol matematik notu: "
            f"S_vol≡tanh(k·V/(2·v_crit)), midpoint=v_crit·ln(3)/k)"
        )
    if surge_segment_scale is not None:
        fc.surge_segment_scale_ = surge_segment_scale
        log.info(f"ℹ️  [{label}] Faz 2b Segment Scale aktif: segmentler={surge_segment_scale}")
    if surge_predict_threshold is not None:
        fc.surge_predict_continuous_threshold_ = surge_predict_threshold
        log.info(
            f"ℹ️  [{label}] Tahmin-zamanı sıkı sürekli-tetikleyici eşiği aktif: "
            f"{surge_predict_threshold} (eğitim eşiği 0.05 sabit kalıyor, sadece "
            f"predict()'te hangi satırlara correction uygulanacağı bu eşikle süzülüyor)"
        )

    # --- [Adım 1 — ucuz doğrulama testi] Unconstraining (sahte-tavan) A/B ---
    # bkz. features.py::unconstrain_censored_demand. Model attribute'ları
    # (censor_*_) load_model() içinde eski .joblib'ler için de varsayılanla
    # (v2/persistence açık) enjekte edilir — burada SADECE açıkça CLI'dan
    # verilenler ezilir (surge_calibration_factor ile AYNI opt-in deseni).
    if censor_require_persistence is not None:
        fc.censor_require_weekday_persistence_ = (censor_require_persistence == "true")
        log.info(
            f"ℹ️  [{label}] censor_require_weekday_persistence_ override: "
            f"{fc.censor_require_weekday_persistence_} "
            f"({'v1 (eski, tek-seferlik tespit)' if not fc.censor_require_weekday_persistence_ else 'v2 (yeni, kalıcılık şartlı)'})"
        )
    if censor_inflation_factor is not None:
        fc.censor_inflation_factor_ = censor_inflation_factor
        log.info(
            f"ℹ️  [{label}] censor_inflation_factor_ override: {censor_inflation_factor}"
            + (" (unconstraining FİİLEN KAPALI)" if censor_inflation_factor == 1.0 else "")
        )
    if censor_persistence_occurrences is not None:
        fc.censor_persistence_occurrences_ = censor_persistence_occurrences
    if censor_persistence_min_hits is not None:
        fc.censor_persistence_min_hits_ = censor_persistence_min_hits

    # --- [Adım 2 — kalıcı düzeltme] GERÇEK kapasite verisiyle gate'leme ---
    # bkz. features.py::unconstrain_censored_demand Mantık 3.5. VERİLİRSE
    # yukarıdaki persistence (istatistiksel proxy) sadece capacity_df'te
    # eşleşmeyen TM'ler için fallback olarak kalır.
    if censor_capacity_df is not None:
        fc.censor_capacity_df_ = censor_capacity_df
        fc.censor_source_tm_column_ = censor_source_tm_column or "kaynak_tm"
        if censor_real_capacity_ratio is not None:
            fc.censor_real_capacity_ratio_ = censor_real_capacity_ratio
        log.info(
            f"ℹ️  [{label}] GERÇEK kapasite gate'i AKTİF: "
            f"{len(censor_capacity_df)} TM için elleçleme kapasitesi yüklendi, "
            f"source_tm_column='{fc.censor_source_tm_column_}', "
            f"real_capacity_ratio={fc.censor_real_capacity_ratio_}"
        )

    # Context buffer'ı CUTOFF'a göre GEÇİCİ olarak yeniden hesapla
    # (fit-zamanı buffer'ı değiştirmiyoruz, sadece bu backtest için)
    hist_df = full_df[full_df[DATE_COL] <= pd.Timestamp(cutoff)].copy()
    fc._save_context_buffer(hist_df)

    target_dates = pd.date_range(start, end, freq="D")
    pred_df = build_prediction_frame(full_df, target_dates)

    # Adım 4 — SHAP teşhisi için tam feature yakalamayı aç (varsayılan kapalı,
    # bkz. forecasters.py::__init__ açıklaması — normal akışı etkilemez).
    fc.capture_debug_features_ = True
    fc.debug_captured_rows_ = []

    preds = fc.predict_sequential(pred_df)
    preds_df = pd.DataFrame(preds).assign(tarih=lambda d: d["tarih"].astype(str).str[:10])

    _izlenen_rotalar_shap = ["Yalova → İstanbul", "Kocaeli → İstanbul", "İstanbul → Yalova"]
    _print_shap_diagnosis(fc, _izlenen_rotalar_shap, label)
    _print_bucket_correction_diagnosis(fc, _izlenen_rotalar_shap, label)
    _print_oof_bucket_raw_diagnosis(fc, _izlenen_rotalar_shap, label)
    _print_bucket_threshold_calibration_diagnosis(fc, label)

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
    _izlenen_cols = [GROUP_COL, "tarih", "y_true", "q50", "q90", "hata_pct"]
    # q50_base (Model 1 HAM çıktısı, Model 2/surge düzeltmesinden ÖNCE) ve
    # correction_contrib (Model 2'nin eklediği miktar = q50 - q50_base) —
    # bu ikisi olmadan "düşüş nereden geliyor" sorusuna cevap veremeyiz:
    # q50_base da mı ufuk arttıkça düşüyor (Model 1/lag-rolling kaynaklı),
    # yoksa q50_base sabit kalıp SADECE Model 2'nin katkısı mı küçülüyor
    # (surge model kaynaklı)?
    _has_base = "q50_base" in merged.columns
    _tracked = merged[merged[GROUP_COL].isin(_izlenen_rotalar)].copy()
    _tracked["hata_pct"] = (_tracked["q50"] - _tracked["y_true"]) / _tracked["y_true"].replace(0, np.nan) * 100
    if _has_base:
        _tracked["base_hata_pct"] = (_tracked["q50_base"] - _tracked["y_true"]) / _tracked["y_true"].replace(0, np.nan) * 100
        _tracked["correction_contrib"] = _tracked["q50"] - _tracked["q50_base"]
        _izlenen_cols = [GROUP_COL, "tarih", "y_true", "q50_base", "base_hata_pct", "correction_contrib", "q50", "q90", "hata_pct"]
    print(
        _tracked[_izlenen_cols]
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

    # (B) Rota bazlı WAPE + YÖN (bias) — WAPE/decision_regret TEK BAŞINA
    # yeterli değil: yüksek hata hem "sistematik eksik/fazla tahmin"
    # (Yalova tipi — HUB feature / bias-correction gerektirir) hem de
    # "ortalamada dengeli ama gürültülü" (shrinkage / trust-decay gerektirir)
    # bir rotadan gelebilir — ikisi FARKLI tedavi ister. Bu yüzden bias'ı
    # ayrı, açık bir sütunda raporluyoruz; tier/formül tasarımı buradan
    # sonra gelmeli.
    route_stats = merged.groupby(GROUP_COL).apply(
        lambda g: pd.Series({
            "n": len(g),
            "ort_hacim": g["y_true"].mean(),
            "toplam_hacim": g["y_true"].sum(),
            "WAPE": wape(g["y_true"].values, g["q50"].values),
            "decision_regret": decision_regret(g["y_true"].values, g["q50"].values, spot_multiplier=9.0),
            # Hacim-ağırlıklı bias: toplam(q50-y_true)/toplam(y_true) — satır
            # bazlı ortalama DEĞİL, çünkü tek büyük günlük outlier bias
            # yönünü boğabilir. WAPE'nin payda mantığıyla tutarlı.
            "bias_pct": (
                (g["q50"].sum() - g["y_true"].sum()) / g["y_true"].sum() * 100
                if g["y_true"].sum() > 0 else np.nan
            ),
        })
    ).round(3)

    # Yön etiketi: |bias_pct| küçükse hata muhtemelen VARYANS kaynaklı
    # (trust-decay/shrinkage adayı); büyükse SİSTEMATİK yön kaynaklı
    # (bias-correction/HUB adayı). %15 eşiği keyfi bir başlangıç noktası —
    # aşağıdaki dağılıma bakıp rotalarınıza göre ayarlayın.
    def _yon(b):
        if pd.isna(b):
            return "?"
        if b <= -15:
            return "EKSİK (sistematik)"
        if b >= 15:
            return "FAZLA (sistematik)"
        return "dengeli (varyans?)"
    route_stats["yon"] = route_stats["bias_pct"].apply(_yon)

    # Shrinkage ağırlığı ÖNİZLEMESİ (Adım 2'de forecasters.py'ye taşınacak):
    # az gözlemli rota → global davranışa yaklaşsın (w küçük), çok gözlemli
    # rota → kendi sinyaline güvensin (w büyük). Sert 4'lü tier yerine
    # SÜREKLİ bir fonksiyon — komşu rota sınırında ani sıçrama olmaz.
    SHRINKAGE_K = 30.0  # kaç "eşdeğer gözlem" sonra rotaya tam güvenilsin
    route_stats["w_shrink_preview"] = (
        route_stats["n"] / (route_stats["n"] + SHRINKAGE_K)
    ).round(3)

    # Sadece anlamlı hacimli rotalara odaklan (örn. ort_hacim >= 50) — aksi halde
    # WAPE zaten payda küçük olduğu için yapısal olarak şişer, teşhisi bozar.
    route_stats_relevant = route_stats[route_stats["ort_hacim"] >= 50].sort_values("decision_regret", ascending=False)
    print(f"\n🚩 [{label}] En Kötü 20 Rota (ort_hacim>=50, decision_regret'e göre sıralı):")
    print(route_stats_relevant.head(20).to_string())

    # YENİ: bias'a göre ayrı sıralama — decision_regret'in maskeleyebileceği
    # "sistematik yön" sorununu doğrudan öne çıkarır. Adım 1 (HUB) ve
    # Adım 3 (trust decay) gibi müdahaleler SADECE burada üstte çıkan
    # rotalara uygulanmalı, decision_regret listesindeki HER rotaya değil.
    route_stats_by_bias = route_stats_relevant.reindex(
        route_stats_relevant["bias_pct"].abs().sort_values(ascending=False).index
    )
    print(f"\n🧭 [{label}] En Kötü 20 Rota (ort_hacim>=50, |bias_pct|'e göre sıralı — YÖN teşhisi):")
    print(route_stats_by_bias.head(20).to_string())

    print(f"\n📊 [{label}] Rota Segmentasyonu Özeti:")
    print(f"   Toplam rota sayısı           : {len(route_stats)}")
    print(f"   ort_hacim>=50 rota sayısı    : {len(route_stats_relevant)}")
    print(f"   Bunların medyan WAPE'i       : {route_stats_relevant['WAPE'].median():.3f}")
    print(f"   Bunların WAPE>0.5 olan sayısı: {(route_stats_relevant['WAPE'] > 0.5).sum()}")
    print(f"   Sistematik EKSİK (bias<=-15%): {(route_stats_relevant['yon'] == 'EKSİK (sistematik)').sum()} rota")
    print(f"   Sistematik FAZLA (bias>=+15%): {(route_stats_relevant['yon'] == 'FAZLA (sistematik)').sum()} rota")

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

    # --- [Adım 2 — kalıcı düzeltme] Gerçek elleçleme kapasitesi dosyası ---
    # bkz. --censor-capacity-file --help metni. Bir kez okunup her iki
    # hedefe (09:00/17:00) de AYNI DataFrame referansı geçiriliyor.
    censor_capacity_df = None
    if args.censor_capacity_file is not None:
        censor_capacity_df = pd.read_excel(args.censor_capacity_file)
        log.info(
            f"ℹ️  --censor-capacity-file okundu: '{args.censor_capacity_file}' "
            f"({len(censor_capacity_df)} TM, sütunlar: {list(censor_capacity_df.columns)})"
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
            surge_vol_damping_k=args.surge_vol_damping_k,
            surge_calibration_factor=args.surge_calibration_factor,
            surge_predict_threshold=args.surge_predict_threshold,
            censor_require_persistence=args.censor_require_persistence,
            censor_inflation_factor=args.censor_inflation_factor,
            censor_persistence_occurrences=args.censor_persistence_occurrences,
            censor_persistence_min_hits=args.censor_persistence_min_hits,
            censor_capacity_df=censor_capacity_df,
            censor_source_tm_column=args.censor_source_tm_column,
            censor_real_capacity_ratio=args.censor_real_capacity_ratio,
        )


if __name__ == "__main__":
    main()