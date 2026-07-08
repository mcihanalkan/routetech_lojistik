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
    python optimize.py --gamma 0.6               # spot/atıl maliyet oranı: (C_spot/C_atil) - 1
    python optimize.py --gamma 0               # asimetrik ceza kapalı (eski davranış)

Her çalıştırmada JSON'daki ilgili bucket güncellenir.
Yeni bir veri boyutu (örn. 500K satır) gelirse otomatik yeni bucket eklenir.

--- Asimetrik Maliyet Mantığı (v4) ---
Önceki sürümler sabit alpha=0.5 (medyan) ile simetrik WAPE minimize ediyordu.
Lojistik operasyonlarda eksik tahmin → spot araç → fazla tahmin maliyetinin
katları kadar pahalıdır. Bu asimetriyi modele yansıtmak için:

  1. alpha (Quantile seviyesi) artık Optuna'nın arama uzayında [0.50, 0.85]
     → model hangi kuantilin en iyi kapasite kararını verdiğini kendisi bulur.

  2. Objective fonksiyonu hibrit:
       L_hybrid = WAPE + γ × Underprediction_Penalty
     WAPE terimi modelin "sonsuz yukarı tahmin" kaçış yolunu kapatır.
     γ (gamma) = (C_spot / C_atıl) - 1  →  operasyonel maliyet rasyondan türetilir.
     Artık --gamma verilmezse Araç_Kapasite_Maliyet.xlsx'ten OTOMATİK hesaplanır
     (bkz. derive_gamma_from_costs()). Gerçek veriyle γ ≈ 0.60 (spot ≈1.6× atıl).

  3. Hem alpha hem de hybrid_score JSON'a yazılır; forecasters.py alpha'yı
     loss_function içinde kullanmalıdır.

--- v5 Değişiklikleri (run_forecast.py ile tutarlılık) ---
Önceki sürümde optimize.py, forecasters.py'nin GERÇEKTE eğittiği modelden
(4-fold ensemble, kırpılmamış veri, early stopping yok) yapısal olarak farklı
bir model (tek model, IQR kırpılmış veri, early stopping ile) üzerinde arama
yapıyordu — bu yüzden aynı JSON bucket'ı iki farklı çalıştırmada oldukça
farklı sonuçlar üretiyordu ve optimize.py'nin bulduğu WAPE, run_forecast.py'nin
gerçek test WAPE'siyle örtüşmüyordu. v5'te:
  - TPESampler'a sabit seed verildi (reprodüktibilite).
  - build_feature_matrix artık drop_na=False (forecasters.py ile aynı satırlar).
  - Objective, forecasters.py::fit()'teki 4-fold ensemble şemasını GERÇEKTEN
    simüle ediyor (early stopping kaldırıldı, iterations sabit kullanılıyor).
  - Hem tüm-gün hem "temiz" (anormal hafta hariç) WAPE ayrı ayrı loglanıyor.
Bu fold-ensemble simülasyonu trial başına ~4x daha pahalı; --timeout'u buna
göre büyütmek (örn. 1800-3600s) veya --trials sayısını azaltmak gerekebilir.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

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
# optimize.py, run_forecast.py ile AYNI dizinde yaşar: routetech_lojistik/src/predict_model/
# _HERE         → predict_model/  (hyperparams_map.json, alns_payload.json vb. burada yaşar)
# _PROJECT_ROOT → routetech_lojistik/  (data/raw/ burada yaşar — 2 üst dizin, run_forecast.py ile aynı mantık)
_HERE            = Path(__file__).resolve().parent
_PROJECT_ROOT    = _HERE.parent.parent
HYPERPARAMS_PATH = _HERE / "hyperparams_map.json"
TARGET_COL  = "desi_hacmi"
DATE_COL    = "tarih"
GROUP_COL   = "rota"
KAYNAK_COL  = "kaynak_tm"
VARIS_COL   = "varis_tm"

# run_forecast.py (DemandForecaster) ile BİREBİR aynı tutulmalı.
# Biri değişirse diğeri de manuel güncellenmeli — forecasters.py'nin
# varsayılanı da bu değerlerle eşleşmeli (bkz. modül üstü not).
ROLLING_WINDOWS  = [7, 14]

# --- Uyarlanabilir lag seçimi -----------------------------------------------
# lag_21 / lag_30, her rota için ilk N günü NaN yapıp DÜŞÜRÜYOR (drop_na=True).
# Küçük veri setlerinde (örn. mevcut: 10.770 gerçek kayıt) bu kayıp
# (max_lag × rota_sayısı) feature'ın kattığı sinyale değmiyor. Veri büyüdükçe
# (daha çok gün / rota biriktikçe) otomatik olarak devreye girsinler diye
# eşik değerine bağladık — elle açıp kapatmaya gerek kalmasın.
#
# Eşikler İLK TAHMİN — kesin bilim değil. Gerçek veri büyüdükçe (hyperparams_map.json
# geçmişindeki satır sayılarına bakarak) ayarlanabilir.
LAG_21_MIN_ROWS = 15_000   # bu eşiğin altında lag_21 kullanılmaz
LAG_30_MIN_ROWS = 20_000   # LAG_21'den biraz yüksek — ama "çok" değil


def select_lags(n_real_rows: int) -> list:
    """Veri büyüklüğüne göre lag_21/lag_30'u otomatik ekler/çıkarır — bkz. yukarıdaki not."""
    lags = [1, 7, 14]
    if n_real_rows >= LAG_21_MIN_ROWS:
        lags.append(21)
    if n_real_rows >= LAG_30_MIN_ROWS:
        lags.append(30)
    return lags

# forecasters.py → DemandForecaster.outlier_clip_multiplier varsayılanıyla AYNI (3.0).
# _fit_clip() mantığı: upper = Q75 + multiplier × (Q75 - Q25), rota bazlı, sadece üst uç.
OUTLIER_CLIP_MULTIPLIER = 3.0

# ---------------------------------------------------------------------------
# Varsayılan gamma — CLI ile veya optimize() çağrısında override edilebilir.
# γ = (C_spot / C_atıl) - 1
#
# Araç_Kapasite_Maliyet.xlsx'ten türetildi (kapasite-ağırlıklı ortalama, 4 araç
# tipi): γ ≈ 0.5988 → spot araç ortalama ≈1.60× atıl (kiralık) maliyeti.
# Bu, yalnızca --cost-file bulunamazsa kullanılan FALLBACK değeridir;
# normal çalışmada gamma her seferinde dosyadan yeniden hesaplanır (bkz. CLI).
# Asimetrik cezayı tamamen kapatmak için → --gamma 0.0
# ---------------------------------------------------------------------------
DEFAULT_GAMMA = 0.5988

# alpha arama aralığı: medyanın altına düşmesin (eksik tahmin yanlılığı artar)
# v9: --alpha-sweep sonucuna göre knee noktası bulundu ve SABİTLENDİ.
# Marjinal analiz (ΔWAPE / ΔUnderprediction): 0.70→0.72 aralığında oran
# hâlâ ucuz (0.31), 0.72→0.76 arasında oran 5.2'ye fırlıyor, 0.82→0.84
# arasında ise açık bir "duvar" var (oran 9.7 — +1.65pp WAPE karşılığında
# yalnızca -0.17pp Underpred kazancı). alpha=0.72 hem bu marjinal analizin
# hem de (gürültülü olsa da) hibrit-skor minimumunun ortak işaret ettiği
# nokta. Artık arama uzayında DEĞİL, sabit — Optuna diğer 5 parametreye
# (iterations/depth/lr/l2/bagging_temp) odaklanıyor.
# alpha arama aralığı: medyanın altına düşmesin (eksik tahmin yanlılığı artar)
# v11: v9'da ALPHA_MIN=ALPHA_MAX=0.72 (TAM sabit) denendi ve haklı bir
# itiraz geldi — bu, farklı/gelecekteki veriye adapte olma yeteneğini
# yok eder. --alpha-sweep sonucu ZATEN net bir sinyal verdi: 0.82'den
# sonrası (yüksek alpha) sistematik olarak kötü bir takas (WAPE'de büyük
# sıçrama, Underpred'de ihmal edilebilir kazanç — bkz. marjinal analiz).
# Bu üst-uç davranışı, verinin çarpıklık yapısından (skewness ~1.3+) gelen
# YAPISAL bir özellik; veri büyüklüğü/rota karışımı değişse de muhtemelen
# kalıcı — ama garanti değil. Bu yüzden TEK NOKTAYA değil, kanıtla
# doğrulanmış, hâlâ arama yapılabilen bir BANDA sabitliyoruz: [0.60, 0.78].
# Alt sınır önceki geniş taramada iyi bölgenin başlangıcı (~0.60), üst sınır
# knee'nin hemen ötesi (~0.78, 0.82+ köşesini dışarıda bırakır). Optuna bu
# bandın içinde hâlâ serbestçe arar — gelecekte veri değişirse farklı bir
# nokta bulabilir, ama kanıtlanmış kötü uçlara (0.84-0.90) gidemez.
# Bu bandı periyodik olarak (özellikle veri belirgin şekilde değiştiğinde)
# --alpha-sweep ile yeniden doğrulayın.
ALPHA_MIN = 0.60
ALPHA_MAX = 0.78

# Bant genişliği (q90-q10)/q_orta cezası — Optuna sadece WAPE+underprediction'a
# bakınca, WAPE'i minik bir miktar iyileştirmek için bandı aşırı genişletebiliyordu
# (gözlemlenen: rel_width=3.075, hiçbir cezası yokken). Bu ceza olmadan HPO,
# hangi config'in üretimde dar/geniş bant üreteceğini önemsemiyordu.
# ⚠️ Bu ağırlık İLK TAHMİN — kesin kalibre edilmiş bir değer değil. rel_width
# tipik olarak 0.5-3.0 aralığında, WAPE ise 0.2-0.3 civarında; 0.03 ile
# rel_width=1.0 → skora +0.03 eklenir (WAPE'nin ~%10-15'i kadar, anlamlı ama
# baskın değil). Sonucu görüp gerekirse büyütün/küçültün.
BETA_WIDTH = 0.03

# ---------------------------------------------------------------------------
# Kuantil Çakışma (Quantile Crossing) Cezası — v6
# ---------------------------------------------------------------------------
# MultiQuantile tek modelde q10/q_mid/q90'ı BİRLİKTE öğretir ama CatBoost
# satır bazında q10 ≤ q_mid ≤ q90 sıralamasını GARANTİ ETMEZ. Bu satırlarda
# q90 < q10 çıkabilir → rel_width_q90_q10 NEGATİF olur (v5'te gözlemlenen
# -0.214 tam olarak bu).
#
# v5'in objective'inde BETA_WIDTH pozitif olduğu için negatif rel_width skoru
# DÜŞÜRÜYORDU — yani Optuna kuantil çakışmasını yanlışlıkla ÖDÜLLENDİRİYORDU.
# uncertainty.py::DemandBand zaten üretimde min/max ile bu çakışmayı düzeltiyor
# (payload bozuk değil), ama HPO'nun optimize ettiği sinyal gerçek üretim
# davranışını yansıtmıyordu.
#
# v6: (1) skor hesaplanmadan önce üretimdeki AYNI monotonluk düzeltmesi
#     uygulanır (q10=min(q10,mid), q90=max(q90,mid)) → rel_width artık hep ≥0.
#     (2) çakışma oranı (crossing_rate) doğrudan skora ceza olarak eklenir.
# Bu ağırlık İLK TAHMİN — WAPE'nin (~0.18-0.25) yaklaşık %10-15'i kadar,
# baskın değil ama artık gerçek bir sinyal.
CROSSING_PENALTY_WEIGHT = 0.10


# ---------------------------------------------------------------------------
# Gamma'yı gerçek maliyet verisinden türet
# γ = (C_spot / C_atıl) - 1
#
# Araç_Kapasite_Maliyet.xlsx yapısı (4 araç tipi: Tır/Kamyon/Hafif Kamyon/Kamyonet):
#   "Kiralık Araç Günlük Kira (TL)"        → C_atıl: taahhütlü/kiralık aracın
#                                             sabit günlük maliyeti — kullanılsa da
#                                             kullanılmasa da ödenir (boş/atıl kapasite maliyeti).
#   "Spot Araç Sabit Günlük Maliyet (TL)"  → C_spot: kapasite yetmediğinde çağrılan
#                                             spot aracın sabit günlük maliyeti.
#
# Her araç tipi için gamma_tip = (Spot/Kiralık) - 1 hesaplanır, sonra filodaki
# kapasite (desi) ile ağırlıklandırılıp tek bir γ'ye indirgenir (büyük araçlar
# toplam hacmin daha büyük kısmını taşıdığı için ağırlığı daha yüksek).
#
# Gerçek veriyle (2026-07) hesaplanan değerler:
#   Tır: γ=0.6714 | Kamyon: γ=0.5276 | Hafif Kamyon: γ=0.75 | Kamyonet: γ=0.2667
#   → Kapasite-ağırlıklı ortalama: γ ≈ 0.5988   (eşit ağırlıklı ort.: 0.5539,
#     ratio-of-means: 0.5825 — üç yöntem de 0.55-0.60 bandında, tutarlı)
# ---------------------------------------------------------------------------

_SPOT_COST_CANDIDATES = ["Spot Araç Sabit Günlük Maliyet (TL)", "Spot Araç Sabit Günlük Maliyet",
                          "spot_maliyet", "spot_cost", "C_spot", "spot"]
_IDLE_COST_CANDIDATES = ["Kiralık Araç Günlük Kira (TL)", "Kiralık Araç Günlük Kira",
                          "atil_maliyet", "idle_cost", "C_atil", "atıl"]
_CAPACITY_CANDIDATES  = ["Kapasite (desi)", "kapasite", "capacity"]


def derive_gamma_from_costs(cost_file: str, fallback: float = None) -> tuple:
    """
    Araç_Kapasite_Maliyet.xlsx'ten gamma türet: γ = (C_spot / C_atıl) - 1.

    Her satır (araç tipi) için γ_tip = (Spot Günlük / Kiralık Günlük) - 1
    hesaplanır, ardından Kapasite (desi) sütunu ile ağırlıklandırılıp
    filo-geneli tek bir γ'ye indirgenir. Kapasite sütunu bulunamazsa
    eşit ağırlıklı ortalamaya düşer.

    Dönüş: (gamma, kaynak_notu) — kaynak_notu hyperparams_map.json'a yazılır.
    """
    if fallback is None:
        fallback = DEFAULT_GAMMA
    path = Path(cost_file)
    if not path.exists():
        logger.warning(
            f"⚠️  Maliyet dosyası bulunamadı: {cost_file}\n"
            f"   Gamma otomatik türetilemedi → fallback kullanılıyor: {fallback}"
        )
        return fallback, f"fallback (dosya bulunamadı): {fallback}"

    df = pd.read_excel(path) if str(path).endswith(".xlsx") else pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    cols_lower = {c.lower().strip(): c for c in df.columns}

    def _find(candidates):
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    spot_col = _find(_SPOT_COST_CANDIDATES)
    idle_col = _find(_IDLE_COST_CANDIDATES)
    cap_col  = _find(_CAPACITY_CANDIDATES)

    if spot_col is None or idle_col is None:
        logger.warning(
            f"⚠️  Maliyet sütunları isim bazlı bulunamadı "
            f"(spot={spot_col}, atıl={idle_col}) → fallback kullanılıyor: {fallback}\n"
            f"   Mevcut sütunlar: {list(df.columns)}\n"
            f"   Beklenen adaylar → spot: {_SPOT_COST_CANDIDATES} | atıl: {_IDLE_COST_CANDIDATES}"
        )
        return fallback, f"fallback (sütun bulunamadı): {fallback}"

    if (df[idle_col] <= 0).any():
        logger.warning(f"⚠️  '{idle_col}' içinde <=0 değer var → fallback kullanılıyor: {fallback}")
        return fallback, f"fallback (atıl maliyet <=0): {fallback}"

    gamma_per_type = (df[spot_col] / df[idle_col]) - 1.0

    if cap_col is not None and (df[cap_col] > 0).all():
        weights = df[cap_col]
        gamma = float((gamma_per_type * weights).sum() / weights.sum())
        method = f"kapasite-ağırlıklı ort. (sütun: '{cap_col}')"
    else:
        gamma = float(gamma_per_type.mean())
        method = "eşit ağırlıklı ortalama (kapasite sütunu bulunamadı)"

    breakdown = ", ".join(
        f"{row.get('Araç Adı', f'satır{i}')}=γ{gamma_per_type[i]:.4f}"
        for i, row in df.iterrows()
    )
    note = (
        f"gamma={method}: {gamma:.4f} | araç-tipi bazlı: {breakdown} "
        f"[kaynak: {path.name}, sütunlar: spot='{spot_col}', atıl='{idle_col}']"
    )
    logger.info(f"✅ Gamma gerçek veriden türetildi → {note}")
    return gamma, note


# ---------------------------------------------------------------------------
# Hibrit kayıp fonksiyonu
# ---------------------------------------------------------------------------

# NOT (v5): Bu fonksiyon artık optimize() içinde ÇAĞRILMIYOR — fold-ensemble
# simülasyonu forecasters.py::fit() ile birebir aynı şekilde KIRPILMAMIŞ
# veri üzerinde eğitiliyor (fold train setleri df_features'tan, IQR clip
# uygulanmadan alınıyor). Referans/gelecekte kullanım için saklandı.
def _iqr_clip(series: pd.Series, multiplier: float = OUTLIER_CLIP_MULTIPLIER) -> pd.Series:
    """
    forecasters.py'deki DemandForecaster._fit_clip() ile BİREBİR AYNI mantık
    (doğrulandı — bkz. forecasters.py satır ~312-360):

      1. Negatif desi → 0.0  (fiziksel kural)
      2. Üst sınır kırpma: upper = Q75 + multiplier × (Q75 - Q25), ROTA BAZLI.
         Alt sınır kırpılmaz — sadece aşırı yüksek talep sıçramaları kırpılır.

    forecasters.py IQR eşiklerini SADECE train'den öğrenip train+test'e uygular
    (data leakage önlenir); burada da aynı şekilde SADECE train fold'una
    uygulanır, validation ham kalır.
    """
    s = series.clip(lower=0.0)
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    upper = q75 + multiplier * (q75 - q25)
    return s.clip(upper=upper)


def _hybrid_score(y_true: np.ndarray, y_pred: np.ndarray, gamma: float) -> float:
    """
    L_hybrid = WAPE + γ × Underprediction_Penalty

    WAPE  : genel tahmin doğruluğunu ölçer; modelin "sonsuz yukarı"
            tahmin yaparak γ cezasından kaçmasını engeller (çapa görevi).
    Ceza  : sadece eksik tahmin (y_true > y_pred) olduğunda devreye girer.
    γ = 0 → saf WAPE (eski davranış), γ > 0 → asimetrik ceza aktif.
    """
    total = np.sum(y_true)
    if total <= 0:
        return 1.0
    wape        = np.sum(np.abs(y_true - y_pred)) / total
    underpred   = np.sum(np.maximum(y_true - y_pred, 0.0)) / total
    return float(wape + gamma * underpred)


# ---------------------------------------------------------------------------
# Veri boyutuna göre Optuna arama uzayı
# ---------------------------------------------------------------------------

def _search_space(n_rows: int) -> dict:
    """
    Veri boyutuna göre Optuna arama uzayı.
    Küçük veri → sığ/regularize, büyük veri → derin/kapasiteli.

    v4 değişikliği: alpha alanı eklendi (tüm bucket'larda ortak [0.50, 0.85]).
    """
    if n_rows < 5_000:
        return dict(iter=(200, 600, 100), depth=(3, 6), lr=(0.02, 0.15), l2=(5.0, 25.0), bag=(0.1, 0.5))
    if n_rows < 30_000:
        # v7: v6 depth=4-6 + l2=15-40 kombinasyonu FAZLA sıkıştırdı —
        # gerçek run'da Optuna l2=36.9 (tavana yakın) + depth=6 seçti ve
        # sonuç UNDERFIT oldu: Train WAPE %12.6→%15.8, Test WAPE %25.7→%27.1
        # (her ikisi de KÖTÜLEŞTİ, overfit farkı 13.1pp→11.3pp ile appenas
        # kapandı). Ayrıca ortalama bant genişliği (U_rel) 0.86→1.16'ya
        # fırladı — düşük kapasiteli + ağır regularize model, kuyruk
        # kuantillerini (q10/q90) çok daha az güvenle/geniş tahmin etti.
        # Bu da risk modelinin (tau_base=0.50, ~0.55 taban varsayımıyla
        # kalibre) sistematik olarak MEDIUM/HIGH üretmesine yol açtı
        # (HIGH 5→37) — gerçek risk artışı değil, kalibrasyon uyumsuzluğu.
        #
        # v7: v5 (depth=8, aşırı kapasiteli) ile v6 (depth=4-6, aşırı
        # v11: v10'da depth TAM sabitlendi (6,6) — haklı itiraz geldi, bu
        # da gelecekte veri değişirse adaptasyonu kapatır. Kanıt hâlâ geçerli
        # (depth=7, iki bağımsız run'da gerçek tahmin haftasında kötü sonuç
        # verdi: HIGH=19/DR=5984 ve HIGH=12/DR=6419; depth=6 ise HIGH=2/
        # DR=5289 ile en iyisiydi) ama TEK NOKTAYA değil, kötü ucu (7-8)
        # dışlayan bir BANDA (5-6) sabitliyoruz. Optuna bu bandın içinde
        # hâlâ arar — 5 mi 6 mı daha iyi, veri değiştikçe kendi bulur.
        # Bu bandı da --alpha-sweep gibi periyodik olarak (veri belirgin
        # değiştiğinde) tam arama uzayına (5-8) açıp yeniden doğrulayın.
        # l2 alt sınırı 8→15'e çekildi: v9'daki l2=11.82 (düşük regularizasyon)
        # overfit'e katkıda bulunan diğer faktördü.
        return dict(iter=(350, 550, 50), depth=(5, 6), lr=(0.02, 0.05), l2=(15.0, 26.0), bag=(0.2, 0.5))
    if n_rows < 100_000:
        return dict(iter=(600, 1500, 100), depth=(5, 8), lr=(0.005, 0.08), l2=(2.0, 40.0),  bag=(0.1, 0.6))
    if n_rows < 500_000:
        return dict(iter=(800, 2000, 100), depth=(6, 8), lr=(0.003, 0.05), l2=(0.5, 40.0),  bag=(0.1, 0.7))
    return dict(iter=(1000, 3000, 200), depth=(7, 9), lr=(0.001, 0.03), l2=(0.1, 40.0),  bag=(0.1, 0.8))


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
# Fold tarihleri — forecasters.py::DemandForecaster.fit() ile BİREBİR AYNI.
# ---------------------------------------------------------------------------
FOLD_DATES = [
    ("Fold 1", "2026-04-14", "2026-04-20"),
    ("Fold 2", "2026-04-21", "2026-04-27"),
    ("Fold 3", "2026-04-28", "2026-05-04"),
    ("Fold 4", "2026-05-05", "2026-05-10"),
]


def _fit_fold_ensemble(feat_df, feature_cols, cat_features, params, fold_dates=FOLD_DATES):
    models = []
    for fold_name, val_start, val_end in fold_dates:
        fold_train = feat_df[feat_df[DATE_COL] < val_start]
        if fold_train.empty:
            logger.warning(f"   ⚠️  {fold_name}: train seti boş, atlanıyor.")
            continue
        X_ft = fold_train[feature_cols]
        y_ft = fold_train[TARGET_COL]
        model = CatBoostRegressor(**params)
        model.fit(Pool(X_ft, y_ft, cat_features=cat_features), verbose=False)
        models.append(model)
    return models


def _ensemble_predict(models, X):
    all_preds = [np.maximum(m.predict(X), 0) for m in models]
    return np.median(all_preds, axis=0)


def _correct_and_score(y_true, raw_preds, gamma, beta_width):
    """
    Ham MultiQuantile çıktısına (n,3) üretimdeki (uncertainty.py::DemandBand)
    AYNI monotonluk düzeltmesini uygular ve hibrit+bant+çakışma skorunu
    hesaplar. rel_width bu düzeltmeden sonra hep ≥0 olur.

    Dönüş: (score, rel_width_corrected, crossing_rate)
    """
    preds = raw_preds[:, 1]
    q10c  = np.minimum(raw_preds[:, 0], preds)   # DemandBand: q10 = min(q10, q50)
    q90c  = np.maximum(raw_preds[:, 2], preds)   # DemandBand: q90 = max(q90, q50)

    crossing_rate = float(np.mean(raw_preds[:, 2] < raw_preds[:, 0]))
    rel_width = float(np.mean((q90c - q10c) / np.maximum(preds, 1.0)))

    score = (
        _hybrid_score(y_true, preds, gamma=gamma)
        + beta_width * rel_width
        + CROSSING_PENALTY_WEIGHT * crossing_rate
    )
    return score, rel_width, crossing_rate


def _fit_fold_ensemble_and_score(
    feat_df, feature_cols, cat_features, params, gamma, beta_width,
    fold_dates=FOLD_DATES,
):
    """
    v6 — Gerçek Walk-Forward Cross-Validation.

    v5'in objective'i 4 fold modelini eğitip ensemble'ı SADECE Fold 4'ün
    (son hafta) validasyon penceresinde değerlendiriyordu. Bu, Optuna'nın
    hiperparametreleri tek bir haftanın gürültüsüne göre seçmesine yol
    açıyordu — gerçek run_forecast.py sonucunda görülen overfit'in
    (Train WAPE %12.6 / Test WAPE %25.7) başlıca nedeni budur.

    v6'da HER fold model, KENDİ validasyon haftasında (görmediği veri)
    değerlendirilir ve skorlar ortalanır. Eğitim maliyeti AYNI kalır —
    zaten eğitilmesi gereken 4 model bu; sadece hangi veride skorlandığı
    değişiyor. Yani bu düzeltme run süresini UZATMAZ.

    Dönüş
    -----
    models         : Eğitilen fold modelleri (final ensemble/raporlama için)
    fold_scores    : Her fold'un kendi val setinde ölçülen skoru
    crossing_rates : Her fold'daki kuantil çakışma oranı (tanı amaçlı)
    """
    models = []
    fold_scores = []
    crossing_rates = []

    for fold_name, val_start, val_end in fold_dates:
        fold_train = feat_df[feat_df[DATE_COL] < val_start]
        fold_val   = feat_df[
            (feat_df[DATE_COL] >= val_start) & (feat_df[DATE_COL] <= val_end)
        ]
        if fold_train.empty or fold_val.empty:
            logger.warning(f"   ⚠️  {fold_name}: train veya val seti boş, atlanıyor.")
            continue

        X_ft = fold_train[feature_cols]
        y_ft = fold_train[TARGET_COL]
        model = CatBoostRegressor(**params)
        model.fit(Pool(X_ft, y_ft, cat_features=cat_features), verbose=False)
        models.append(model)

        X_fv = fold_val[feature_cols]
        y_fv = fold_val[TARGET_COL].values
        raw_fv = model.predict(X_fv)

        fold_score, _, crossing = _correct_and_score(y_fv, raw_fv, gamma, beta_width)
        fold_scores.append(fold_score)
        crossing_rates.append(crossing)

    return models, fold_scores, crossing_rates


# ---------------------------------------------------------------------------
# Alpha Sweep — v8 Tanı Aracı
# ---------------------------------------------------------------------------
# HPO'nun 3 ayrı çalıştırmada alpha'yı sürekli arama sınırına yakın seçmesi
# (0.7856, 0.7663, 0.8404), "en agresif nokta en iyisi" sinyali mi yoksa
# objective'in o yöne eğimli kurulmuş olması mı belirsizdi. Tavan koyup
# tahmin etmek (örn. ALPHA_MAX=0.70) yeni bir varsayım eklemekten öteye
# gitmiyordu. Bu fonksiyon, diğer hiperparametreleri SABİT tutup sadece
# alpha'yı ince bir grid'de tarar; WAPE/Underprediction/Decision-Regret
# ödünleşimi TABLO olarak görülür — "knee" (WAPE'nin hızlı artmaya
# başladığı, regret kazancının düzleştiği nokta) gözle/sayıyla bulunur.
# Bulunan değer sonra ALPHA_MIN=ALPHA_MAX=<o değer> yapılarak sabitlenir.
# ---------------------------------------------------------------------------

def alpha_sweep(
    data_path: str,
    fixed_params: dict,
    gamma: float,
    alphas: List[float],
    hpo_folds: int = 4,
) -> List[Dict[str, Any]]:
    """
    Diğer hiperparametreleri sabit tutarak yalnızca alpha'yı tarar.

    Her alpha için 4 (veya hpo_folds) fold walk-forward CV ile eğitim
    yapılır, tüm fold'ların validasyon tahminleri BİRLEŞTİRİLİR (pool
    edilir) ve tek bir WAPE/Underprediction/rel_width/crossing hesaplanır
    — küçük fold'ların WAPE'sinin ayrı ayrı ortalanması yerine, toplam
    hacme göre ağırlıklı gerçek WAPE elde edilir (metrics.py::wape ile
    aynı mantık).

    Returns
    -------
    List[Dict] : her alpha için {alpha, wape, underpred_rate, rel_width,
                 crossing_rate, hybrid_score} — knee'yi görmek için
                 sırayla yazdırılmaya hazır.
    """
    full_df = load_data(data_path)
    n_rows  = len(full_df[full_df[TARGET_COL] > 0])
    lags    = select_lags(n_rows)

    feat_df = build_feature_matrix(
        df=full_df, target_column=TARGET_COL, date_column=DATE_COL,
        group_column=GROUP_COL, lags=lags, rolling_windows=ROLLING_WINDOWS,
        drop_na=False,
    )
    drop_cols    = [TARGET_COL, DATE_COL]
    feature_cols = [c for c in feat_df.columns if c not in drop_cols]
    cat_features = get_categorical_columns(feat_df[feature_cols])

    fold_dates = FOLD_DATES[-hpo_folds:] if 0 < hpo_folds < len(FOLD_DATES) else FOLD_DATES

    logger.info(
        f"\n🔬 Alpha Sweep başlıyor: {len(alphas)} değer × {len(fold_dates)} fold "
        f"(sabit params: {fixed_params})"
    )

    results: List[Dict[str, Any]] = []

    for alpha in alphas:
        params = {
            **fixed_params,
            "loss_function":       f"MultiQuantile:alpha=0.1,{alpha:.4f},0.9",
            "random_seed":         42,
            "verbose":             False,
            "allow_writing_files": False,
            "thread_count":        -1,
        }

        y_true_all, preds_all, q10_all, q90_all = [], [], [], []

        for fold_name, val_start, val_end in fold_dates:
            fold_train = feat_df[feat_df[DATE_COL] < val_start]
            fold_val   = feat_df[
                (feat_df[DATE_COL] >= val_start) & (feat_df[DATE_COL] <= val_end)
            ]
            if fold_train.empty or fold_val.empty:
                continue

            model = CatBoostRegressor(**params)
            model.fit(
                Pool(fold_train[feature_cols], fold_train[TARGET_COL], cat_features=cat_features),
                verbose=False,
            )
            raw = np.maximum(model.predict(fold_val[feature_cols]), 0.0)

            y_true_all.append(fold_val[TARGET_COL].values)
            preds_all.append(raw[:, 1])
            q10_all.append(raw[:, 0])
            q90_all.append(raw[:, 2])

        y_true = np.concatenate(y_true_all)
        preds  = np.concatenate(preds_all)
        q10    = np.concatenate(q10_all)
        q90    = np.concatenate(q90_all)

        q10c = np.minimum(q10, preds)
        q90c = np.maximum(q90, preds)

        sum_true = np.sum(y_true)
        wape_val = float(np.sum(np.abs(y_true - preds)) / sum_true) if sum_true > 0 else 1.0
        underpred_rate = float(np.sum(np.maximum(y_true - preds, 0.0)) / sum_true) if sum_true > 0 else 1.0
        rel_width = float(np.mean((q90c - q10c) / np.maximum(preds, 1.0)))
        crossing_rate = float(np.mean(q90 < q10))

        hybrid = (
            wape_val
            + gamma * underpred_rate
            + BETA_WIDTH * rel_width
            + CROSSING_PENALTY_WEIGHT * crossing_rate
        )

        row = {
            "alpha":          round(alpha, 4),
            "wape":           round(wape_val, 6),
            "underpred_rate": round(underpred_rate, 6),
            "rel_width":      round(rel_width, 4),
            "crossing_rate":  round(crossing_rate, 4),
            "hybrid_score":   round(hybrid, 6),
        }
        results.append(row)
        logger.info(
            f"   alpha={alpha:.2f} | WAPE={wape_val:.4%} | Underpred={underpred_rate:.4%} | "
            f"rel_width={rel_width:.3f} | crossing={crossing_rate:.2%} | hibrit={hybrid:.4%}"
        )

    return results


def print_alpha_sweep_table(results: List[Dict[str, Any]]) -> None:
    """Sweep sonuçlarını okunaklı tablo + knee önerisi olarak yazdırır."""
    print("\n" + "=" * 78)
    print("  ALPHA SWEEP SONUÇLARI")
    print("=" * 78)
    print(f"{'alpha':>7} | {'WAPE':>9} | {'Underpred':>10} | {'rel_width':>10} | {'crossing':>9} | {'hibrit':>9}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['alpha']:>7.2f} | {r['wape']:>8.2%} | {r['underpred_rate']:>9.2%} | "
            f"{r['rel_width']:>10.3f} | {r['crossing_rate']:>8.2%} | {r['hybrid_score']:>8.2%}"
        )
    print("-" * 78)

    best = min(results, key=lambda r: r["hybrid_score"])
    print(f"\n💡 Hibrit skora göre en iyi: alpha={best['alpha']:.2f}")
    print(
        "   NOT: Tabloyu gözle de inceleyin — 'knee' noktası (WAPE'nin hızla\n"
        "   artmaya başladığı, Underpred kazancının düzleştiği alpha) genelde\n"
        "   saf minimum hibrit skordan 1-2 basamak farklı olabilir ve daha\n"
        "   dengeli bir üretim seçimi olabilir (örn. hibrit min 0.84 ise ama\n"
        "   0.68'den sonra WAPE ivmeleniyorsa, 0.68-0.70 daha güvenli bir seçim)."
    )
    print("=" * 78)


# ---------------------------------------------------------------------------
# Optuna Optimizasyonu
# ---------------------------------------------------------------------------

def optimize(
    data_path: str,
    n_trials: int  = 30,
    timeout: int   = 180,
    gamma: float   = DEFAULT_GAMMA,
    gamma_note: str = f"manuel varsayılan: {DEFAULT_GAMMA}",
    beta_width: float = BETA_WIDTH,
    optuna_seed: int = 42,
    hpo_folds: int = len(FOLD_DATES),
) -> dict:
    """
    Hibrit L_hybrid = WAPE + γ × Underprediction_Penalty skorunu minimize eden
    hiperparametreleri bul. alpha (Quantile seviyesi) artık arama uzayında.

    gamma=0 → eski simetrik WAPE davranışı (geriye dönük uyumluluk).

    v5 değişiklikleri (forecasters.py ile tutarlılık için):
      1. TPESampler'a sabit seed → aynı veri + argümanlarla aynı trial dizisi
         (önceden her çalıştırmada farklı sonuç çıkmasının başlıca nedeniydi).
      2. build_feature_matrix artık drop_na=False — forecasters.py::fit() ile
         birebir aynı satır sayısı (önceden ~1.781 lag-NaN satırı burada
         siliniyor, orada silinmiyordu).
      3-4. Objective artık tek split + early_stopping yerine forecasters.py'deki
         GERÇEK 4-fold ensemble şemasını simüle ediyor; early stopping tamamen
         kaldırıldı (production'da da yok). Bu yüzden trial başına ~4x daha
         pahalı — --trials/--timeout'u buna göre ayarla. Hem tüm-gün hem
         "temiz" (anormal hafta hariç) WAPE ayrı ayrı loglanır, böylece
         run_forecast.py'nin wape_test/wape_clean çıktısıyla kıyaslanabilir.
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("🔍 Optuna Hiperparametre Optimizasyonu  (v6 — Walk-Forward CV + Pruning)")
    logger.info("=" * 60)

    if gamma > 0:
        logger.info(
            f"   γ (gamma) = {gamma:.2f}  →  spot maliyet ≈ {gamma + 1:.1f}× atıl maliyet\n"
            f"   Objective : WAPE + {gamma:.2f} × Underprediction_Penalty\n"
            f"   alpha arama aralığı: [{ALPHA_MIN}, {ALPHA_MAX}]"
        )
    else:
        logger.info("   γ = 0 → Simetrik WAPE modu (asimetrik ceza devre dışı)")

    # Veri
    full_df = load_data(data_path)
    n_rows  = len(full_df[full_df[TARGET_COL] > 0])  # sadece gerçek kayıtlar
    bucket  = _bucket_name(n_rows)
    space   = _search_space(n_rows)

    logger.info(f"\n   Gerçek kayıt: {n_rows:,} | Bucket: {bucket}")

    # Feature engineering
    # lags → veri büyüklüğüne göre uyarlanır (bkz. select_lags() ve LAG_21_MIN_ROWS/LAG_30_MIN_ROWS).
    # rolling_windows → run_forecast.py (DemandForecaster) ile birebir eşit.
    lags = select_lags(n_rows)
    logger.info(f"   Kullanılan lag'ler: {lags}  (eşikler: lag_21≥{LAG_21_MIN_ROWS:,}, lag_30≥{LAG_30_MIN_ROWS:,})")
    feat_df = build_feature_matrix(
        df             = full_df,
        target_column  = TARGET_COL,
        date_column    = DATE_COL,
        group_column   = GROUP_COL,
        lags           = lags,
        rolling_windows= ROLLING_WINDOWS,
        drop_na        = False,   # fix #2: forecasters.py::fit() ile AYNI
    )

    # Anormal haftaları tespit et (tatil/birikim → temiz-WAPE hesabında dışlanır)
    weekly_avg  = full_df[full_df[TARGET_COL] > 0].copy()
    weekly_avg["week"] = weekly_avg[DATE_COL].dt.isocalendar().week.astype(int)
    weekly_avg["year"] = weekly_avg[DATE_COL].dt.year
    wk_means    = weekly_avg.groupby(["year", "week"])[TARGET_COL].mean()
    threshold   = wk_means.mean() * 1.4
    abnormal_wk = set(wk_means[wk_means > threshold].index)
    if abnormal_wk:
        logger.info(f"⚠️  Anormal haftalar dışlandı: {abnormal_wk}")

    # Sadece hedef ve tarih düşürülür; rota/kaynak_tm/varis_tm cat_features olarak modele girer.
    drop_cols    = [TARGET_COL, DATE_COL]
    feature_cols = [c for c in feat_df.columns if c not in drop_cols]
    cat_features = get_categorical_columns(feat_df[feature_cols])

    # -----------------------------------------------------------------------
    # Final validasyon seti: forecasters.py'nin son fold'unun (Fold 4)
    # validasyon penceresi. Production'da her gelecek tarih tahmini TÜM 4 fold
    # modelinin medyanıyla üretiliyor; HPO'nun da ensemble'ı aynı şekilde bu
    # pencerede ölçmesi, üretimdeki gerçek davranışı en sadık yansıtan seçenek.
    # Validasyon HAM (kırpılmamış) kalır — gerçek dünya performansı dürüst ölçülsün.
    # -----------------------------------------------------------------------
    _, final_val_start, final_val_end = FOLD_DATES[-1]
    val_df = feat_df[
        (feat_df[DATE_COL] >= final_val_start) & (feat_df[DATE_COL] <= final_val_end)
    ].copy()
    val_df["_week"] = val_df[DATE_COL].dt.isocalendar().week.astype(int)
    val_df["_year"] = val_df[DATE_COL].dt.year
    is_abnormal = val_df.apply(lambda r: (r["_year"], r["_week"]) in abnormal_wk, axis=1)

    X_val       = val_df[feature_cols]
    y_val       = val_df[TARGET_COL].values
    clean_mask  = (~is_abnormal).values
    y_val_clean = val_df.loc[~is_abnormal, TARGET_COL].values

    # v6: --hpo-folds ile ARAMA sırasında kullanılan fold sayısı azaltılabilir
    # (hız için, örn. 2 → sadece son 2 hafta). Final "best params" raporlaması
    # yine TAM FOLD_DATES ile yapılır (production tutarlılığı korunur) —
    # sadece Optuna'nın her trial'da eğittiği model sayısı azalır.
    hpo_fold_dates = FOLD_DATES[-hpo_folds:] if 0 < hpo_folds < len(FOLD_DATES) else FOLD_DATES

    logger.info(
        f"\n📦 Fold-ensemble train havuzu: {len(feat_df):,} satır (kırpılmamış, forecasters.py ile aynı)\n"
        f"   Final validasyon (Fold 4 penceresi {final_val_start}→{final_val_end}): "
        f"{len(X_val):,} satır ({int(clean_mask.sum())} normal gün)\n"
        f"   {len(feature_cols)} feature | Kategorik: {cat_features}\n"
        f"   {n_trials} trial, {timeout}s timeout | Arama için {len(hpo_fold_dates)}/{len(FOLD_DATES)} fold "
        f"({'tümü' if len(hpo_fold_dates) == len(FOLD_DATES) else [f[0] for f in hpo_fold_dates]}) "
        f"+ MedianPruner (kötü trial'lar ilk fold(lar) sonunda kesilir)"
    )

    # -----------------------------------------------------------------------
    # Objective — Hibrit asimetrik kayıp, GERÇEK 4-fold ensemble üzerinden
    # (early_stopping YOK — production'da da yok, use_best_model=False)
    # -----------------------------------------------------------------------
    def objective(trial):
        s = space

        # alpha: Quantile seviyesi
        # v9: ALPHA_MIN==ALPHA_MAX ise (--alpha-sweep ile knee bulunup
        # sabitlendiyse) Optuna'ya boş bir boyut açtırmıyoruz, direkt sabit
        # değeri kullanıyoruz — arama artık sadece diğer 5 parametrede.
        if ALPHA_MIN == ALPHA_MAX:
            alpha = ALPHA_MIN
        elif gamma > 0:
            alpha = trial.suggest_float("alpha", ALPHA_MIN, ALPHA_MAX)
        else:
            alpha = 0.5

        params = {
            "iterations":          trial.suggest_int("iterations", s["iter"][0], s["iter"][1], step=s["iter"][2]),
            "depth":               trial.suggest_int("depth", s["depth"][0], s["depth"][1]),
            "learning_rate":       trial.suggest_float("learning_rate", s["lr"][0], s["lr"][1], log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", s["l2"][0], s["l2"][1]),
            "bagging_temperature": trial.suggest_float("bagging_temperature", s["bag"][0], s["bag"][1]),
            # forecasters.py'nin GERÇEKTEN eğittiği kayıp fonksiyonu: q10/q50/q90
            # AYNI ağaçlarda, birlikte öğreniliyor (MultiQuantile).
            "loss_function":       f"MultiQuantile:alpha=0.1,{alpha:.4f},0.9",
            "random_seed":         42,
            "verbose":             False,
            "allow_writing_files": False,
            "thread_count":        -1,
        }

        # v6: Gerçek Walk-Forward CV — 4 fold modeli eğitilir (eğitim maliyeti
        # v5 ile AYNI), ama her biri KENDİ görmediği validasyon haftasında
        # değerlendirilir (v5'te sadece Fold 4'ün haftasına bakılıyordu, bu da
        # o tek haftaya aşırı uyuma yol açıyordu). Skorlar fold'lar arası
        # ortalanır. Ayrıca her fold sonunda ara skor Optuna'ya raporlanır ki
        # kötü giden bir trial, kalan fold'lar için zaman harcamadan kesilsin
        # (pruning) — bu HIZ kazandırır, doğruluğu değiştirmez.
        models = []
        fold_scores = []
        crossing_rates = []
        for step, (fold_name, val_start, val_end) in enumerate(hpo_fold_dates):
            fold_train = feat_df[feat_df[DATE_COL] < val_start]
            fold_val   = feat_df[
                (feat_df[DATE_COL] >= val_start) & (feat_df[DATE_COL] <= val_end)
            ]
            if fold_train.empty or fold_val.empty:
                logger.warning(f"   ⚠️  {fold_name}: train veya val seti boş, atlanıyor.")
                continue

            X_ft = fold_train[feature_cols]
            y_ft = fold_train[TARGET_COL]
            fold_model = CatBoostRegressor(**params)
            fold_model.fit(Pool(X_ft, y_ft, cat_features=cat_features), verbose=False)
            models.append(fold_model)

            X_fv = fold_val[feature_cols]
            y_fv = fold_val[TARGET_COL].values
            raw_fv = fold_model.predict(X_fv)

            fold_score, _, crossing = _correct_and_score(y_fv, raw_fv, gamma, BETA_WIDTH)
            fold_scores.append(fold_score)
            crossing_rates.append(crossing)

            # Ara raporlama + pruning: kötü trial'lar tüm fold'ları beklemeden kesilir.
            trial.report(float(np.mean(fold_scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_scores:
            return 1.0  # teorik olarak olmamalı ama güvenlik için

        trial.set_user_attr("alpha", alpha)
        trial.set_user_attr("n_folds_trained", len(fold_scores))
        trial.set_user_attr("mean_crossing_rate", float(np.mean(crossing_rates)))

        # Walk-forward ortalama skor — v5'teki "sadece Fold 4" yerine
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=optuna_seed)  # fix #1: tekrarlanabilirlik
    # v6: MedianPruner — bir trial ilk fold(lar)da açıkça kötüyse kalan
    # fold'ları eğitmeden durdurulur. n_warmup_steps=1 → en az 1 fold
    # tamamlanmadan pruning yapılmaz (adil karşılaştırma için).
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=5)
    study = optuna.create_study(
        direction="minimize", study_name=f"rtopt_{bucket}_v6", sampler=sampler, pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best    = study.best_params
    elapsed = time.time() - t0

    best_alpha   = study.best_trial.user_attrs.get("alpha", 0.5)
    n_folds_used = study.best_trial.user_attrs.get("n_folds_trained", len(FOLD_DATES))

    # En iyi hiperparametrelerle fold-ensemble'ı yeniden eğit (raporlama için)
    best_params_full = {
        "iterations":          best["iterations"],
        "depth":               best["depth"],
        "learning_rate":       best["learning_rate"],
        "l2_leaf_reg":         best["l2_leaf_reg"],
        "bagging_temperature": best["bagging_temperature"],
        "loss_function":       f"MultiQuantile:alpha=0.1,{best_alpha:.4f},0.9",
        "random_seed":         42,
        "verbose":             False,
        "allow_writing_files": False,
        "thread_count":        -1,
    }
    best_models    = _fit_fold_ensemble(feat_df, feature_cols, cat_features, best_params_full)
    best_raw       = _ensemble_predict(best_models, X_val)
    best_preds     = best_raw[:, 1]
    best_q10_corr  = np.minimum(best_raw[:, 0], best_preds)
    best_q90_corr  = np.maximum(best_raw[:, 2], best_preds)
    best_rel_width = float(np.mean((best_q90_corr - best_q10_corr) / np.maximum(best_preds, 1.0)))
    best_crossing_rate = float(np.mean(best_raw[:, 2] < best_raw[:, 0]))

    pure_wape    = float(np.sum(np.abs(y_val - best_preds)) / np.sum(y_val)) if np.sum(y_val) > 0 else 1.0
    underpred_rt = float(np.sum(np.maximum(y_val - best_preds, 0)) / np.sum(y_val)) if np.sum(y_val) > 0 else 1.0

    # fix #4: hem tüm günler hem "temiz" (anormal hafta hariç) WAPE ayrı ayrı
    if clean_mask.sum() > 0 and np.sum(y_val_clean) > 0:
        preds_clean = best_preds[clean_mask]
        wape_clean  = float(np.sum(np.abs(y_val_clean - preds_clean)) / np.sum(y_val_clean))
    else:
        wape_clean = pure_wape

    logger.info(
        f"\n✅ Optimizasyon tamamlandı ({elapsed/60:.1f} dakika)\n"
        f"   Hibrit skor       : {study.best_value:.4%}  "
        f"(WAPE={pure_wape:.4%} | UnderpredRate={underpred_rt:.4%} | "
        f"BantCezası={BETA_WIDTH*best_rel_width:.4%})\n"
        f"   WAPE (tüm günler) : {pure_wape:.4%}\n"
        f"   WAPE (temiz)      : {wape_clean:.4%}   ← run_forecast.py'deki 'wape_clean' ile kıyaslanabilir\n"
        f"   alpha (kuantil)   : {best_alpha:.4f}\n"
        f"   Bant genişliği    : (q90-q10)/q_orta ort. ≈ {best_rel_width:.3f}  "
        f"(artık skora dahil - BETA_WIDTH={BETA_WIDTH}, monotonluk duzeltmeli)\n"
        f"   Kuantil çakışması : {best_crossing_rate:.2%}  (ham q90<q10 oranı - 0 ideal)\n"
        f"   iterations        : {best['iterations']}  (sabit — early stopping YOK, forecasters.py ile aynı)\n"
        f"   depth             : {best['depth']}\n"
        f"   learning_rate     : {best['learning_rate']:.6f}\n"
        f"   l2_leaf_reg       : {best['l2_leaf_reg']:.3f}\n"
        f"   bagging_temp      : {best['bagging_temperature']:.3f}\n"
        f"   Eğitilen fold sayısı: {n_folds_used}/{len(FOLD_DATES)}"
    )

    # -----------------------------------------------------------------------
    # JSON güncelle
    # iterations: artık early-stopping'in dur noktası DEĞİL — trial'ın seçtiği
    # sabit değer (forecasters.py da fold'ları sabit iterasyonla eğitiyor).
    # alpha: forecasters.py'de loss_function içinde kullanılmalıdır.
    # -----------------------------------------------------------------------
    result_entry = {
        "row_count":                 n_rows,
        "lags_used":                 lags,   # run_forecast.py'nin AYNI lag'leri kullanması gerekir — bkz. select_lags()
        "gamma":                     gamma,
        "gamma_source":              gamma_note,  # γ nereden geldi (dosya/sütun/formül) — izlenebilirlik için
        "best_hybrid_score":         round(study.best_value, 6),
        "best_wape":                 round(pure_wape, 6),
        "best_wape_clean":           round(wape_clean, 6),   # YENİ: anormal hafta hariç WAPE
        "underprediction_rate":      round(underpred_rt, 6),
        "rel_width_q90_q10":         round(best_rel_width, 4),  # v6: monotonluk duzeltmeli, artik negatif çıkmaz
        "quantile_crossing_rate":    round(best_crossing_rate, 4),  # v6: ham q90<q10 oranı - tanı amaçlı
        "optimization_time_minutes": round(elapsed / 60, 2),
        "optuna_seed":               optuna_seed,  # YENİ: tekrarlanabilirlik izi
        "params": {
            "iterations":          best["iterations"],   # ← sabit değer, early-stop noktası DEĞİL
            "alpha":               round(best_alpha, 4),
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
    parser = argparse.ArgumentParser(
        description="Hiperparametre Optimizasyonu — Hibrit Asimetrik WAPE (v4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gamma kalibrasyonu:
  --gamma verilmezse --cost-file'dan (Araç_Kapasite_Maliyet.xlsx) OTOMATİK türetilir:
    γ = (C_spot / C_atıl) - 1  (kapasite-ağırlıklı ortalama, araç tipleri üzerinden)
    Gerçek veriyle: γ ≈ 0.60  (spot araç ortalama ≈1.6× atıl/kiralık maliyeti)
  Manuel override: --gamma 1.0  (spot 2× pahalı varsayımı) gibi elle de verilebilir.
  Asimetrik ceza kapalı   →  --gamma 0
        """
    )
    parser.add_argument("--data",    default=str(_PROJECT_ROOT / "data" / "raw" / "Desi_talep.xlsx"), help="Veri dosyası")
    parser.add_argument("--trials",  type=int,   default=50,           help="Optuna trial sayısı")
    parser.add_argument("--timeout", type=int,   default=1800,
                        help="Timeout (saniye) — 0 = sınırsız.")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Optuna TPESampler seed — aynı seed + aynı veri = aynı trial dizisi (tekrarlanabilirlik).")
    parser.add_argument("--gamma",   type=float, default=None,
                        help="Asimetrik ceza katsayısı (C_spot/C_atıl - 1). "
                             "Verilmezse --cost-file'dan otomatik türetilir.")
    parser.add_argument("--cost-file", default=str(_PROJECT_ROOT / "data" / "raw" / "Araç_Kapasite_Maliyet.xlsx"),
                        help="Spot/atıl maliyet oranının türetileceği dosya (--gamma verilmezse kullanılır)")
    parser.add_argument("--hpo-folds", type=int, default=4,
                        help="v6: ARAMA sırasında kullanılan fold sayısı (1-4). VARSAYILAN 4 — "
                             "--hpo-folds 2 testinde son 2 haftaya aşırı özelleşmiş, üretimde "
                             "(tam 4-fold eğitim) genellemeyen parametreler seçildiği gözlendi. "
                             "MedianPruner zaten hız sağlıyor; folds azaltmak yerine --trials'i "
                             "düşürmeyi tercih edin. Final 'best params' raporu HER ZAMAN tam "
                             "4 fold ile hesaplanır.")

    # --- v8: Alpha Sweep tanı modu ---
    parser.add_argument("--alpha-sweep", action="store_true",
                        help="Tam HPO yerine SADECE alpha'yı tarar (diğer hiperparametreler "
                             "--sweep-* argümanlarıyla sabit tutulur). WAPE/Underprediction/ "
                             "Decision-Regret ödünleşimini tablo olarak gösterir; ALPHA_MAX gibi "
                             "bir sınırı tahmin etmek yerine gerçek 'knee' noktasını bulmak içindir.")
    parser.add_argument("--sweep-start", type=float, default=0.50, help="Alpha sweep alt sınır")
    parser.add_argument("--sweep-end",   type=float, default=0.90, help="Alpha sweep üst sınır")
    parser.add_argument("--sweep-step",  type=float, default=0.02, help="Alpha sweep adım aralığı")
    parser.add_argument("--sweep-depth",         type=int,   default=6,
                        help="Sweep sırasında sabit tutulacak depth (son iyi run'dan)")
    parser.add_argument("--sweep-iterations",    type=int,   default=450,
                        help="Sweep sırasında sabit tutulacak iterations")
    parser.add_argument("--sweep-lr",            type=float, default=0.0262,
                        help="Sweep sırasında sabit tutulacak learning_rate")
    parser.add_argument("--sweep-l2",            type=float, default=18.80,
                        help="Sweep sırasında sabit tutulacak l2_leaf_reg")
    parser.add_argument("--sweep-bagging-temp",  type=float, default=0.398,
                        help="Sweep sırasında sabit tutulacak bagging_temperature")
    args = parser.parse_args()

    timeout = args.timeout if args.timeout > 0 else None

    if args.gamma is not None:
        gamma, gamma_note = args.gamma, f"CLI --gamma ile manuel: {args.gamma}"
    else:
        gamma, gamma_note = derive_gamma_from_costs(args.cost_file, fallback=DEFAULT_GAMMA)

    if args.alpha_sweep:
        alphas = list(np.round(
            np.arange(args.sweep_start, args.sweep_end + 1e-9, args.sweep_step), 4
        ))
        fixed_params = {
            "iterations":          args.sweep_iterations,
            "depth":               args.sweep_depth,
            "learning_rate":       args.sweep_lr,
            "l2_leaf_reg":         args.sweep_l2,
            "bagging_temperature": args.sweep_bagging_temp,
        }
        sweep_results = alpha_sweep(
            args.data, fixed_params=fixed_params, gamma=gamma,
            alphas=alphas, hpo_folds=args.hpo_folds,
        )
        print_alpha_sweep_table(sweep_results)
        sys.exit(0)

    result  = optimize(
        args.data, n_trials=args.trials, timeout=timeout,
        gamma=gamma, gamma_note=gamma_note, optuna_seed=args.seed,
        hpo_folds=args.hpo_folds,
    )
    print(f"\nJSON'a yazılan parametreler:\n{json.dumps(result, indent=2)}")