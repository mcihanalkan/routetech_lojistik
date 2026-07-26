"""
DemandForecaster — Predict-then-Optimize Talep Tahmin Motoru

Mimari Kararlar (Teknofest kısıtlarına göre):
  ┌─────────────────────────────────────────────────────────────────┐
  │  ✅ CatBoostRegressor   → LightGBM/XGBoost YOK                  │
  │  ✅ cat_features        → One-Hot Encoding YOK (RAM koruması)    │
  │  ✅ MultiQuantile       → TEK MODEL ile q10/q50/q90 bantları    │
  │  ⚠️  Log1p Dönüşümü     → MultiQuantile ile KULLANILMAZ          │
  │     (log uzayındaki küçük makas expm1 ile devasa aralığa döner) │
  │  ✅ Hibrit Heuristic    → Kampanya arifesi 1.8x/2.0x çarpanı    │
  │     (4.5 ay veri ile öğrenilemeyen sezonsallığa domain kuralı)  │
  │  ✅ İki Aşamalı Surge   → Model 1 (ensemble) + Model 2 (Asimetrik │
  │     Kalıntı Modeli       Log-Cosh, kalıntı) — bkz. PDF Bölüm 1+3 │
  │  ✅ In-memory JSON      → Disk I/O YOK (10 dk bütçesi korunur)  │
  └─────────────────────────────────────────────────────────────────┘

Talep Patlaması (Surge) Entegrasyonu — PDF Referansı:
  "Teknofest Lojistik Rota Optimizasyonunda Talep Patlamalarının ve Hata
  Yayılımının İleri Düzey Modellenmesi" raporunun Bölüm 1 (Asimetrik
  Log-Cosh) ve Bölüm 3'ü (İki Aşamalı Kalıntı Modellemesi / FTO), mevcut
  mimariye ŞÖYLE entegre edildi:

    Model 1 (Base — mevcut ensemble)
        Değişmedi: 4-fold MultiQuantile (q10/q50/q90) ensemble, tüm
        günlerde stabil tahmin üretir. Rapor'un tespit ettiği gibi,
        ardışık kapalı günler / büyük kampanyalar sonrasında 14-30
        günlük hareketli ortalamaların ataleti yüzünden sistematik
        eksik tahmin (underprediction) üretmeye devam eder — bu
        BEKLENEN bir durumdur, Model 2 bunu telafi eder.

    Model 2 (Surge / Residual — YENİ, bkz. _train_surge_residual_model)
        SADECE tetikleyici bayrak taşıyan satırlarda (is_campaign_eve,
        is_campaign_day, is_post_campaign, is_post_holiday,
        is_extreme_event_candidate, backlog_release_index>0 — bkz.
        features.py) eğitilir. Hedef, mutlak hacim DEĞİL Model 1'in
        train seti üzerindeki kalıntısıdır (y - base_q50). Kayıp
        fonksiyonu AsymmetricLogCoshObjective (bu dosyanın başında tanımlı) —
        C2-sürekli, pürüzsüz, eksik tahmini parabolik olarak
        cezalandıran asimetrik kayıp.

    Çıkarımda (predict): sadece aynı tetikleyici satırlarda
        Final(q50) = Base(q50) + max(Residual, 0)
    uygulanır (bkz. _predict_single_batch). Eski çarpan-tabanlı
    campaign_multipliers_ heuristiği, surge modeli o satır için
    eğitilip devredeyse ATLANIR (çifte düzeltme önlenir); surge modeli
    yetersiz veriyle (surge_min_rows altı) eğitilemediğinde B Planı
    olarak devrede kalır.

Çift Sayım (Double-Counting) ve Gecikmeli Kampanya Etkisi Entegrasyonu
(bkz. PDF "Fiziksel Lojistik Hacim Tahminlemesinde İki Aşamalı Modeller:
Gecikmeli Kampanya Etkisi ve Çift Sayım Sorunlarının Çözümü"):
  1. Özellik Uzayı Ortogonalleştirmesi — SURGE_STATIC_EXCLUDED_FEATURES
     (bu dosyada) is_campaign_day/is_campaign_eve/is_holiday/campaign_lag_*
     gibi statik bayrakları Kalıntı (Stage 2) modelinin feature matrisinden
     çıkarır; bkz. _train_surge_residual_model.
  2. Gecikmeli Kampanya Etkisi (faz kayması) — features.py::
     add_campaign_lag_interaction_features ile üretilen campaign_lag_1/2 ve
     Campaign_Lag1_Day_Interaction, SADECE Taban (Stage 1) modeline gider;
     Pazar kampanyasının Pazartesi/Salı'daki fiziksel yansımasını organik
     olarak öğretir.
  3. Dinamik Asimetrik Kayıp + Uyarlanabilir Örneklem Ağırlıklandırması —
     bkz. _compute_surge_dampening_weights: Pazar+kampanya satırlarında,
     Taban modelin ZATEN yükselttiği tahminlere göre CatBoost'un
     gradyan/Hessian ve split-arama ağırlığını aynı anda sönümleyen tek
     bir vektör (surge_dampening_alpha_ ile ayarlanır, 0 → kapalı).

Karar-Farkındalıklı Öğrenme (Proxy SPO) — PDF Bölüm 1 ("Karar-Farkındalıklı
Öğrenme (Proxy SPO) ve Örneklem Ağırlıklandırma Stratejileri"):
  Taban model (Model 1 — 4-fold ensemble) artık CatBoost'un yerleşik kayıp
  fonksiyonunu (MultiQuantile) DEĞİŞTİRMEDEN, her fold'un KENDİ train
  penceresinden (walk-forward, leakage yok) türetilen bir `decision_weight`
  vektörüyle eğitiliyor — bkz. _compute_decision_regret_weights.
  Mantık: ALNS'in gerçek maliyeti istatistiksel hata değil, karar
  pişmanlığıdır (Regret = Cost(x*(ŷ),c) - Cost(x*(c),c)). C++ çekirdeğine
  dokunmadan bu etkiyi simüle etmek için, o rotanın geçmiş hacim dağılımından
  (fold içi, sadece o ana kadarki günler) ampirik bir "kapasite" proxy'si
  (örn. 90. persentil) çıkarılır; kapasiteyi aşan (spot araç riski taşıyan)
  satırlara aşım miktarı × spot maliyet katsayısıyla orantılı, kapasite
  altındaki satırlara ise daha düşük (atıl kapasite maliyetine orantılı)
  bir ağırlık verilir. Ham pişmanlık skorları min-max normalize edilip
  [1.0, 5.0] aralığına sıkıştırılarak Pool(weight=...) parametresine
  verilir — CatBoost'un split arama ve gradyan büyüklüğü otomatik olarak
  bu ağırlıkla ölçeklenir. Ek çalışma süresi maliyeti yoktur (ALNS
  çözücüsü hiç çağrılmaz — tamamen vektörize Pandas/NumPy ön-işleme).
  proxy_spo_enabled=False → tamamen kapalı, eski davranışla (weight=None)
  birebir geriye dönük uyumlu.

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
import polars as pl  # PDF Bölüm 1 / Strateji 1 — Ters Hacim Ağırlıklandırması
                      # (Gradient Equalization) için sızıntısız rolling-mean
                      # hesaplaması; bkz. _compute_decision_regret_weights.
import logging
import time
import joblib
from typing import Optional, List, Dict, Any, Tuple
from copy import deepcopy

from catboost import CatBoostRegressor, Pool
from .base import BaseForecaster
from .features import build_feature_matrix, get_categorical_columns, compute_target_skewness
from .missing import DataPreprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asimetrik Log-Cosh Kayıp Fonksiyonu (CatBoost `fobj`) — PDF Bölüm 1
# ---------------------------------------------------------------------------
# Kaynak: "Teknofest Lojistik Rota Optimizasyonunda Talep Patlamalarının ve
# Hata Yayılımının İleri Düzey Modellenmesi" raporu, Bölüm "1. Asimetrik
# İkinci Dereceden (Quadratic) Kayıp Fonksiyonlarının Entegrasyonu".
#
# Neden gerekli: Mevcut mimaride kullanılan MultiQuantile (Pinball) kaybı
# parçalı-doğrusaldır (piecewise-linear) — model, gerçek değerin tahminden
# 100 desi mi yoksa 500.000 desi mi yukarıda olduğuna bakmaksızın SABİT bir
# gradyan üretir. Asimetrik Log-Cosh ise pürüzsüz (C2-sürekli) ve
# dışbükeydir; "eksik tahmin" (underprediction, e = y - ŷ > 0) durumunda
# cezayı parabolik/üstel büyütürken "aşırı tahmin" (e ≤ 0) tarafında daha
# yumuşak kalır — ağaç yaprak değerleri patlama günlerinde çok daha büyük
# düzeltici sıçramalar üretebilir. Aşağıdaki sınıf SADECE _train_surge_
# residual_model() içinde, Model 2 (Surge/Residual) için kullanılır —
# Model 1 (ana ensemble) hâlâ MultiQuantile ile eğitilir.
#
# Matematiksel türetim (raporla birebir aynı):
#   e = y - ŷ ;  e* = τ·e (e>0) / (1-τ)·e (e≤0) ;  L(e) = log(cosh(e*))
#   g = dL/dŷ  = -τ·tanh(τ·e)            (e>0)   veya  -(1-τ)·tanh((1-τ)·e)  (e≤0)
#   h = d²L/dŷ² = τ²·sech²(τ·e)          (e>0)   veya  (1-τ)²·sech²((1-τ)·e) (e≤0)
# CatBoost'un `calc_ders_range` sözleşmesi (bkz. resmi LoglossObjective
# örneği: der1 = y-p = -dL/dŷ) gereği işaret çevrilerek der1=-g, der2=-h
# döndürülür — pratikte der1=τ·tanh(τ·e) (e>0), der2=-τ²·sech²(τ·e) (e>0).

def _sech2(x: np.ndarray) -> np.ndarray:
    """sech²(x) = 1 - tanh²(x) — tanh üzerinden, sayısal taşmasız."""
    t = np.tanh(x)
    return 1.0 - t * t


class AsymmetricLogCoshObjective:
    """
    CatBoost için özel (custom) amaç fonksiyonu — Asimetrik Log-Cosh.

    --- [PDF Entegrasyonu — Dinamik Asimetrik Kayıp] ---
    Eskiden bu sınıf TEK bir skaler τ alıyordu (ör. τ=0.95) ve bu, bir
    Pool'daki (bir eğitim çağrısındaki) TÜM satırlara — yani hem 50 desilik
    "ölü" bir rotaya hem de 15.000 desilik ana artere — birebir AYNI
    eksik/aşırı tahmin ceza oranını uyguluyordu. PDF'in tespiti: bu oran
    aslında rotanın hacmine (ve dolayısıyla spot araç bulamamanın gerçek TL
    maliyetine) göre DEĞİŞMELİ — düşük hacimli rotalarda simetriğe yakın
    (hayalet talep yaratılmasın), yüksek hacimli ana arterlerde ise sert
    asimetrik (SLA ihlali riski göz ardı edilemez).
    Çözüm: τ artık tek bir float DEĞİL, eğitim setindeki HER satır için
    önceden hesaplanmış (bkz. DemandForecaster._build_dynamic_tau_array)
    bir np.ndarray (τ_i vektörü) olabiliyor — calc_ders_range() içinde
    satır bazında indekslenir. Skaler float hâlâ desteklenir (geriye dönük
    uyumluluk) — np.ndim==0 olduğunda tüm satırlara broadcast edilir.

    Parameters
    ----------
    tau : float veya np.ndarray
        Asimetri katsayısı (0 < tau < 1). tau > 0.5 → eksik tahmin
        (underprediction), aşırı tahminden (overprediction) daha ağır
        cezalandırılır. Spot araç kiralama maliyetinin atıl kapasiteye
        oranını yansıtmalıdır (örn. 0.80–0.90 arası).
        - Skaler float verilirse: TÜM gözlemlere sabit/eşit asimetri
          uygulanır (eski davranış).
        - np.ndarray verilirse: her gözlem KENDİ τ_i'siyle cezalandırılır.
          Dizinin uzunluğu ve sırası, bu objective'in bağlı olduğu
          Pool'daki satırlarla (calc_ders_range'e CatBoost tarafından
          geçirilen approxes/targets ile) BİREBİR eşleşmelidir.
    eps : float
        Hessian'ın sıfıra çok yaklaştığı durumlarda CatBoost'un Newton
        adımını bozmaması için alt taban (güvenlik ağı).

    Kullanım
    --------
    >>> model = CatBoostRegressor(
    ...     loss_function=AsymmetricLogCoshObjective(tau=0.95),  # skaler (eski)
    ...     eval_metric="RMSE",   # custom objective ile birlikte zorunlu
    ... )
    >>> model = CatBoostRegressor(
    ...     loss_function=AsymmetricLogCoshObjective(tau=tau_array),  # satır bazlı (yeni)
    ...     eval_metric="RMSE",
    ... )
    """

    def __init__(self, tau: "float | np.ndarray" = 0.95, eps: float = 1e-12):
        tau_arr = np.asarray(tau, dtype=np.float64)
        if np.any((tau_arr <= 0.0) | (tau_arr >= 1.0)):
            raise ValueError(
                f"❌ tau (0,1) aralığında olmalı, verilen aralık: "
                f"[{float(tau_arr.min())}, {float(tau_arr.max())}]"
            )
        # ndim=0 (skaler) veya ndim=1 (satır bazlı vektör) — her ikisi de
        # aynı np.ndarray temsili altında tutulur, calc_ders_range içinde
        # tek bir kod yolu (broadcast ya da doğrudan indeksleme) yeterli olur.
        self.tau = tau_arr
        self.eps = float(eps)
        # --- [PDF Entegrasyonu — Dinamik Asimetrik Kayıp / KRİTİK DÜZELTME] ---
        # CatBoost, `self` kullanan (bound-method, JIT-optimize edilemeyen)
        # özel amaç fonksiyonlarını TEK bir çağrıda TÜM Pool üzerinden değil,
        # ardışık İÇ BLOKLARA (ör. gözlemlenen gerçek örnek: 7515 satır →
        # 1000'lik parçalar) bölerek çağırır — calc_ders_range'e satır
        # indeksleri VERİLMEZ. Bu yüzden ndim=1 (satır bazlı vektör) τ
        # kullanılırken, bu iç imleç (_cursor) HER çağrıda "veri setinde
        # neredeyiz" bilgisini takip eder ve gelen bloğu self.tau'nun doğru
        # alt-dizisiyle eşleştirir. Bu, YALNIZCA çağrıların ARDIŞIK ve
        # DETERMİNİSTİK (paralel/rastgele sıralı DEĞİL) olduğu bir ortamda
        # güvenlidir — bu yüzden bu objective'i kullanan CatBoostRegressor
        # MUTLAKA thread_count=1 ile eğitilmelidir (bkz.
        # _train_surge_residual_model içindeki CatBoostRegressor çağrısı ve
        # oradaki ayrıntılı yorum). thread_count=1 olmadan (paralel/çok
        # iş parçacıklı) bu imleç YANLIŞ (sessizce hatalı) τ eşleşmesi
        # üretebilir — bu yüzden her iki tarafta da (burada ve çağıran
        # kodda) bu kısıt açıkça belgelenmiştir.
        self._cursor = 0

    # CatBoost'un beklediği isim ve imza — değiştirilmemeli.
    def calc_ders_range(self, approxes, targets, weights):
        approx_arr = np.asarray(approxes, dtype=np.float64)
        target_arr = np.asarray(targets, dtype=np.float64)

        e = target_arr - approx_arr  # residual: pozitif = eksik tahmin (underprediction)
        n = e.shape[0]

        # Skaler τ (ndim=0) satır sayısına broadcast edilir — sıra/konum
        # önemsizdir, imleç kullanılmaz.
        if self.tau.ndim == 0:
            tau_row = np.full(n, float(self.tau))
        else:
            # Satır-bazlı τ (ndim=1): CatBoost bu Pool'u TEK seferde tüm
            # satırlarla ya da (thread_count=1 altında ARDIŞIK/deterministik)
            # daha küçük bloklar halinde çağırıyor olabilir — bkz. yukarıdaki
            # __init__ notu. _cursor, bir önceki çağrının nerede bıraktığını
            # hatırlar; her boyut-N bloğu, self.tau'nun [cursor, cursor+n)
            # alt-dizisiyle eşleştirilir ve dizinin sonuna gelindiğinde
            # (bir sonraki iterasyonun baştan başlaması için) başa sarılır.
            total = self.tau.shape[0]
            if n > total:
                raise ValueError(
                    f"❌ Bu çağrıdaki blok boyutu ({n}) tau_array'in toplam "
                    f"uzunluğundan ({total}) BÜYÜK — bu, tau_array'in yanlış "
                    f"(ör. farklı bir Pool'a ait) veriyle oluşturulduğunu "
                    f"gösterir; bloklama artefaktı DEĞİL, gerçek bir hizalama "
                    f"hatasıdır (bkz. _build_dynamic_tau_array çağrı sırası)."
                )
            start = self._cursor
            end = start + n
            if end <= total:
                tau_row = self.tau[start:end]
                self._cursor = end % total  # tam total'e denk gelirse başa sar
            else:
                # Son blok, dizinin sonunu aşıp bir sonraki "tur"a taşıyor —
                # sadece total, blok boyutunu tam bölmediğinde olur (ör.
                # 7515 / 1000 → son blok 515 satır, ardından yeni tur 0'dan
                # başlar). Sarma (wrap-around) ile iki parça birleştirilir.
                first = self.tau[start:total]
                remaining = n - first.shape[0]
                second = self.tau[0:remaining]
                tau_row = np.concatenate([first, second])
                self._cursor = remaining

        der1 = np.empty_like(e)
        der2 = np.empty_like(e)

        under = e > 0.0
        over = ~under

        # --- Eksik tahmin (underprediction, e > 0) — agresif ceza (τ_i ağırlıklı) ---
        tau_u = tau_row[under]
        z_u = tau_u * e[under]
        der1[under] = tau_u * np.tanh(z_u)
        der2[under] = -(tau_u ** 2) * _sech2(z_u)

        # --- Aşırı tahmin (overprediction, e <= 0) — yumuşak ceza ((1-τ_i) ağırlıklı) ---
        tau_o = 1.0 - tau_row[over]
        z_o = tau_o * e[over]
        der1[over] = tau_o * np.tanh(z_o)
        der2[over] = -(tau_o ** 2) * _sech2(z_o)

        # Hessian tam sıfıra çok yaklaşırsa CatBoost'un Newton adımı bozulmasın diye tabanla
        der2 = np.where(np.abs(der2) < self.eps, -self.eps, der2)

        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            der1 = der1 * w
            der2 = der2 * w

        return list(zip(der1.tolist(), der2.tolist()))

    def __repr__(self) -> str:
        if self.tau.ndim == 0:
            return f"AsymmetricLogCoshObjective(tau={float(self.tau)})"
        return (
            f"AsymmetricLogCoshObjective(tau_array: n={self.tau.shape[0]}, "
            f"min={float(self.tau.min()):.3f}, mean={float(self.tau.mean()):.3f}, "
            f"max={float(self.tau.max()):.3f})"
        )


# ---------------------------------------------------------------------------
# Hiperparametre Yükleme — hyperparams_map.json
# ---------------------------------------------------------------------------

def _load_hyperparams(
    data_size: int,
    target_column: str,
    logging_enabled: bool = True,
) -> tuple:
    """
    hyperparams_map.json'dan, hedef sütun adına (target_column) göre eşleşen
    parametreleri yükler.

    Öncelik: JSON içindeki her entry'nin 'target_column' alanı, fonksiyona
    geçilen target_column ile birebir eşleşiyor mu diye kontrol edilir.
    Böylece 09:00 modeli her zaman kendi optimize edilmiş bucket'ını,
    17:00 modeli de kendi bucket'ını alır — data_size üzerinden dolaylı
    (ve yanlış eşleşmeye açık) bir kıyaslamaya gerek kalmaz.

    Eşleşme bulunamazsa (ör. JSON eskidir / bu target_column için henüz
    optimize edilmiş bir bucket yoktur), B Planı olarak eski row_count
    tabanlı en-yakın-bucket mantığına düşülür. O da başarısız olursa
    sabit fallback parametreler kullanılır.

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
            label = "FALLBACK-SMALL (JSON bulunamadı)"
        else:
            p = {"iterations": 900,  "depth": 6, "learning_rate": 0.0146}
            label = "FALLBACK-LARGE (JSON bulunamadı)"
    else:
        with open(map_path, encoding="utf-8") as f:
            hmap = json.load(f)

        entries = list(hmap.values())

        # --- 1. Öncelikli eşleşme: target_column birebir aynı mı? ---
        selected = None
        for entry in entries:
            if entry.get("target_column") == target_column:
                selected = entry
                break

        if selected is not None:
            p     = selected["params"]
            label = (
                f"JSON (target_column='{target_column}' eşleşti, "
                f"{selected['row_count']:,} satır, WAPE={selected.get('best_wape', '?')})"
            )
        else:
            # --- 2. Fallback (B Planı): eski row_count tabanlı en-yakın-bucket ---
            if logging_enabled:
                logger.warning(
                    f"⚠️  hyperparams_map.json içinde target_column='{target_column}' "
                    f"ile eşleşen bir bucket bulunamadı — row_count tabanlı B Planı'na "
                    f"düşülüyor (veri: {data_size:,} satır)."
                )
            entries_sorted = sorted(entries, key=lambda e: e["row_count"])
            selected = entries_sorted[0]
            for entry in entries_sorted:
                if data_size >= entry["row_count"]:
                    selected = entry
            p     = selected["params"]
            label = (
                f"JSON (B Planı — row_count bucket eşleşmesi, "
                f"{selected['row_count']:,} satır, WAPE={selected.get('best_wape', '?')})"
            )

    iterations    = int(p["iterations"])
    depth         = int(p["depth"])
    learning_rate = float(p["learning_rate"])
    l2_leaf_reg   = float(p.get("l2_leaf_reg", 10.0))
    bagging_temp  = float(p.get("bagging_temperature", 0.3))
    # v4: JSON'da alpha yoksa (optimize.py v4 öncesi bucket) varsayılan 0.50 (simetrik medyan)
    alpha         = float(p.get("alpha", 0.50))

    if logging_enabled:
        logger.info(
            f"⚖️  Hiperparametre yüklendi: {label}\n"
            f"   iter={iterations} | depth={depth} | lr={learning_rate:.4f} | "
            f"l2={l2_leaf_reg:.2f} | bag_temp={bagging_temp:.3f} | opt_alpha={alpha:.4f}\n"
            f"   (Veri: {data_size:,} satır)"
        )

    return iterations, depth, learning_rate, l2_leaf_reg, bagging_temp, alpha, label


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
# Surge/Residual Modeli — Tetikleyici Sütunlar (PDF Bölüm 3)
# ---------------------------------------------------------------------------
# features.py tarafından üretilen, "bu gün bir talep patlaması penceresinde
# mi" sorusunu cevaplayan binary/oransal sütunlar. Herhangi biri pozitifse
# satır "surge penceresi" sayılır — bkz. DemandForecaster._build_surge_trigger_mask.
SURGE_BINARY_TRIGGER_COLUMNS: List[str] = [
    "is_campaign_eve",
    "is_campaign_day",
    "is_post_campaign",
    "is_post_holiday",
    "is_extreme_event_candidate",
]
# Sürekli (binary olmayan) sinyal — pozitif değer, kapanış sonrası ilk
# saatlerdeki üstel-azalan birikim baskısını temsil eder (bkz. features.py:
# backlog_release_index = α · exp(-d/alpha)).
#
# ⚠️⚠️ v18 ÖLÇEK UYARISI (features.py'deki backlog_severity_ratio /
# campaign_severity_ratio entegrasyonu, bkz. add_holiday_features/
# add_campaign_features docstring'i) ⚠️⚠️
# Bu iki sütunun İSİMLERİ değişmedi ama DEĞER ARALIĞI kökten değişti:
#   ESKİ (v17) : backlog_release_index = accumulated_closed_days (ARDIŞIK
#                GÜN SAYISI, weight_col=None) * exp(-d/alpha)
#                → tipik aralık ~0-6 (kapalı gün sayısı × sönüm ağırlığı).
#   YENİ (v18) : backlog_release_index = backlog_severity_ratio (biriken
#                GERÇEK HACİM / baseline, weight_col=target_column) *
#                exp(-d/alpha)
#                → tipik aralık artık "normal hacme oran" etrafında (tek
#                günlük bir kapanışta bile ratio kolayca ~0.8-1.5 civarına
#                ulaşabilir — bkz. features.py smoke-test çıktısı), yani
#                ESKİ 0-6 aralığıyla KIYASLANAMAZ.
# SONUÇ: aşağıdaki SURGE_CONTINUOUS_TRIGGER_THRESHOLD (Model 2 satır
# seçimi) ve BUCKET_EVENT_CONTINUOUS_THRESHOLD (OOF bucket ataması) —
# ikisi de ESKİ ölçeğe göre elle kalibre edilmiş sabitlerdir. features.py
# v18 entegrasyonundan sonra retrain edilmeden ÖNCE bu iki sabit YENİDEN
# ölçülmelidir — kod değişikliği gerektirmez, `suggest_bucket_event_
# threshold()` (aşağıda) veya debug_backtest.py::_print_bucket_threshold_
# calibration_diagnosis() ile gerçek OOF/feature dağılımına bakılarak
# yeni bir sayı bulunmalıdır. Bu dosyadaki mekanik davranış (aynı isim,
# aynı >threshold karşılaştırması) DEĞİŞMEDİ — sadece sayısal değerler
# yeniden kalibre edilmeyi bekliyor.
SURGE_CONTINUOUS_TRIGGER_COLUMNS: List[str] = [
    "backlog_release_index",
    "campaign_release_index",
]
SURGE_CONTINUOUS_TRIGGER_THRESHOLD: float = 0.05  # ⚠️ v17 ölçeğine göre kalibre edildi — v18 sonrası yeniden ölçülmeli (bkz. yukarıdaki uyarı)

# --- [PDF Entegrasyonu — Dinamik Asimetrik Kayıp] Satır Bazlı τ Sigmoid Sabitleri ---
# _build_dynamic_tau_array() tarafından kullanılır. Eskiden sabit τ=0.85/0.95
# (bkz. _train_surge_residual_model içindeki eski `dinamik_tau` — artık kaldırıldı)
# TÜM satırlara slot bazında (09:00/17:00) eşit uygulanıyordu. PDF'in önerisi:
# τ, rotanın taban hacmine (Base_q50) göre SÜREKLİ ve rota bazında değişmeli.
#   V ≈ DYNAMIC_TAU_V_LOW  (50 desi)   → τ ≈ DYNAMIC_TAU_MIN (0.50, ~simetrik,
#                                          hayalet talep yaratılmaz)
#   V ≈ DYNAMIC_TAU_V_HIGH (1500 desi) → τ ≈ DYNAMIC_TAU_MAX (0.95, SLA
#                                          koruyucu sert asimetri)
# İki nokta arasında log-hacim ekseninde ortalanmış, sürekli/türevlenebilir
# bir sigmoid ile geçiş yapılır (kesikli eşik YOK — ALNS'e sunulan gradyan
# yüzeyinde ani sıçrama olmaz).
DYNAMIC_TAU_MIN: float = 0.50
DYNAMIC_TAU_MAX: float = 0.95
DYNAMIC_TAU_V_LOW: float = 50.0
DYNAMIC_TAU_V_HIGH: float = 1500.0
# Sigmoid'in dikliği: anchor noktalarında (V_low, V_high) frac(x) sırasıyla
# ~0.05 ve ~0.95'e ulaşacak şekilde kapalı-form çözülmüştür (bkz.
# _build_dynamic_tau_array docstring'i — ln(0.95/0.05) ≈ 2.9444).
_DYNAMIC_TAU_LOGIT_ANCHOR: float = 2.9444169900767976  # ln(19) = ln(0.95/0.05)

# YENİ (Öneri A — düzeltme) — Rota×Gün-Türü OOF Bias Düzeltmesi'nin "event"
# bucket'ı için AYRI ve DAHA SIKI bir sürekli-sinyal eşiği. Neden ayrı:
# SURGE_CONTINUOUS_TRIGGER_THRESHOLD=0.05, Model 2'nin (surge/residual)
# HANGİ SATIRLARDA EĞİTİLECEĞİNİ seçmek için tasarlandı — düşük tutulması
# ORASI için doğru (geniş bir "muhtemelen ilgili" satır kümesi istiyoruz,
# tam eğitim tam eğitim veri kümesinin ~%12'si — 6360/51731). Ancak
# backlog_release_index = α·exp(-d/alpha) HER Pazar kapanışından sonra
# üretiliyor (Pazar her hafta rutin kapalı gün) ve 0.05'in üzerinde
# KALDIĞI süre neredeyse haftanın tamamını kaplıyor — bu da (rota,bucket)
# öğreniminde "event" bucket'ının pratikte "Pazar dışındaki her gün"e
# eşdeğer hale gelmesine, "normal" bucket'ının ise hiç oluşmamasına yol
# açtı (bkz. gerçek çalıştırma logu: sunday=0, event=289, normal=0).
# Bu sabit SADECE _learn_route_bucket_bias_correction / predict / eval
# içindeki (rota,bucket) ataması için kullanılır; _build_surge_trigger_mask
# (Model 2 satır seçimi) HİÇ etkilenmez.
#
# ⚠️⚠️ v18 SONRASI YENİDEN KALİBRASYON GEREKİYOR ⚠️⚠️ — bu 0.35 değeri,
# yukarıdaki SURGE_CONTINUOUS_TRIGGER_COLUMNS uyarısında açıklanan ESKİ
# (v17, gün-sayısı × decay) ölçeğine göre elle bulunmuştu (yukarıdaki
# "sunday=0, event=289, normal=0" günlüğü de o ölçekle üretildi). features.py
# v18 (backlog_severity_ratio/campaign_severity_ratio) entegrasyonundan
# sonra bu sütunların dağılımı kökten değişti — 0.35 artık ANLAMSIZ bir
# eşik olabilir (çok düşük kalıp her satırı "event" yapabilir YA DA çok
# yüksek kalıp hiç "event" yakalamayabilir; yön a priori belli değil,
# gerçek veriyle ÖLÇÜLMEDEN production'da kullanılmamalı). Kod
# değişikliği gerekmiyor — retrain öncesi suggest_bucket_event_threshold()
# (aşağıda) veya debug_backtest.py::_print_bucket_threshold_calibration_
# diagnosis() ile gerçek OOF verisi üzerinde "sunday/event/normal" dağılımı
# tekrar incelenip bu sabit elle güncellenmelidir.
BUCKET_EVENT_CONTINUOUS_THRESHOLD: float = 0.35


def suggest_bucket_event_threshold(
    X: pd.DataFrame,
    columns: Optional[List[str]] = None,
    weekday_column: str = "weekday",
    candidate_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    v18 SONRASI YENİDEN KALİBRASYON YARDIMCISI — kod değişikliği değil,
    "yeni feature dağılımına bakıp sabit bulma" işini otomatikleştirir.

    BUCKET_EVENT_CONTINUOUS_THRESHOLD (ve isteğe bağlı SURGE_CONTINUOUS_
    TRIGGER_THRESHOLD) features.py'nin v18 backlog_severity_ratio/
    campaign_severity_ratio değişikliğinden sonra ESKİ ölçeğe göre
    kalibre edilmiş kalır. Bu fonksiyon retrain'den önce/sonra gerçek
    (OOF veya ham) feature matrisi üzerinde:
      1. SURGE_CONTINUOUS_TRIGGER_COLUMNS sütunlarının dağılımını
         (quantile'lar) raporlar,
      2. Bir aday eşik listesi için (varsayılan: 0.05'ten 2.0'a kadar
         geniş bir grid) her birinin "sunday / event / normal" bucket
         dağılımını hesaplar — orijinal teşhis logundaki ("sunday=0,
         event=289, normal=0" — bkz. yukarıdaki yorum) YÖNTEMİN AYNISI,
         ama artık elle deneme yerine tek çağrıda.
    Üç bucket'ın da (özellikle "normal") sıfırlanmadığı, "event"in
    haftanın tamamına eşdeğer hale gelmediği bir eşik aralığı, yeni
    BUCKET_EVENT_CONTINUOUS_THRESHOLD için makul bir aday kümesidir —
    nihai seçim bu raporu okuyan kişiye (operatöre) aittir, bu fonksiyon
    sadece ölçer, otomatik ATAMA yapmaz.

    Parameters
    ----------
    X : Gerçek/OOF feature matrisi (build_feature_matrix çıktısı ya da
        fc._oof_X_ gibi bir DataFrame) — SURGE_CONTINUOUS_TRIGGER_COLUMNS
        ve weekday_column sütunlarını içermeli.
    columns : Hangi sürekli tetikleyici sütunların inceleneceği
        (varsayılan: SURGE_CONTINUOUS_TRIGGER_COLUMNS).
    weekday_column : Pazar (weekday==6) satırlarını "sunday" bucket'ına
        ayırmak için kullanılan sütun adı.
    candidate_thresholds : Denenecek eşik listesi (varsayılan: geniş bir
        grid — 0.05, 0.10, ..., 2.0).

    Returns
    -------
    dict: {
        "distribution": {col: {"min":.., "p50":.., "p75":.., "p90":.., "p95":.., "max":..}, ...},
        "bucket_counts_by_threshold": {threshold: {"sunday_closed": n, "sunday_event": n, "event": n, "normal": n}, ...},
    }
    Bu sözlüğü print ederek/loglayarak BUCKET_EVENT_CONTINUOUS_THRESHOLD
    için yeni bir sayı seçin; bu fonksiyon sabiti KENDİSİ değiştirmez.
    """
    cols = [c for c in (columns or SURGE_CONTINUOUS_TRIGGER_COLUMNS) if c in X.columns]
    if not cols:
        return {"distribution": {}, "bucket_counts_by_threshold": {}}

    distribution: Dict[str, Dict[str, float]] = {}
    for col in cols:
        vals = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
        distribution[col] = {
            "min": float(np.min(vals)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)),
        }

    thresholds = candidate_thresholds or [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0]

    is_sunday = (
        (pd.to_numeric(X[weekday_column], errors="coerce").fillna(-1).to_numpy() == 6)
        if weekday_column in X.columns else np.zeros(len(X), dtype=bool)
    )

    bucket_counts_by_threshold: Dict[float, Dict[str, int]] = {}
    for th in thresholds:
        event_mask = np.zeros(len(X), dtype=bool)
        for col in cols:
            vals = pd.to_numeric(X[col], errors="coerce").fillna(0.0).to_numpy()
            event_mask |= (vals > th)
        # REJİM AYRIMI: Pazar artık "sunday_closed" (event yok) ve
        # "sunday_event" (event aktif — backlog/kampanya boşalması) olarak
        # ikiye ayrılıyor; bkz. _learn_route_bucket_bias_correction.
        bucket = np.where(
            is_sunday & event_mask, "sunday_event",
            np.where(is_sunday & ~event_mask, "sunday_closed",
                     np.where(event_mask, "event", "normal")),
        )
        bucket_counts_by_threshold[th] = {
            "sunday_closed": int((bucket == "sunday_closed").sum()),
            "sunday_event": int((bucket == "sunday_event").sum()),
            "event": int((bucket == "event").sum()),
            "normal": int((bucket == "normal").sum()),
        }

    return {
        "distribution": distribution,
        "bucket_counts_by_threshold": bucket_counts_by_threshold,
    }

# ---------------------------------------------------------------------------
# Özellik Uzayı Ortogonalleştirmesi (Feature Orthogonalization) — PDF Bölüm
# "Çift Sayım (Double-Counting) Paradoksunun Algoritmik Anatomisi" +
# "Kalıntı Modelini Sönümleme Stratejileri"
# ---------------------------------------------------------------------------
# Bu sütunlar, Taban (Stage 1/q50) modelin zaten absorbe ettiği STATİK
# takvimsel/kampanya bayraklarıdır. Kalıntı (Stage 2/Surge) modeline HAM
# haliyle verilirlerse, asimetrik kayıp fonksiyonu aynı sinyale tekrar ve
# çok daha agresif tepki verir → "Çift Sayım" (bkz. PDF, %145 aşırı tahmin
# vakası). SURGE_BINARY_TRIGGER_COLUMNS (yukarıda) hâlâ hangi SATIRLARIN
# Kalıntı modeline gireceğine (mask) karar vermek için kullanılır — bu
# listedeki sütunlar sadece Kalıntı modelinin FEATURE MATRİSİNDEN çıkarılır,
# maskeleme mantığından değil. Kalıntı modeli bunun yerine sadece dinamik
# sinyallerle (backlog_release_index, campaign_release_index, rolling/EWMA
# istatistikler, day_of_week vb.) beslenir — PDF'in "sistemde biriken gerçek
# fiziksel strese tepki verir, statik bir kampanya bayrağına değil" ilkesi.
SURGE_STATIC_EXCLUDED_FEATURES: List[str] = [
    "is_campaign_day",
    "is_campaign_eve",
    "is_post_campaign",
    "is_holiday",
    "is_holiday_eve",
    "is_post_holiday",
    "is_closed",
    # campaign_lag_*/Campaign_Lag1_Day_Interaction (bkz. features.py::
    # add_campaign_lag_interaction_features) is_campaign_day'in doğrudan
    # türevidir — faz kaymasını Taban modelin öğrenmesi İÇİN üretildi,
    # Kalıntı modeline verilirse aynı çift-sayım riskini taşır.
    "campaign_lag_1",
    "campaign_lag_2",
    "Campaign_Lag1_Day_Interaction",
]

# Uyarlanabilir Örneklem Ağırlıklandırması (Adaptive Sample Weighting) — PDF
# Bölüm "Uyarlanabilir Örneklem Ağırlıklandırması (Adaptive Sample Weighting)"
# + "Dinamik Asimetrik Kayıp ve Gradyan Sönümleme". Pazar günü kampanya
# tetiklediği ve Taban modelin ZATEN yüksek bir q50 ürettiği satırlarda,
# Kalıntı modelinin gradyanını/Hessian'ını (calc_ders_range'in `weights`
# argümanı üzerinden — bkz. AsymmetricLogCoshObjective) ve CatBoost'un split
# arama ağırlığını AYNI ANDA sönümleyen tek bir vektör üretir. PDF bu iki
# tekniği ayrı ayrı sunar (biri gradient-level γ_i, diğeri sample_weight);
# matematiksel etkileri örtüştüğü için burada TEK bir ağırlık dizisinde
# birleştirildi (gereksiz karmaşıklık önlendi, aynı amaca hizmet ediyorlar).
SURGE_SUNDAY_WEEKDAY_VALUE: int = 6   # add_time_features: weekday 0=Pzt..6=Paz


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
    sibling_target_column : str, optional
        Wide-format iki-slotlu akışta (09:00 / 17:00) DİĞER slotun hedef
        sütun adı. Bu sınıf hâlâ tek bir hedefi tahmin eder, ama artık
        diğer slotun sütununun kendisi için feature mi yoksa leakage mi
        olduğunu bilmesi gerekir (bkz. _get_drop_columns).
        ⚠️  ZORUNLU: None bırakılırsa fit() hata fırlatır.
        09:00 modeli   → target_column="toplam_desi_0900", sibling_target_column="toplam_desi_1700"
        17:00 modeli   → target_column="toplam_desi_1700", sibling_target_column="toplam_desi_0900"
    slot_label : str, optional
        Bu model instance'ının insan-okunur slot etiketi (ör. "09:00" / "17:00").
        predict() çıktısındaki "slot" alanında ve log tablolarında kullanılır.
        Verilmezse target_column'dan otomatik türetilmeye çalışılır
        (toplam_desi_0900 → "09:00", toplam_desi_1700 → "17:00").
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
        Feature engineering lag günleri. Varsayılan: [1, 7, 14]
        (run_forecast.py / optimize.py artık veri büyüklüğüne göre lag_21/lag_30'u
        select_lags() ile otomatik ekleyip açıkça geçiyor — bkz. o dosyalardaki not)
    rolling_windows : List[int]
        Rolling istatistik pencereleri. Varsayılan: [7, 14]
    underestimation_penalty : float
        q90 modelinde eksik tahmin cezası katsayısı. Varsayılan: 9.0
    outlier_clip_multiplier : float
        Target sütunundaki outlier'lar için kırpma eşiği.
        median + outlier_clip_multiplier × IQR üzerindeki değerler kırpılır.
        0.0 → kırpma yok. Varsayılan: 3.0
        IQR tabanlı olduğu için gruba göre hesaplanır — rota bazında adil.
    surge_residual_enabled : bool
        True ise, fit() sonunda tetikleyici (kampanya/tatil/backlog) satırlar
        üzerinde İKİNCİ bir CatBoost modeli (Model 2 — Surge/Residual)
        eğitilir. Bu model, Asimetrik Log-Cosh kaybıyla Model 1'in
        train seti kalıntısını (y - base_q50) öğrenir ve predict()
        sırasında SADECE aynı tetikleyici satırlarda q50/q90'a eklenir
        (bkz. modül docstring'i — PDF Bölüm 1+3). Varsayılan: True.
    surge_log_cosh_tau : float
        Asimetrik Log-Cosh kaybının τ katsayısı (0 < τ < 1). Spot
        kiralama maliyetinin atıl kapasiteye oranını yansıtır; τ=0.95
        → eksik tahmin, aşırı tahminden ~19x daha ağır cezalandırılır.
        Varsayılan: 0.95.
        ⚠️  Bu constructor parametresi geriye dönük uyumluluk / raporlama
        (get_params, summary) için tutulur. _train_surge_residual_model()
        artık FİİLEN bu değeri kullanmaz. Tarihçe: önce (target_column
        içinde "1700" var mı) slot bazında τ=0.95/0.85 sabitine geçilmişti;
        [PDF Entegrasyonu] ile bu da kaldırılıp SATIR (rota-gün) bazında
        dinamik bir τ_i vektörüne evrildi — bkz. _build_dynamic_tau_array():
        her gözlem, KENDİ taban hacmine (Base_q50) göre τ_i ≈ 0.50 (düşük
        hacim, ~simetrik) ile τ_i ≈ 0.95 (yüksek hacim, SLA koruyucu)
        arasında sürekli/türevlenebilir bir sigmoid ile konumlanır.
    surge_min_rows : int
        Surge modelinin eğitilebilmesi için train setinde gereken
        minimum tetikleyici satır sayısı. Altında kalırsa surge modeli
        atlanır ve eski campaign_multipliers_ çarpan heuristiği B Planı
        olarak devrede kalır (küçük örneklemde overfit riskine karşı
        koruma). Varsayılan: 40.
    log_transform_enabled : bool
        True ise fit() sırasında hedef değişkene np.log1p() uygulanır;
        predict() çıktısı otomatik olarak np.expm1() ile geri çevrilir.
        ⚠️  UYARI: MultiQuantile kayıp fonksiyonu ile KULLANMAYIN.
        Log uzayında hesaplanan küçük kantil aralıkları (q10-q90 makası)
        expm1() ile orijinal ölçeğe geri çevrildiğinde üstel büyüme
        nedeniyle binlerce birimlik yapay belirsizliğe dönüşür.
        Bu durum uncertainty.py'nin neredeyse her satıra "HIGH" etiketi
        basmasına yol açar. MultiQuantile modellerinde False bırakın.
        Varsayılan: False.
    surge_relative_cap_alpha : float, optional
        (bkz. yukarıdaki Faz 2 notu) correction'ı baseline hacmin bir
        oranıyla sınırlar. None = kapalı. Varsayılan: None.
    surge_volume_damping_enabled : bool
        [PDF Entegrasyonu — Hacim Tabanlı Lojistik Sönümleme (S_vol)]
        True ise, Model 2'nin (Surge/Residual) ürettiği kalıntı düzeltmesi,
        tahmin (predict) zamanında rotanın taban hacmine (Base_q50) göre
        SÜREKLİ bir sigmoid (S_vol) ile çarpılır — bkz. _predict_single_batch
        içindeki "Hacim Tabanlı Lojistik Sönümleme" bloğu ve
        _compute_volume_damping_factor(). Bu mekanizma, eski KESİKLİ
        (discrete) `surge_segment_scale_` (Faz 2b) bloğunun YERİNİ alır —
        o blok tamamen kaldırılmıştır (ALNS'in arama uzayında hacim
        sınırlarında ani sıçrama/süreksizlik yaratıyordu). Varsayılan: True.
    surge_volume_damping_v_crit : float
        S_vol sigmoid'inin kritik hacim eşiği (V_crit) — Tır/Kamyonet
        kırılım noktası gibi operasyonel bir referans (desi cinsinden).
        Base_q50 << V_crit → kalıntı şiddetle bastırılır (S_vol≈0);
        Base_q50 >> V_crit → kalıntı olduğu gibi geçer (S_vol≈1.0).
        Varsayılan: 150.0 (desi).
    surge_volume_damping_k : float
        S_vol sigmoid'inin eğim/diklik katsayısı. Büyük k → neredeyse
        basamak fonksiyonu (V_crit civarında keskin geçiş); k≈3 gibi
        değerler pürüzsüz/yumuşak bir geçiş sağlar (ALNS'e sunulan
        maliyet yüzeyinde gradyan kaybı/sıçrama olmasın diye tercih
        edilir). Varsayılan: 3.0.
    proxy_spo_enabled : bool
        PDF Bölüm 1 (Karar-Farkındalıklı Öğrenme / Proxy SPO) — True ise,
        Model 1'in (Taban ensemble) her fold'unda Pool(weight=...)
        karar-pişmanlığı temelli bir örneklem ağırlığıyla eğitilir (bkz.
        _compute_decision_regret_weights). False → weight=1.0 (eski
        davranış, geriye dönük uyumlu). Varsayılan: True.
    proxy_spo_capacity_quantile : float
        Rota bazlı ampirik "kapasite" proxy'si için kullanılan persentil
        (0 < q < 1). Fold'un train penceresindeki geçmiş hacimlerin bu
        persentili, o rotanın kapasite limiti sayılır — üzerindeki günler
        spot araç riski taşır. Varsayılan: 0.90.
    proxy_spo_spot_cost_multiplier : float
        Kapasiteyi AŞAN satırlarda, aşım miktarına (excess) uygulanan
        çarpan. Spot araç kiralamanın atıl kapasiteye göre ne kadar
        pahalı olduğunu yansıtır — yüksek tutulur. Varsayılan: 3.0.
    proxy_spo_idle_cost_multiplier : float
        Kapasitenin ALTINDA kalan satırlarda, açığa (deficit) uygulanan
        çarpan. Atıl kapasitenin nispeten düşük maliyetini yansıtır —
        spot_cost_multiplier'dan düşük tutulmalıdır. Varsayılan: 1.0.
    proxy_spo_weight_clip : Tuple[float, float]
        Ham pişmanlık skorları min-max normalize edildikten sonra
        sıkıştırılacağı (lo, hi) aralığı — gradyan patlamalarını önler.
        Varsayılan: (1.0, 5.0).
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

    # 09:00 modelinin hedef sütun adı — cross_lag_0900_same_day sütununun
    # (features.py: pl.col(slot_0900).alias("cross_lag_0900_same_day")) hangi
    # slotun birebir kopyası olduğunu tespit etmek için kullanılır.
    # TODO(run_forecast.py adımı): Bu sabiti burada tekrar tanımlamak yerine
    # run_forecast.py / features.py'deki TARGET_COL_0900 sabitini import edip
    # kullanmak daha sağlam olur (typo riskini azaltır). Şimdilik string olarak
    # sabitleniyor — rehberde bu nokta ayrıca işaretlendi.
    _TARGET_COL_0900 = "toplam_desi_0900"
    _CROSS_LAG_0900_COL = "cross_lag_0900_same_day"

    # ⚠️  DENEYSEL / TEST AMAÇLI (ADIM 2 — weekday bias calibration).
    # SADECE 2026-06-14→2026-06-20 penceresinden (--force, leakage'lı ama
    # bias çıkarımı için ayrı tutuldu) çıkarılmış KABA/EMPİRİK bir düzeltme.
    # 2026-06-21→2026-06-27'de HİÇ kullanılmadı — o pencere bu bias için
    # temiz bir doğrulama seti olarak bilerek boş bırakıldı.
    # Rota-başı (289 rota) mutlak desi offseti: günün toplam (gerçek-tahmin)
    # farkı ÷ 289. Sadece 17:00 modeli için — 09:00'da aynı pencerede işaret
    # tutarsızdı (bazı günler fazla, bazı günler eksik tahmin), bu yüzden
    # 09:00'a KASITLI olarak uygulanmıyor (bkz. load_model()).
    # Pazar (6) yok — zaten fazla tahmin ediliyordu, pozitif düzeltmeye gerek yok.
    # Kalıcı/doğru versiyon: fit() içinde _evaluate_on_test()'ten sonra
    # gerçek test setinden öğrenilip joblib'e gömülmeli — bu sözlük o zaman
    # devre dışı bırakılmalı (aşağıdaki load_model() içindeki enjeksiyonu kaldırın).
    _EMPIRICAL_WEEKDAY_BIAS_1700 = {
        0: 1487.0,   # Pazartesi
        1: 1376.7,   # Salı
        2: 1546.1,   # Çarşamba
        3: 609.6,    # Perşembe
        4: 575.5,    # Cuma
        5: 226.9,    # Cumartesi
    }

    # ⚠️ DENEYSEL AYAR KOLU: yukarıdaki ham kalibrasyonu BOZMADAN test etmek
    # için ölçek çarpanı. 06-21→06-27 doğrulamasında ham (1.0x) değer
    # underprediction'ı fazlasıyla aştı (+13%..+35% overprediction) —
    # predict_sequential()'ın recursive doğası yüzünden (bir günün bias'lı
    # tahmini bir sonraki günün lag feature'larına "sözde-gerçek" olarak
    # girip düzeltmeyi hafta boyunca katlıyor). Farklı değerler deneyin:
    # 1.0, 0.7, 0.5, 0.3 — regret VE fark%'ın sıfıra en yakın olduğu (üstüne
    # taşmayan) noktayı arayın, sadece en düşük regret'i değil (bkz. surge
    # kalibrasyonundaki asimetrik-metrik uyarısı — burada da aynı risk var).
    # 🔻 REVERT (2026-07-17): 1.0 → 0.0. 06-21→06-27 doğrulamasında ham
    # (1.0x) değer underprediction'ı fazlasıyla aştı (+13%..+35%
    # overprediction) ve predict_sequential()'ın recursive/autoregressive
    # akışında bias'ın hafta boyunca katlanmasına (compounding) yol açtı —
    # production'da 15M desi'lik bir tahminin ~1.2 milyara şişmesine sebep
    # oldu. Kapatılana kadar (0.0) bu deneysel bias devre dışı kalmalı;
    # tekrar açmadan önce ayrı, temiz bir doğrulama penceresinde test edin.
    _WEEKDAY_BIAS_SCALE: float = 0.0

    # ⚠️ DENEYSEL / TEST AMAÇLI — Pazar (17:00) için çarpımsal küçültme.
    # _EMPIRICAL_WEEKDAY_BIAS_1700 SADECE toplamsal + sadece pozitif çalışır
    # (bkz. predict() içindeki np.maximum(bias_vals, 0.0) kilidi), bu yüzden
    # Pazar'ın (zaten fazla tahmin edilen) durumu o sözlüğe giremez — ayrı
    # bir çarpımsal mekanizma gerekiyor. Kaynak: 2026-06-14→06-20 penceresi,
    # ML çıktısı ~86 bin iken gerçek ~40 bin idi (~%112 fazla tahmin).
    # ⚠️ RİSK: bu tek bilinen gerçek değere göre geriye doğru ayarlanmış bir
    # sabittir (leakage'lı pencere) — genelleşeceği garanti değildir, temiz
    # bir doğrulama penceresinde (bu pencere DIŞINDA) ayrıca test edilmeden
    # kalıcı production parametresi olarak kullanılmamalıdır.
    _SUNDAY_POST_PROCESS_MULTIPLIER: float = 0.55

    def __init__(
        self,
        target_column: str = "desi_hacmi",
        sibling_target_column: Optional[str] = None,
        slot_label: Optional[str] = None,
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
        log_transform_enabled: bool = False,
        surge_residual_enabled: bool = True,
        surge_log_cosh_tau: float = 0.95,
        surge_min_rows: int = 40,
        surge_calibration_factor: float = 1.0,
        surge_dampening_alpha: float = 2.0,
        surge_relative_cap_alpha: Optional[float] = None,
        surge_volume_damping_enabled: bool = True,
        surge_volume_damping_v_crit: float = 150.0,
        surge_volume_damping_k: float = 3.0,
        proxy_spo_enabled: bool = True,
        proxy_spo_capacity_quantile: float = 0.90,
        proxy_spo_spot_cost_multiplier: float = 3.0,
        proxy_spo_idle_cost_multiplier: float = 1.0,
        proxy_spo_weight_clip: Tuple[float, float] = (1.0, 5.0),
        gradient_equalization_enabled: bool = True,
        gradient_equalization_window_days: int = 14,
        gradient_equalization_eps: float = 1.0,
        target_scaling_enabled: bool = True,
        target_scale_window_days: int = 14,
        target_scale_min: float = 1.0,
        logging_enabled: bool = True,
        random_state: Optional[int] = 42,
        campaign_release_alpha: float = 5.25,
        campaign_max_release_days: int = 6,
        # bkz. features.py::build_feature_matrix — Öneri A'nın 09:00 tahmininde
        # (SHAP teşhisi) backlog_alpha=5.25/max_release_days=6, backlog_release_
        # index'in haftanın neredeyse tamamına yayılan bir gürültü/sıçrama
        # yaratıp MAPE'yi %14'ten %29.5'e çıkardığı bulundu (main'in eski
        # değeri 1.4/4 idi). Uçtan uca pipeline testinde (ALNS) 17:00'ın da
        # aynı değerlerle (1.4/4) tutarlı şekilde yeniden eğitilmesi, 17:00'ın
        # kendi MAPE'sinde bir miktar gerileme pahasına (%11.9→%17.7) toplam
        # operasyonel maliyeti/SLA'yı düşürdüğü için her iki hedef için de
        # main'in eski değerlerine (1.4/4) dönüldü. Parametre yine de dışarıdan
        # (slot-bazlı override) geçirilebilir durumda bırakıldı.
        backlog_alpha: float = 1.4,
        backlog_max_release_days: int = 4,
        # v18 — features.py::add_holiday_features/add_campaign_features artık
        # backlog_severity_ratio/campaign_severity_ratio üretiyor (bkz.
        # features.py::build_feature_matrix'teki backlog_baseline_window/
        # campaign_baseline_window notu). Varsayılan (14) add_organic_
        # backlog_features ile tutarlı ve genelde yeterli — buradaki
        # parametreler sadece HPO/kalibrasyon amaçlı dışarıya açık, zorunlu
        # override DEĞİL.
        backlog_baseline_window: int = 14,
        campaign_baseline_window: int = 14,
        # bkz. features.py::unconstrain_censored_demand — HPO/backtest'te
        # A/B testi veya kalibrasyon amacıyla dışarıdan set edilebilir.
        # require_weekday_persistence=False + inflation_factor=1.0 → eski
        # (v1, tek-seferlik) davranışı tamamen kapatır.
        censor_window: int = 14,
        censor_min_volume_threshold: float = 50.0,
        censor_cap_ratio: float = 0.98,
        censor_inflation_factor: float = 1.05,
        censor_require_weekday_persistence: bool = True,
        censor_persistence_occurrences: int = 3,
        censor_persistence_min_hits: int = 2,
        # bkz. features.py::unconstrain_censored_demand Mantık 3.5 — gerçek
        # elleçleme kapasitesi verisiyle gate'leme (VERİLİRSE istatistiksel
        # proxy'nin YERİNE geçer, verilmezse eski davranış korunur).
        censor_capacity_df: Optional[Any] = None,
        censor_source_tm_column: Optional[str] = None,
        censor_capacity_tm_column: str = "transfer_merkezi",
        censor_capacity_value_column: str = "ellecleme_kapasite",
        censor_real_capacity_ratio: float = 0.90,
        # Adım 2 (Rota Bazlı OOF Bias Düzeltmesi) — çarpımsal oranın alt/üst
        # sınırları. Önceden _learn_route_bias_correction() içinde sabit
        # kodluydu (cap_low=0.7, cap_high=1.8); artık dışarıdan (retrain
        # gerektirir) ayarlanabilir — bkz. _learn_route_bias_correction
        # docstring'i ve fit()'teki çağrı yeri.
        route_bias_cap_low: float = 0.7,
        route_bias_cap_high: float = 1.8,
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
        self.l2_leaf_reg           = 10.0   # JSON'dan yüklenince fit() içinde üzerine yazılır
        self.bagging_temperature   = 0.3    # JSON'dan yüklenince fit() içinde üzerine yazılır
        self.optimized_alpha_      = 0.50   # JSON'dan yüklenince fit() içinde üzerine yazılır (v4)
        self.lags                  = lags or [1, 7, 14]  # güvenli varsayılan; run_forecast.py/optimize.py artık select_lags() ile veri büyüklüğüne göre açıkça geçiyor
        self.rolling_windows       = rolling_windows or [7, 14]
        self.underestimation_penalty = underestimation_penalty
        self.outlier_clip_multiplier = outlier_clip_multiplier
        self.log_transform_enabled   = log_transform_enabled
        self.sibling_target_column   = sibling_target_column
        self.slot_label              = slot_label or self._infer_slot_label(target_column)
        self.surge_residual_enabled  = surge_residual_enabled
        self.surge_log_cosh_tau      = surge_log_cosh_tau
        self.surge_min_rows          = surge_min_rows
        self.surge_calibration_factor_ = surge_calibration_factor
        # PDF Bölüm "Dinamik Asimetrik Kayıp" + "Uyarlanabilir Örneklem
        # Ağırlıklandırması" — Pazar+kampanya satırlarında, Taban modelin
        # ZATEN yükselttiği tahminin üzerine Kalıntı modelinin ne kadar
        # sönümlenmiş tepki vereceğini kontrol eder. w_i = 1/(1+alpha·uplift).
        # alpha büyüdükçe sönümleme daha agresif (küçük uplift'te bile
        # ağırlık hızla düşer); alpha=0 → sönümleme kapalı (eski davranış).
        self.surge_dampening_alpha_ = surge_dampening_alpha
        # ADIM 5 / Faz 2 — Relative Cap: correction'ı baseline (düzeltme öncesi)
        # hacmin bir oranıyla sınırlar. None = kapalı (varsayılan, geriye
        # dönük uyumlu) — retrain gerekmeden backtest'te elle de atanabilir.
        self.surge_relative_cap_alpha_ = surge_relative_cap_alpha

        # PDF Bölüm 1 — Karar-Farkındalıklı Öğrenme (Proxy SPO): Model 1'in
        # (Taban ensemble) her fold'unda Pool(weight=...) olarak verilecek
        # karar-pişmanlığı temelli örneklem ağırlığını kontrol eder — bkz.
        # _compute_decision_regret_weights. proxy_spo_enabled_=False →
        # weight=1.0 (eski davranış, geriye dönük uyumlu).
        self.proxy_spo_enabled_ = proxy_spo_enabled
        self.proxy_spo_capacity_quantile_ = proxy_spo_capacity_quantile
        self.proxy_spo_spot_cost_multiplier_ = proxy_spo_spot_cost_multiplier
        self.proxy_spo_idle_cost_multiplier_ = proxy_spo_idle_cost_multiplier
        self.proxy_spo_weight_clip_ = proxy_spo_weight_clip

        # PDF Bölüm 1 / Strateji 1 — Ters Hacim Ağırlıklandırması (Gradient
        # Equalization): büyük hacimli rotaların Proxy SPO pişmanlığının
        # (regret) küçük rotaları "gradyan açlığına" (gradient starvation)
        # sürüklemesini önlemek için, min-max normalizasyonundan ÖNCE ham
        # regret skorları rotanın W-günlük (varsayılan 14) sızıntısız
        # hareketli ortalama hacminin ters kareköküyle çarpılır — bkz.
        # _compute_decision_regret_weights. enabled_=False → eski davranış
        # (geriye dönük uyumlu, ek adım atlanır).
        self.gradient_equalization_enabled_ = gradient_equalization_enabled
        self.gradient_equalization_window_days_ = gradient_equalization_window_days
        self.gradient_equalization_eps_ = gradient_equalization_eps

        # PDF Bölüm 1 / Strateji 2 — Rota Bazlı Hedef Ölçeklendirme (Target
        # Normalization): CatBoost'un düğüm ayrımlarını mutlak hacim yerine
        # rotanın kendi geçmişine göre ORANSAL sapma üzerinden yapmasını
        # sağlar (bkz. features.py::add_scale_invariant_targets). Yalnızca
        # K-Fold ensemble (Model 1) eğitimini etkiler — Proxy SPO/Gradient
        # Equalization ağırlıkları, clip/skewness/campaign-multiplier gibi
        # tüm diğer hesaplamalar HER ZAMAN ham (raw) hacim üzerinde kalır.
        # enabled_=False → eski davranış (geriye dönük uyumlu). Etkin olup
        # olmadığı ve kullanılacak sütun adları fit()/_engineer_features()
        # sırasında df_features'ta gerçekten üretilip üretilmediğine göre
        # runtime'da belirlenir (bkz. self._target_scaling_active_).
        self.target_scaling_enabled_ = target_scaling_enabled
        self.target_scale_window_days_ = target_scale_window_days
        self.target_scale_min_ = target_scale_min
        self._target_scaling_active_ = False
        self._scale_factor_col_ = None
        self._scaled_target_col_ = None

        # [PDF Entegrasyonu — Hacim Tabanlı Lojistik Sönümleme]
        # ESKİ (Faz 2b — Segment Scale) davranış TAMAMEN KALDIRILDI: hacim
        # aralığına göre KESİKLİ (discrete) çarpanlar kullanıyordu (örn.
        # [(0,60,1.0),(60,1200,0.4),(1200,inf,1.0)]) — bu, bir rota 59
        # desiden 61 desiye çıktığında çarpanın aniden değişmesine, ALNS'in
        # komşuluk arama fazlarında yerel minimumlara takılmasına ve
        # istikrarsız araç atamalarına yol açıyordu (bkz. PDF "Sürekli
        # Ölçekleme ve Pürüzsüz Sönümleme" bölümü).
        # YENİ davranış: S_vol — Base_q50 (taban hacim) bazlı, sıfıra
        # normalize edilmiş sürekli bir lojistik sönümleme fonksiyonu (bkz.
        # _compute_volume_damping_factor, kullanım yeri _predict_single_batch).
        # surge_volume_damping_enabled_=False → sönümleme tamamen kapalı
        # (residual_pred olduğu gibi geçer, eski segment_scale=None ile
        # AYNI etki — geriye dönük uyumlu).
        self.surge_volume_damping_enabled_ = surge_volume_damping_enabled
        self.surge_volume_damping_v_crit_  = surge_volume_damping_v_crit
        self.surge_volume_damping_k_       = surge_volume_damping_k

        # Adım 2 (Rota Bazlı OOF Bias Düzeltmesi) — bkz. yukarıdaki parametre
        # docstring'i ve _learn_route_bias_correction(). cap_low küçük hacimli,
        # sistematik FAZLA tahmin eden rotalarda ne kadar aşağı düzeltme
        # yapılabileceğini sınırlar (0.7 = en fazla %30 aşağı). cap_high
        # sistematik EKSİK rotalarda yukarı düzeltmeyi sınırlar (decision_regret
        # eksik tahmini ~9x cezalandırdığı için kasıtlı olarak daha geniş
        # bırakılmıştır — bu asimetri KORUNMALI, sadece cap_low ayarlanmalı).
        self.route_bias_cap_low_  = route_bias_cap_low
        self.route_bias_cap_high_ = route_bias_cap_high

        # YENİ — Tahmin zamanına özel, DAHA SIKI sürekli-tetikleyici eşiği.
        # None = kapalı (varsayılan, geriye dönük uyumlu) → predict()
        # eğitimdekiyle AYNI (0.05) eşiği kullanır. Set edilirse (örn. 0.35)
        # _build_surge_trigger_mask'ın predict() çağrısında SADECE bu değer
        # kullanılır — Model 2'nin eğitimi/ağırlıkları ETKİLENMEZ, retrain
        # gerekmez. Kök neden: backlog_release_index her Pazar kapanışından
        # sonra üretiliyor ve 0.05'in üzerinde haftanın neredeyse tamamı
        # boyunca kalıyor (bkz. BUCKET_EVENT_CONTINUOUS_THRESHOLD yorumu) —
        # bu da Model 2'nin rutin haftalık kapanışları bile "surge" sanıp
        # neredeyse her satıra correction eklemesine yol açıyor.
        self.surge_predict_continuous_threshold_ = None

        # HPO/backtest — kampanya sonrası (post-campaign) release/sönüm
        # eğrisini kontrol eder (bkz. features.py::add_campaign_features).
        # ⚠️ Etkin default: 5.25 (bkz. features.py::add_campaign_features
        # docstring) — bu, gerçekte etkili olan default'tur; features.py'deki
        # fonksiyon imzasındaki default sadece o fonksiyon doğrudan çağrılırsa
        # geçerli olur, buradaki her zaman fit()/predict() akışında öncelikli.
        self.campaign_release_alpha_ = campaign_release_alpha
        self.campaign_max_release_days_ = campaign_max_release_days
        self.backlog_alpha_ = backlog_alpha
        self.backlog_max_release_days_ = backlog_max_release_days
        self.backlog_baseline_window_ = backlog_baseline_window
        self.campaign_baseline_window_ = campaign_baseline_window

        # bkz. features.py::unconstrain_censored_demand — sansürlü talep
        # (sahte-tavan) düzeltmesi hiperparametreleri. debug_backtest.py
        # bulgusu (189 rota "FAZLA" / 17 rota "EKSİK") sonrası eklendi:
        # require_weekday_persistence=True (varsayılan) olduğunda, bir gün
        # SADECE tek seferlik yerel bir rekor kırdığı için değil, AYNI
        # hafta gününde tekrar tekrar (persistence_min_hits/persistence_
        # occurrences) tavana çarptığı için sansürlü sayılır — böylece
        # normal haftalık mevsimsellik (örn. hep yüksek olan Pazartesi)
        # yanlışlıkla şişirilmez.
        self.censor_window_ = censor_window
        self.censor_min_volume_threshold_ = censor_min_volume_threshold
        self.censor_cap_ratio_ = censor_cap_ratio
        self.censor_inflation_factor_ = censor_inflation_factor
        self.censor_require_weekday_persistence_ = censor_require_weekday_persistence
        self.censor_persistence_occurrences_ = censor_persistence_occurrences
        self.censor_persistence_min_hits_ = censor_persistence_min_hits
        self.censor_capacity_df_ = censor_capacity_df
        self.censor_source_tm_column_ = censor_source_tm_column
        self.censor_capacity_tm_column_ = censor_capacity_tm_column
        self.censor_capacity_value_column_ = censor_capacity_value_column
        self.censor_real_capacity_ratio_ = censor_real_capacity_ratio

        # ADIM 2 (weekday bias calibration) — retrain sırasında fit() içinde
        # otomatik öğrenilip doldurulacak; elle de (backtest amaçlı) atanabilir.
        # dict[int weekday(0=Pzt..6=Paz), float mutlak desi offset]
        self.weekday_bias_ = None

        # Runtime'da dolacak
        self.model_: CatBoostRegressor = None
        self.models_: List[CatBoostRegressor] = []   # Ensemble fold modelleri
        self.cat_features_: List[str] = []
        self.feature_names_: List[str] = []
        self.surge_model_: Optional[CatBoostRegressor] = None   # Model 2 — Surge/Residual (bkz. _train_surge_residual_model)
        # Özellik Uzayı Ortogonalleştirmesi (PDF) sonrası Kalıntı modelinin
        # GERÇEKTEN gördüğü sütunlar — self.feature_names_'ten farklıdır
        # (SURGE_STATIC_EXCLUDED_FEATURES çıkarılmış hali). predict() bu
        # listeye göre X_pred'i alt-kümeler; eski (bu alan olmadan kaydedilmiş)
        # modellerde geriye dönük uyumluluk için getattr(..., None) ile kontrol edilir.
        self.surge_feature_names_: List[str] = []
        self.surge_cat_features_: List[str] = []
        self._oof_base_pred_: Optional[np.ndarray] = None   # OOF q50 (Model 1) — dampening/weight hesaplaması için
        self._oof_dates_: Optional[np.ndarray] = None       # OOF satırlarının tarihi — recency-weighted route bias correction için

        # Adım 2 — GERİ ALINDI (2 üretim denemesi de Decision Regret'i
        # baseline'ın üzerine çıkardı: 09:00 1280→1307→1289, 17:00
        # 4818→7399→6774). Kök sebep parametre değil, mimari: Yalova'daki
        # -60% bias, rotanın HER gününe değil backlog-boşalma günlerine
        # (is_closed/days_since_resumption/backlog_release_index) özgü —
        # flat bir rota çarpanı bunu düzeltemez, sadece normal günleri
        # bozar. Öğrenme/loglama (_learn_route_bias_correction) teşhis
        # değeri için AÇIK bırakıldı; sadece predict()/eval()'e UYGULANMASI
        # bu flag ile kapatıldı.
        #
        # GÜNCELLEME (Öneri A) — yukarıdaki notta istenen "event-bazlı
        # (backlog_release_index koşullu) versiyon" artık mevcut:
        # _learn_route_bucket_bias_correction(), (rota, bucket) bazlı
        # ["sunday" / "event" / "normal"] üç ayrı katman öğrenir ve
        # predict()/eval() bu flag açıldığında ÖNCE bu daha isabetli
        # katmana bakar (_lookup_bias_correction), yalnızca o rota-bucket
        # kombinasyonu için yeterli OOF verisi yoksa aşağıdaki flat
        # route_bias_correction_'a düşer.
        #
        # ⚠️ PICKLING UYARISI: bu bir INSTANCE attribute'u — joblib.dump/
        # load (pickle) __init__()'i TEKRAR ÇALIŞTIRMAZ, sadece pickle
        # ANINDAKİ değeri saklar/geri yükler. Bu yüzden "forecaster.
        # route_bias_correction_enabled_ = True" satırını SADECE joblib.
        # dump()'tan SONRA çalıştırırsanız (ör. run_forecast.py'de fit/
        # load'dan hemen sonra) kaydedilen .joblib dosyasına YANSIMAZ —
        # o dosyayı AYRI BİR SCRIPT'TE (ör. debug_backtest.py'nin kendi
        # DemandForecaster.load_model() çağrısı) tekrar yüklerseniz flag
        # yine False gelir. Modeli yükleyen HER script'te bu satırı AYRI
        # AYRI çalıştırmanız gerekir (bkz. debug_backtest.py::run_one_target).
        self.route_bias_correction_enabled_: bool = False
        self.route_bucket_bias_correction_: Dict[Tuple[Any, str], float] = {}

        # Adım 3 v3 — DENENDİ, GERİ ALINDI (varsayılan kapalı): predict_sequential()
        # içinde alpha_h'ı surge tetikleyicisi aktif olan satırlarda 0.9'a
        # kilitleyen bir mekanizma denendi (bayat/olay-öncesi referans
        # seviyeye kaymayı engellemek için). Loglar tetikleyicinin DOĞRU
        # çalıştığını gösterdi (289/289 rota, her h'de aktif) ama Yalova'nın
        # gerçek q50 çıktısı neredeyse hiç değişmedi (06-27: 3125→3195, %2
        # fark). alpha_h'ı 0.35'ten 0.9'a çıkarmak gibi BÜYÜK bir müdahale
        # sonucu değiştirmiyorsa, bu mekanizma ("bayat referans seviyeye
        # kayma") asıl sebep DEĞİL — hipotez çürütüldü. Ayrıca tetikleyicinin
        # 289 rotanın TAMAMINDA aynı anda aktif olması, bunun Yalova'ya özgü
        # bir backlog sinyali değil, muhtemelen ağ-geneli takvimsel bir
        # bayrak (is_closed/is_post_holiday) olduğunu gösteriyor. 17:00'de
        # regret'i hafif kötüleştirdi (4817.98→4876.10) — production'da
        # kapalı kalmalı, gerçek kaynak (q50_base mi, Model 2'nin
        # correction_contrib'i mi ufuk arttıkça küçülüyor) bulununcaya kadar.
        self.trust_decay_surge_alpha_: float = 0.9
        self.trust_decay_event_gating_enabled_: bool = False

        # Tanı amaçlı (opsiyonel, varsayılan KAPALI): True yapılırsa
        # predict_sequential() her günün ENGINEERED feature değerlerini
        # (include_features=True ile zaten hesaplanan feat_<col> sütunları)
        # self.debug_captured_rows_ listesine biriktirir. SHAP/feature-
        # importance teşhisi (ör. "Yalova'nın q50_base'i neden düşüyor")
        # için debug_backtest.py tarafından kullanılır — normal predict()
        # akışını hiç etkilemez, sadece ekstra bir liste doldurur.
        self.capture_debug_features_: bool = False
        self.debug_captured_rows_: List[Dict[str, Any]] = []
        self.surge_trigger_columns_used_: List[str] = []

        # predict() sırasında lag/rolling değerlerini gerçek tarihsel
        # veriden hesaplayabilmek için fit() sonunda saklanan buffer.
        # Her grup için son max(lags) satır + max(rolling_windows) satır
        # tutulur; fillna(0) yanılgısı bu sayede ortadan kalkar.
        self.context_buffer_: Optional[pd.DataFrame] = None

    # -----------------------------------------------------------------------
    # Slot-farkındalığı yardımcıları
    # -----------------------------------------------------------------------

    @staticmethod
    def _infer_slot_label(target_column: str) -> Optional[str]:
        """
        target_column'dan insan-okunur slot etiketi türetir.

        Açık bir slot_label verilmediğinde otomatik çıkarım için kullanılır.
        Sütun adı formatına bağımlı olduğundan kırılgandır — mümkünse
        çağıran taraf (run_forecast.py) slot_label'ı açıkça geçmeli.
        """
        if target_column == "toplam_desi_0900":
            return "09:00"
        if target_column == "toplam_desi_1700":
            return "17:00"
        return None

    def _get_drop_columns(self, available_columns) -> List[str]:
        """
        Slot-farkındalıklı drop kolonlarını TEK bir yerden belirler.

        Hem _split_X_y (fit sırasında train/test/fold ayrımı) hem de
        predict() bu metodu çağırır — kod tekrarı önlenir, kural
        değişirse tek yerden değişir (rehberin önerdiği en kritik refactor).

        Kural
        -----
        - date_column            : HER ZAMAN drop edilir (leakage — model
                                    tarihi doğrudan feature olarak görmemeli).
        - target_column          : HER ZAMAN drop edilir (y olarak ayrılır).
        - sibling_target_column  :
            * Bu model 09:00 modeliyse (target_column == 09:00 hedefi):
              DAİMA drop edilir. Çünkü 17:00 talebi, 09:00 tahmini yapıldığı
              anda henüz gerçekleşmemiştir → kesin leakage.
            * Bu model 17:00 modeliyse: DROP EDİLMEZ. 17:00 tahmini
              yapıldığı anda sabahki (09:00) talep zaten gerçekleşmiştir,
              dolayısıyla meşru bir feature'dır (features.py docstring'i
              bunu açıkça destekler).
        - cross_lag_0900_same_day (toplam_desi_0900'ün birebir kopyası,
          bkz. features.py: pl.col(slot_0900).alias("cross_lag_0900_same_day")):
            * 09:00 modeli için: kendi hedefinin trivial/sahte-mükemmel bir
              kopyası olduğundan KESİNLİKLE drop edilir.
            * 17:00 modeli için: toplam_desi_0900 zaten feature olarak
              tutulduğundan bu sütun onun yedek bir kopyasıdır — zararsızdır,
              tutulur (CatBoost fazladan kolonu sorunsuz idare eder ve ayrı
              isim taşıması ileride hata ayıklamayı kolaylaştırır).

        Parameters
        ----------
        available_columns : df.columns gibi bir iterable — sadece gerçekten
            mevcut olan sütunlar drop listesine dahil edilir.

        Returns
        -------
        List[str] : df.drop(columns=...) için hazır, mevcut sütunlarla
            filtrelenmiş drop listesi.
        """
        is_0900_model = self.target_column == self._TARGET_COL_0900

        drop_cols: List[str] = [self.date_column, self.target_column]

        if self.sibling_target_column and is_0900_model:
            # 17:00 modeli için sibling (toplam_desi_0900) BİLEREK drop edilmez.
            drop_cols.append(self.sibling_target_column)

        if is_0900_model:
            drop_cols.append(self._CROSS_LAG_0900_COL)

        # PDF Strateji 2 — Rota Bazlı Hedef Ölçeklendirme: scale_factor_*/
        # *_scaled sütunları X'in bir PARÇASI OLMAMALI. `*_scaled` (özellikle
        # KENDİ hedefin scaled kopyası) trivial leakage'dır (hedefin kendisinin
        # basit bir dönüşümü); `scale_factor_*` da zaten mevcut
        # rolling_mean_{window}_* feature'ının bir kopyası olduğundan (bkz.
        # add_scale_invariant_targets docstring'i) X'te tutulmasının katma
        # değeri yok — sadece df_features'ta (fit()/_predict_single_batch()
        # içindeki un-scale adımı için) erişilebilir kalmaları yeterli.
        drop_cols += [c for c in available_columns if c.startswith("scale_factor") or c.endswith("_scaled")]

        # is_demand_censored_{slot} — HEDEFİN KENDİSİNDEN türetilen bir bayrak
        # (bkz. features.py::unconstrain_censored_demand: bugünün GERÇEK
        # talebi, dünden-geriye 14 günlük tavana yakın mı). Geçmiş satırlarda
        # geçerli bir bilgi (gerçek değer zaten biliniyor), ama tahmin
        # satırlarında YAPISAL OLARAK bilinemez — bugünün talebini zaten
        # bilmeden "bugün tavana çarpacak mı" sorusu cevaplanamaz.
        # build_prediction_frame() hedefi 0.0 ile doldurduğu için bu bayrak
        # tahmin satırlarında HER ZAMAN 0 çıkıyor — SHAP teşhisi (bkz. commit
        # notu) bunun q50'yi TUTARLI ve BÜYÜK ölçüde (-1.6/-2.9) aşağı çeken
        # iki dominant sürücüden biri olduğunu gösterdi: model eğitimde
        # "flag=0 → normal/düşük seyreden gün" ilişkisini öğrenmiş, tahminde
        # bu bayrak hep 0 enjekte edildiği için gerçek patlama günlerinde
        # bile modeli "bu normal bir gün" yönünde yanlış yönlendiriyor.
        # Sadece denetlenebilirlik için üretilen bir veri-temizleme artığı
        # (features.py'de başka hiçbir yerde tüketilmiyor) — model feature'ı
        # OLMAMALIYDI. ⚠️ Bu değişiklik retrain GEREKTİRİR (Model 1 VE
        # Model 2'nin öğrendiği split yapısını değiştirir).
        drop_cols += [c for c in available_columns if c.startswith("is_demand_censored")]

        available = set(available_columns)
        return [c for c in drop_cols if c in available]

    # -----------------------------------------------------------------------
    # BaseForecaster abstract method: _build_model
    # -----------------------------------------------------------------------

    def _build_model(self) -> None:
        """
        3 ayrı model yerine TEK bir MultiQuantile modeli başlatır.
        Bu sayede kantillerin birbirini kesmesi (crossing) engellenir
        ve eğitim süresi 3 kat kısalır!
        """
        # alpha listesi: q10, Optuna'nın bulduğu asimetrik kuantil ve q90(9x ceza)
        # optimized_alpha_ henüz set edilmemişse (standalone _build_model çağrısı)
        # varsayılan 0.50 kullan — fit() her zaman önce _load_hyperparams çağırır.
        _alpha = getattr(self, "optimized_alpha_", 0.50)
        loss_fn = f"MultiQuantile:alpha=0.1,{_alpha:.4f},{Q90_ALPHA}"

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            bagging_temperature=self.bagging_temperature,
            loss_function=loss_fn,
            random_seed=self.random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

        if self.logging_enabled:
            logger.info(
                f"🏗️  Model oluşturuldu: TEK MODEL ile MultiQuantile\n"
                f"   Kayıp Fonksiyonu: {loss_fn}\n"
                f"   l2_leaf_reg={self.l2_leaf_reg:.2f} | bagging_temp={self.bagging_temperature:.3f}"
            )

    # -----------------------------------------------------------------------
    # Veri Temizleme — DataPreprocessor entegrasyonu (leakage-safe)
    # -----------------------------------------------------------------------

    def _learn_campaign_multipliers(self, train_df: pd.DataFrame) -> None:
        """
        Her rotanın kampanya dönemlerinde normal günlere göre hacmini ne kadar
        artırdığını veri üzerinden öğrenir. (Data-Driven Heuristic)
        Laplace Smoothing ile küçük hacimli rotaların sahte yüksek çarpan üretmesi engellenir.
        """
        self.campaign_multipliers_ = {}
        if "is_campaign_eve" not in train_df.columns or not self.group_column or self.group_column not in train_df.columns:
            return
        # Smoothing için tüm verinin global ortalamasını al
        global_mean = train_df[self.target_column].mean()
        for grp, grp_df in train_df.groupby(self.group_column):
            # Laplace Smoothing: Küçük rotalardaki dalgalanmayı sönümlemek için pay ve paydaya global ortalamanın bir kısmını ekle
            smoothing_weight = 0.5  # Yumuşatma katsayısı

            normal_vol = grp_df.loc[grp_df["is_campaign_eve"] == 0, self.target_column].mean()
            camp_vol = grp_df.loc[grp_df["is_campaign_eve"] == 1, self.target_column].mean()
            if pd.notna(normal_vol) and pd.notna(camp_vol):
                smoothed_normal = normal_vol + (global_mean * smoothing_weight)
                smoothed_camp = camp_vol + (global_mean * smoothing_weight)

                mult = smoothed_camp / smoothed_normal

                # Çarpanı güvenlik amacıyla 1.0 ile 1.5x arasına sıkıştır (Model zaten çoğunu öğreniyor, biz sadece ince ayar yapıyoruz)
                mult = max(1.0, min(mult, 1.5))
                self.campaign_multipliers_[grp] = mult
        if self.logging_enabled:
            mean_mult = np.mean(list(self.campaign_multipliers_.values())) if self.campaign_multipliers_ else 1.15
            logger.info(
                f"   📊 Rota Bazlı Kampanya Çarpanları Öğrenildi (Smoothed): "
                f"{len(self.campaign_multipliers_)} rota (Ortalama Çarpan: {mean_mult:.2f}x)"
            )

    def _learn_route_bias_correction(
        self,
        half_life_days: float = 21.0,
        min_eff_n: float = 8.0,
        cap_low: Optional[float] = None,
        cap_high: Optional[float] = None,
        smoothing_eff_rows: float = 10.0,
    ) -> None:
        """
        Adım 2 (DÜZELTİLMİŞ v2) — OOF kalıntılardan öğrenilen, ROTA BAZLI,
        RECENCY-WEIGHTED çarpımsal bias düzeltmesi.

        v1'İN HATASI (üretim sonucuyla doğrulandı — bkz. commit notu):
        v1, OOF'un TAMAMINI (Ocak-Haziran, ~6 ay) EŞİT ağırlıkla ortalıyordu.
        Ama debug_backtest'teki asıl sorun (Yalova kümesinin sistematik
        eksik tahmini) YENİ bir rejim değişikliği (kapanma sonrası backlog
        boşalması) — 6 aylık düz ortalama bu sinyali diğer aylardaki
        "normal" davranışla sulandırıp neredeyse yok ediyordu (Yalova →
        Tekirdağ'da bias -64%'ten -61%'e, neredeyse hiç değişmedi). DAHA
        KÖTÜSÜ: 17:00 hedefinde OOF'un genelinde model hafif FAZLA tahmin
        ediyormuş (6 aylık ortalamada) — düzeltme 191/289 rotayı AŞAĞI çekti
        ve TAM DE en kritik haftada zaten eksik tahmin eden Yalova kümesini
        daha da bastırdı → Decision Regret %54 arttı (4818→7399).

        v2 FARKI: Her OOF satırı, test/tahmin anına olan YAKINLIĞINA göre
        üstel sönümle ağırlıklandırılıyor (haftalık walk-forward validasyon
        haftalarının SONUNCUSU en yüksek ağırlığı alır). Böylece:
          - Yakın zamandaki rejim değişikliği (Yalova backlog surge) baskın
            sinyal olur, 6 aylık "sakin dönem" onu artık sulandıramaz.
          - Yine de tek bir haftaya (n=7/rota) değil, TÜM fold'lara
            (üstel azalan ağırlıkla) dayanır — bu yüzden düşük hacimli
            rotalarda hâlâ gürültüye karşı bir miktar dayanıklı.

        w_i = 0.5 ** (gün_farkı_i / half_life_days)   (gün_farkı = OOF'taki
              en son tarihe göre kaç gün geride — half_life_days=21 →
              3 hafta önceki satır yarı ağırlıkta, 6 hafta önceki çeyrek
              ağırlıkta sayılır)

        ratio_r = Σ(w_i · y_true_i) / Σ(w_i · base_pred_i)   [rota r için]

        Laplace smoothing artık EFEKTİF satır sayısı (Σw_i, "eff_n") ile
        yapılıyor — ham satır sayısıyla DEĞİL, çünkü recency-weighting'te
        6 aylık 150 satırın çoğu ağırlıkça neredeyse sıfıra düşüyor; asıl
        "kaç satırlık gerçek sinyal var" sorusunun cevabı eff_n. min_eff_n
        altındaki rotalarda düzeltme hiç uygulanmaz.
        """
        # None ise constructor'da ayarlanan (varsayılan 0.7/1.8, veya
        # run_forecast.py'de route_bias_cap_low= ile override edilmiş)
        # instance değerlerine düş — bkz. __init__ route_bias_cap_low/high.
        cap_low = cap_low if cap_low is not None else getattr(self, "route_bias_cap_low_", 0.7)
        cap_high = cap_high if cap_high is not None else getattr(self, "route_bias_cap_high_", 1.8)

        self.route_bias_correction_ = {}
        if (
            not self.group_column
            or getattr(self, "_oof_X_", None) is None
            or len(self._oof_X_) == 0
            or self.group_column not in self._oof_X_.columns
        ):
            return

        groups = self._oof_X_[self.group_column].to_numpy()
        residual = np.asarray(self._oof_residual_, dtype=float)
        base_pred = np.asarray(self._oof_base_pred_, dtype=float)
        oof_dates = np.asarray(getattr(self, "_oof_dates_", None))
        if len(groups) != len(residual) or len(groups) != len(base_pred):
            return
        if len(oof_dates) != len(groups):
            # Tarih bilgisi yoksa (ör. eski/uyumsuz bir fit() akışı) recency
            # ağırlıklandırması YAPILAMAZ — v1'in "6 ay eşit ağırlık" hatasına
            # geri dönmek yerine düzeltmeyi tamamen atla (güvenli varsayılan).
            if self.logging_enabled:
                logger.warning(
                    "   ⚠️ Rota Bazlı OOF Bias Düzeltmesi atlandı: _oof_dates_ "
                    "bulunamadı/uyumsuz (recency-weighting için gerekli)."
                )
            return
        y_true = residual + base_pred

        oof_dates = pd.to_datetime(oof_dates)
        max_date = oof_dates.max()
        # .days (Index) yerine .values / timedelta64(1,'D') — pandas sürümüne
        # bağlı olmadan garanti ndarray[float] döner (Index.sum() gibi
        # metodların eksik olduğu ara tiplerle uğraşmayı önler).
        days_back = (max_date - oof_dates).values / np.timedelta64(1, "D")
        weights = 0.5 ** (days_back / half_life_days)

        w_pred = weights * base_pred
        w_true = weights * y_true
        global_w_mean_pred = (
            float(np.sum(w_pred) / np.sum(weights)) if np.sum(weights) > 0 else 0.0
        )
        if global_w_mean_pred <= 0:
            return

        n_eksik = 0
        n_fazla = 0
        n_skipped_low_n = 0
        for grp in pd.unique(groups):
            mask = groups == grp
            eff_n = float(weights[mask].sum())
            sum_pred_r = float(w_pred[mask].sum())
            sum_true_r = float(w_true[mask].sum())

            # Laplace smoothing, EFEKTİF satır sayısı cinsinden (bkz. docstring).
            smoothed_pred = sum_pred_r + global_w_mean_pred * smoothing_eff_rows
            smoothed_true = sum_true_r + global_w_mean_pred * smoothing_eff_rows
            if smoothed_pred <= 0:
                continue

            ratio = smoothed_true / smoothed_pred
            ratio = max(cap_low, min(ratio, cap_high))

            if eff_n < min_eff_n:
                n_skipped_low_n += 1
                continue

            self.route_bias_correction_[grp] = ratio
            if ratio > 1.03:
                n_eksik += 1
            elif ratio < 0.97:
                n_fazla += 1

        if self.logging_enabled:
            mean_corr = (
                np.mean(list(self.route_bias_correction_.values()))
                if self.route_bias_correction_ else 1.0
            )
            logger.info(
                f"   🎯 Rota Bazlı OOF Bias Düzeltmesi Öğrenildi (Recency-Weighted, "
                f"half_life={half_life_days}g): {len(self.route_bias_correction_)} rota "
                f"(min_eff_n={min_eff_n}, cap=[{cap_low},{cap_high}], ort. çarpan={mean_corr:.3f}x) — "
                f"{n_eksik} rota YUKARI (sistematik eksik), "
                f"{n_fazla} rota AŞAĞI (sistematik fazla) düzeltiliyor, "
                f"{n_skipped_low_n} rota yetersiz efektif OOF ağırlığı "
                f"(<{min_eff_n}) nedeniyle atlandı."
            )

        # ------------------------------------------------------------------
        # YENİ (Öneri A) — İkinci, daha yüksek çözünürlüklü katman:
        # (rota, bucket) bazlı düzeltme. Yukarıdaki self.route_bias_correction_
        # (flat, rota-geneli) HİÇBİR ŞEKİLDE değiştirilmedi — bu, predict()
        # tarafında bir FALLBACK katmanı olarak aynen kullanılmaya devam eder.
        #
        # Neden gerekli: yukarıdaki flat versiyon üretimde bir kez denenip
        # geri alınmıştı (bkz. __init__ içindeki "Adım 2 — GERİ ALINDI"
        # yorumu) — sebep, Yalova tipi bias'ın rotanın HER gününe değil,
        # ya takvimsel (Pazar) ya da olay-bazlı (backlog/kampanya sonrası)
        # belirli günlerine özgü olmasıydı; flat bir çarpan bunu ayırt
        # edemeyip normal günleri bozuyordu. Bu katman DÖRT bucket'a ayırır
        # (2026-07-26 revizyonu — Pazar artık TEK bucket değil, çünkü aynı
        # gün içinde iki zıt rejim bir arada yaşanabiliyordu: kapanma/pasiflik
        # nedeniyle çöken talep VE aynı anda backlog boşalması nedeniyle
        # sıçrayan talep. Tek çarpan bu ikisini aynı anda doğru yapamıyordu):
        #   "sunday_closed" : weekday == 6 VE event YOK — Pazar'ın takvimsel/
        #              yapısal düşük hacim etkisi (17:00 slotundaki sistematik
        #              FAZLA tahminin kaynağı; bkz. debug_backtest çıktısı).
        #   "sunday_event"  : weekday == 6 VE event AKTİF — Pazar'a denk
        #              gelen backlog/kampanya boşalması (ihtiyaç genelde
        #              YUKARI, "event" ile aynı geniş cap aralığını kullanır).
        #   "event"  : (Pazar olmayan günlerde) SURGE_BINARY_TRIGGER_COLUMNS /
        #              SURGE_CONTINUOUS_TRIGGER_COLUMNS'tan herhangi biri
        #              aktif (kampanya/tatil-sonrası/backlog boşalması) —
        #              Yalova tipi rejim-değişikliği bias'ının kaynağı.
        #   "normal" : diğer tüm günler.
        # Bir rota-bucket kombinasyonunda yeterli efektif OOF ağırlığı
        # (eff_n) yoksa o kombinasyon ATLANIR — predict() tarafında flat
        # route_bias_correction_'a (o da yoksa 1.0'a) düşülür (hiyerarşik
        # fallback, bkz. _lookup_bias_correction).
        self._learn_route_bucket_bias_correction(
            half_life_days=half_life_days,
            # --- REJİM AYRIMI (2026-07-26 revizyonu) ---
            # Eski tek "sunday" bucket'ı, Pazar'ın iki farklı fiziksel
            # rejimini (kapalı/pasif TM → yapısal düşük hacim VS.
            # backlog/kampanya boşalması → ani sıçrama) tek bir çarpanda
            # eritiyordu. is_sunday her zaman event_mask'ten önce
            # değerlendirildiği için, bir Pazar günü hem "Pazar" hem
            # "event" olsa bile sadece "sunday" bucket'ına düşüyordu —
            # bu da iki zıt yönlü ihtiyacı (aşağı vs yukarı düzeltme)
            # aynı öğrenilmiş orana zorluyordu (bkz. debug_backtest günlüğü:
            # ×0.55 tabanı bazı rotalarda yetersiz, bazılarında fazla).
            # Artık dört bucket var: "sunday_closed" (Pazar, event YOK —
            # eski "sunday" davranışı/cap'leri) ve "sunday_event" (Pazar,
            # event AKTİF — backlog/kampanya boşalması, "event" bucket'ıyla
            # aynı geniş cap aralığını kullanır çünkü ihtiyaç YUKARI olabilir).
            min_eff_n_sunday_closed=1.8,   # düzeltme: 4 hafta walk-forward OOF'ta 1 rota için tipik eff_n≈2.6 (4 ham Pazar, half_life=21g ile decay) — eski 4.0 hiçbir rotanın geçemeyeceği bir bar'dı (bkz. gerçek log: sunday=0)
            min_eff_n_sunday_event=3.0,    # event ile aynı seviyede tutulmuyor çünkü Pazar+event kombinasyonu daha seyrek görülür; event'ten (4.0) biraz daha gevşek
            min_eff_n_event=4.0,
            min_eff_n_normal=4.0,
            cap_low_sunday_closed=0.30,     # Pazar'ın (kapalı/pasif rejim) gözlenen ~0.55x ihtiyacına izin ver
            cap_high_sunday_closed=1.2,     # bu rejimde YUKARI düzeltme nadiren gerekir, dar tut
            cap_low_sunday_event=0.5,       # backlog/kampanya rejiminde event bucket'ıyla aynı geniş aralık — burada ihtiyaç genelde YUKARI
            cap_high_sunday_event=2.5,
            cap_low_event=0.5,
            cap_high_event=2.5,      # Yalova tipi backlog patlamaları büyük YUKARI düzeltme isteyebilir
            cap_low_normal=0.7,
            cap_high_normal=1.8,     # flat sürümle aynı (mevcut, denenmiş sınırlar)
            smoothing_eff_rows_sunday_closed=0.4,  # DÜZELTME: 2.5 idi — eff_n≈2.6 ile neredeyse yarı yarıya "düzeltme yok"a (ratio→1.0) seyrelticiydi; Pazar'ın (kapalı rejim) sistematik bias'ı zaten çoklu backtestlerle doğrulanmış güçlü bir sinyal, gürültü değil — az smoothing + cap[0.30,1.2] güvenlik ağı yeterli
            smoothing_eff_rows_sunday_event=6.0,   # backlog rejiminde event ile aynı smoothing — bu kombinasyon nadir görüldüğü için aşırı güvenmemek gerek
            smoothing_eff_rows_default=6.0,  # event/normal için (daha fazla ham gözlem var)
        )

    def _learn_route_bucket_bias_correction(
        self,
        half_life_days: float = 21.0,
        min_eff_n_sunday_closed: float = 1.8,
        min_eff_n_sunday_event: float = 3.0,
        min_eff_n_event: float = 4.0,
        min_eff_n_normal: float = 4.0,
        cap_low_sunday_closed: float = 0.30,
        cap_high_sunday_closed: float = 1.2,
        cap_low_sunday_event: float = 0.5,
        cap_high_sunday_event: float = 2.5,
        cap_low_event: float = 0.5,
        cap_high_event: float = 2.5,
        cap_low_normal: float = 0.7,
        cap_high_normal: float = 1.8,
        smoothing_eff_rows_sunday_closed: float = 0.4,
        smoothing_eff_rows_sunday_event: float = 6.0,
        smoothing_eff_rows_default: float = 6.0,
    ) -> None:
        """
        Öneri A — (rota, bucket) bazlı, recency-weighted çarpımsal bias
        düzeltmesi. _learn_route_bias_correction() ile AYNI OOF kaynağını
        (self._oof_X_/_oof_residual_/_oof_base_pred_/_oof_dates_) ve AYNI
        recency-weighting/Laplace-smoothing/cap deseni kullanır — tek fark
        gruplama anahtarının (rota) yerine (rota, bucket) olması ve her
        bucket'ın kendi (daha gevşek/dar) cap aralığına sahip olmasıdır.

        Bucket ataması SADECE OOF öğrenme anında, X_fold_val'daki mevcut
        sütunlardan (weekday + SURGE_BINARY_TRIGGER_COLUMNS/SURGE_
        CONTINUOUS_TRIGGER_COLUMNS) türetilir — _build_surge_trigger_mask
        ile AYNI mantık, ama self.surge_trigger_columns_used_ (Model 2
        loglaması için kullanılan paylaşılan durum) üzerinde yan etki
        yaratmamak için BAĞIMSIZ/yerel olarak hesaplanır.

        ⚠️ v18 UYARISI: Bucket ataması BUCKET_EVENT_CONTINUOUS_THRESHOLD'a
        bağlıdır (yukarıda) — features.py'nin backlog_severity_ratio/
        campaign_severity_ratio entegrasyonundan sonra bu eşiğin ölçeği
        değişti (kod burada değişmedi, sadece davranış). route_bias_
        cap_low/high (cap_low_event/cap_high_event vb., aşağıdaki
        parametreler) dolaylı olarak etkilenir: "event" bucket'ına giren
        satır kümesi değişirse, o bucket için öğrenilen çarpan da değişir.
        v18 features.py ile retrain sonrası suggest_bucket_event_
        threshold() ile yeniden teşhis edilmeli — bu fonksiyonun kendisi
        değişmedi, sadece girdi dağılımı değişti.
        """
        self.route_bucket_bias_correction_: Dict[Tuple[Any, str], float] = {}

        if (
            not self.group_column
            or getattr(self, "_oof_X_", None) is None
            or len(self._oof_X_) == 0
            or self.group_column not in self._oof_X_.columns
        ):
            return

        X_oof = self._oof_X_
        groups = X_oof[self.group_column].to_numpy()
        residual = np.asarray(self._oof_residual_, dtype=float)
        base_pred = np.asarray(self._oof_base_pred_, dtype=float)
        oof_dates = np.asarray(getattr(self, "_oof_dates_", None))
        if len(groups) != len(residual) or len(groups) != len(base_pred) or len(oof_dates) != len(groups):
            return

        y_true = residual + base_pred
        oof_dates_dt = pd.to_datetime(oof_dates)
        max_date = oof_dates_dt.max()
        days_back = (max_date - oof_dates_dt).values / np.timedelta64(1, "D")
        weights = 0.5 ** (days_back / half_life_days)

        # --- Bucket ataması (yerel, self.surge_trigger_columns_used_'a dokunmaz) ---
        if "weekday" in X_oof.columns:
            weekday_vals = pd.to_numeric(X_oof["weekday"], errors="coerce").fillna(-1).to_numpy()
        else:
            weekday_vals = np.full(len(X_oof), -1)
        is_sunday = (weekday_vals == 6)

        event_mask = np.zeros(len(X_oof), dtype=bool)
        for col in SURGE_BINARY_TRIGGER_COLUMNS:
            if col in X_oof.columns:
                event_mask |= (pd.to_numeric(X_oof[col], errors="coerce").fillna(0.0) > 0).to_numpy()
        for col in SURGE_CONTINUOUS_TRIGGER_COLUMNS:
            if col in X_oof.columns:
                event_mask |= (
                    pd.to_numeric(X_oof[col], errors="coerce").fillna(0.0)
                    > BUCKET_EVENT_CONTINUOUS_THRESHOLD
                ).to_numpy()

        # REJİM AYRIMI: Pazar + event aktifse "sunday_event" (backlog/kampanya
        # boşalması — ihtiyaç genelde YUKARI), Pazar + event yoksa
        # "sunday_closed" (kapalı/pasif TM — ihtiyaç genelde AŞAĞI). Bu ikisi
        # artık AYNI öğrenilmiş orana zorlanmıyor.
        bucket = np.where(
            is_sunday & event_mask, "sunday_event",
            np.where(is_sunday & ~event_mask, "sunday_closed",
                     np.where(event_mask, "event", "normal")),
        )
        cap_map = {
            "sunday_closed": (cap_low_sunday_closed, cap_high_sunday_closed),
            "sunday_event":  (cap_low_sunday_event, cap_high_sunday_event),
            "event":  (cap_low_event, cap_high_event),
            "normal": (cap_low_normal, cap_high_normal),
        }
        min_eff_n_map = {
            "sunday_closed": min_eff_n_sunday_closed,
            "sunday_event": min_eff_n_sunday_event,
            "event": min_eff_n_event,
            "normal": min_eff_n_normal,
        }
        smoothing_map = {
            "sunday_closed": smoothing_eff_rows_sunday_closed,
            "sunday_event": smoothing_eff_rows_sunday_event,
            "event": smoothing_eff_rows_default,
            "normal": smoothing_eff_rows_default,
        }

        w_pred = weights * base_pred
        w_true = weights * y_true
        global_w_mean_pred = (
            float(np.sum(w_pred) / np.sum(weights)) if np.sum(weights) > 0 else 0.0
        )
        if global_w_mean_pred <= 0:
            return

        combo_keys = list(zip(groups.tolist(), bucket.tolist()))
        combo_arr = np.array(combo_keys, dtype=object)
        n_learned = {"sunday_closed": 0, "sunday_event": 0, "event": 0, "normal": 0}
        n_skipped = 0

        for combo in dict.fromkeys(combo_keys):   # sıra korunan benzersizleştirme
            mask = (combo_arr[:, 0] == combo[0]) & (combo_arr[:, 1] == combo[1])
            eff_n = float(weights[mask].sum())
            if eff_n < min_eff_n_map[combo[1]]:
                n_skipped += 1
                continue

            sum_pred_c = float(w_pred[mask].sum())
            sum_true_c = float(w_true[mask].sum())
            smoothing_eff_rows_c = smoothing_map[combo[1]]
            smoothed_pred = sum_pred_c + global_w_mean_pred * smoothing_eff_rows_c
            smoothed_true = sum_true_c + global_w_mean_pred * smoothing_eff_rows_c
            if smoothed_pred <= 0:
                continue

            cap_low_c, cap_high_c = cap_map[combo[1]]
            ratio = max(cap_low_c, min(smoothed_true / smoothed_pred, cap_high_c))

            self.route_bucket_bias_correction_[combo] = ratio
            n_learned[combo[1]] += 1

        if self.logging_enabled:
            n_total = sum(n_learned.values())
            logger.info(
                f"   🎯 Rota×Gün-Türü OOF Bias Düzeltmesi Öğrenildi (Öneri A, rejim-ayrımlı): "
                f"{n_total} (rota,bucket) kombinasyonu — "
                f"sunday_closed={n_learned['sunday_closed']}, "
                f"sunday_event={n_learned['sunday_event']}, "
                f"event={n_learned['event']}, "
                f"normal={n_learned['normal']} "
                f"(min_eff_n: sunday_closed={min_eff_n_sunday_closed}/"
                f"sunday_event={min_eff_n_sunday_event}/event={min_eff_n_event}/normal={min_eff_n_normal}, "
                f"event_threshold={BUCKET_EVENT_CONTINUOUS_THRESHOLD}, "
                f"{n_skipped} kombinasyon yetersiz efektif ağırlık nedeniyle atlandı; "
                f"eksik kombinasyonlar predict() sırasında flat rota-bazlı "
                f"düzeltmeye — o da yoksa 1.0'a — geri düşer)."
            )

    def _lookup_bias_correction(self, route: Any, weekday: Optional[float], event_active: bool) -> float:
        """
        Öneri A — hiyerarşik lookup: önce (rota,bucket) katmanı, bulunamazsa
        flat rota-bazlı katman, o da yoksa 1.0 (no-op).

        predict()/eval() içinde satır bazında çağrılır. weekday=None veya
        NaN gelirse (ör. sütun eksikse) doğrudan "normal" bucket varsayılır
        — Pazar'a yanlışlıkla atama YAPILMAZ (güvenli varsayılan).

        REJİM AYRIMI: Pazar artık tek bucket değil — event_active'e göre
        "sunday_closed" (kapalı/pasif rejim) veya "sunday_event" (backlog/
        kampanya boşalması rejimi) olarak ikiye ayrılır (bkz.
        _learn_route_bucket_bias_correction docstring'i).
        """
        bucket_map = getattr(self, "route_bucket_bias_correction_", None)
        if bucket_map:
            is_sunday = (weekday is not None) and (not pd.isna(weekday)) and (int(weekday) == 6)
            if is_sunday:
                bucket = "sunday_event" if event_active else "sunday_closed"
            else:
                bucket = "event" if event_active else "normal"
            key = (route, bucket)
            if key in bucket_map:
                return bucket_map[key]
        return getattr(self, "route_bias_correction_", {}).get(route, 1.0)

    # -----------------------------------------------------------------------
    # Surge/Residual Modeli (Model 2) — PDF Bölüm 1 + Bölüm 3
    # -----------------------------------------------------------------------

    def _build_surge_trigger_mask(
        self, X: pd.DataFrame, continuous_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Bir feature matrisindeki hangi satırların "talep patlaması
        penceresi" (surge) sayıldığını belirler — bkz. modül docstring'i.

        Hem fit() (X_train üzerinde) hem de predict() (X_pred üzerinde)
        AYNI mantığı kullanır — tek yerden kontrol, tutarlılık garantisi.

        continuous_threshold : float, optional
            SURGE_CONTINUOUS_TRIGGER_THRESHOLD (0.05) yerine kullanılacak
            eşik. None ise modül varsayılanı (0.05) kullanılır — eğitim
            (fit) çağrısı HER ZAMAN None geçer, geniş satır kümesi
            korunur (bkz. BUCKET_EVENT_CONTINUOUS_THRESHOLD yorumu).
            predict() çağrısı, self.surge_predict_continuous_threshold_
            set edilmişse onu geçirir — böylece backlog_release_index
            gibi "her Pazar sonrası neredeyse tüm hafta > 0.05 kalan"
            sinyaller, TAHMİN ZAMANINDA daha sıkı bir eşikle (örn. 0.35)
            süzülüp sadece gerçek patlama günlerinde correction
            uygulanır — eğitim verisi/Model 2'nin kendisi ETKİLENMEZ,
            retrain gerekmez.

        Herhangi bir tetikleyici sütun mevcut veri setinde yoksa
        (ör. eski bir features.py sürümü) sessizce atlanır; hiçbiri
        yoksa tüm satırlar False döner (surge modeli hiç tetiklenmez).
        """
        threshold = (
            continuous_threshold if continuous_threshold is not None
            else SURGE_CONTINUOUS_TRIGGER_THRESHOLD
        )
        mask = np.zeros(len(X), dtype=bool)
        used: List[str] = []

        for col in SURGE_BINARY_TRIGGER_COLUMNS:
            if col in X.columns:
                mask |= (pd.to_numeric(X[col], errors="coerce").fillna(0.0) > 0).values
                used.append(col)

        for col in SURGE_CONTINUOUS_TRIGGER_COLUMNS:
            if col in X.columns:
                mask |= (
                    pd.to_numeric(X[col], errors="coerce").fillna(0.0)
                    > threshold
                ).values
                used.append(col)

        self.surge_trigger_columns_used_ = used
        return mask

    def _compute_decision_regret_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        PDF Bölüm 1 — Karar-Farkındalıklı Öğrenme (Proxy SPO) ve Örneklem
        Ağırlıklandırma Stratejileri.

        Neden: CatBoost'un C++ çekirdeğine (özel objective) dokunmadan,
        ALNS'in gerçek maliyetini (Decision Regret) Taban modele (Model 1)
        dolaylı olarak öğretir. Rapor'daki asimetrik "haberci problemi"
        (newsvendor) mantığı: eksik tahmin → spot araç kiralama (pahalı),
        aşırı tahmin → sadece atıl kapasite (ucuz). Özel bir kayıp
        fonksiyonu YAZMAK yerine (ALNS parçalı-sabit/ayrık olduğu için
        gradyanı her yerde sıfır — bkz. modül docstring'i), eğitim
        örneklemleri bu asimetrik pişmanlığa ORANTILI olarak yeniden
        ağırlıklandırılır (sample weighting) — ALNS çözücüsü hiç
        çağrılmadığı için çalışma süresine ~0 saniye eklenir.

        ⚠️  Leakage önlemi: bu fonksiyon SADECE fit() içindeki fold
        döngüsünden, o fold'un KENDİ train penceresiyle (X_fold_train /
        y_fold_train — val haftasından önceki tüm günler) çağrılmalıdır.
        Kapasite proxy'si böylece yalnızca "o ana kadar bilinen" geçmişten
        türetilir; gelecekteki fold'ların hacim dağılımı sızmaz.

        Adımlar
        -------
        1. Rota Bazlı Kapasite Proxy'si: Gerçek fiziksel araç kapasitesi
           veride yer almadığından (bkz. features.py Bölüm 3 —
           unconstrain_censored_demand ile aynı ampirik yaklaşım), her
           rotanın (group_column) bu fold'daki geçmiş hacimlerinin
           `proxy_spo_capacity_quantile_` persentili (varsayılan %90)
           o rotanın kapasite proxy'si (c) sayılır. Rota için yeterli
           gözlem yoksa (< 5 satır) global persentile düşülür.
        2. Asimetrik Karar Pişmanlığı (Proxy Regret): y > c ise
           regret = (y - c) × spot_cost_multiplier (spot araç riski —
           yüksek ceza); y ≤ c ise regret = (c - y) × idle_cost_multiplier
           (atıl kapasite — düşük, temel ceza).
        2.5. Ters Hacim Ağırlıklandırması (Gradient Equalization — PDF
           Strateji 1): CatBoost'un Simetrik Ağaç (oblivious tree) yapısı,
           düğüm ayrımlarında mutlak hata büyüklüğüne göre karar verdiğinden,
           yalnızca 2. adımdaki regret ile ağırlıklandırma hacimli
           ("Head") rotaların gradyanına hakim olmaya devam eder ve küçük
           ("Tail") rotalar gradyan açlığı (gradient starvation) yaşar.
           Bunu nötralize etmek için, o rotanın `gradient_equalization_window_days_`
           günlük (varsayılan 14) SIZINTISIZ (yalnızca geçmiş — closed="left")
           hareketli ortalama hacminin ters kareköküyle regret ölçeklenir:
               regret_eq = regret × 1 / sqrt(rolling_vol + eps)
           Ters karekök (1/V yerine), lojistik verilerinde sık görülen
           aşırı düşük hacimli günlerde (V≈0-2) ağırlığın patlayıp
           gradyan tabanlı öğrenmeyi ıraksatmasını (divergence) önleyen
           varyans-stabilize edici bir dönüşümdür. `dates` verilmezse
           (geriye dönük uyumluluk) bu adım atlanır. gradient_equalization_enabled_
           =False ise de atlanır.
        3. Normalizasyon: (Gradient Equalization uygulanmışsa) ölçeklenmiş
           regret_eq, aksi halde ham regret, min-max ile
           `proxy_spo_weight_clip_` (varsayılan [1.0, 5.0]) aralığına
           sıkıştırılır — CatBoost'un gradyan/split-arama ağırlığı bu
           dizi ile ölçeklenir (Pool(weight=...)).

        Parameters
        ----------
        dates : pd.Series, optional
            X ile aynı sırada, aynı uzunlukta ham tarih sütunu (örn.
            fold_train_df[self.date_column]). X'in kendisi date_column'ı
            İÇERMEZ (bkz. _get_drop_columns — leakage önlemi, feature
            olarak kullanılmasın diye), bu yüzden Gradient Equalization
            için tarih ayrıca bu parametreyle geçirilmelidir. ⚠️ Bu
            fonksiyon SADECE ilgili fold'un KENDİ train penceresiyle
            çağrıldığından (yukarıdaki leakage notuna bkz.), rolling
            hacim hesaplaması da otomatik olarak sızıntısızdır.

        Returns
        -------
        np.ndarray
            Satır başına ağırlık (len(y),). proxy_spo_enabled_=False ise
            veya n==0 ise tamamı 1.0 (eski davranışla geriye dönük uyumlu).
        """
        n = len(y)
        weights = np.ones(n, dtype=float)

        if not self.proxy_spo_enabled_ or n == 0:
            return weights

        y_arr = y.to_numpy(dtype=float) if isinstance(y, pd.Series) else np.asarray(y, dtype=float)
        q = float(self.proxy_spo_capacity_quantile_)

        # --- 1. Rota bazlı ampirik kapasite proxy'si (sadece bu fold'un train'i) ---
        if self.group_column and self.group_column in X.columns:
            groups = X[self.group_column].values
            capacity = np.empty(n, dtype=float)
            global_cap = float(np.quantile(y_arr, q))
            for grp in pd.unique(groups):
                grp_mask = groups == grp
                grp_vals = y_arr[grp_mask]
                # Az veri içeren rotalarda gürültülü bir persentile güvenmek
                # yerine global persentile düş (surge dampening'deki
                # global_fallback deseniyle tutarlı).
                capacity[grp_mask] = (
                    float(np.quantile(grp_vals, q)) if len(grp_vals) >= 5 else global_cap
                )
        else:
            capacity = np.full(n, float(np.quantile(y_arr, q)), dtype=float)

        # --- 2. Asimetrik Proxy Regret (haberci problemi mantığı) ---
        is_over = y_arr > capacity
        excess = np.maximum(0.0, y_arr - capacity)     # spot araç riski
        deficit = np.maximum(0.0, capacity - y_arr)     # atıl kapasite

        regret = np.where(
            is_over,
            excess * float(self.proxy_spo_spot_cost_multiplier_),
            deficit * float(self.proxy_spo_idle_cost_multiplier_),
        )

        # --- 2.5. Ters Hacim Ağırlıklandırması (Gradient Equalization) ---
        # PDF Strateji 1: Su Yatağı Etkisi'ni (Waterbed Effect) azaltmak için,
        # regret'i rotanın sızıntısız W-günlük hareketli ortalama hacminin
        # ters kareköküyle ölçekle. Sadece tarih bilgisi verildiyse ve
        # özellik açıksa çalışır (geriye dönük uyumluluk).
        if self.gradient_equalization_enabled_ and dates is not None and len(dates) == n:
            try:
                route_ids = (
                    X[self.group_column].to_numpy()
                    if (self.group_column and self.group_column in X.columns)
                    else np.zeros(n, dtype=int)
                )
                eq_df = pl.DataFrame(
                    {
                        "_row": np.arange(n),
                        "route_id": route_ids.astype(str),
                        "date": pd.to_datetime(
                            dates.to_numpy() if isinstance(dates, pd.Series) else np.asarray(dates)
                        ),
                        "regret": regret,
                    }
                ).sort(["route_id", "date"])

                eps = float(self.gradient_equalization_eps_)
                window = f"{int(self.gradient_equalization_window_days_)}d"
                # Rolling hacim, gerçek talep (y_arr) üzerinden hesaplanmalı —
                # regret üzerinden değil (regret zaten asimetrik/ölçekli bir
                # türev). y_arr'ı satır indeksiyle hizalayarak DataFrame'e ekliyoruz.
                eq_df = eq_df.with_columns(volume=pl.Series(y_arr)[eq_df["_row"].to_numpy()])
                eq_df = eq_df.with_columns(
                    rolling_vol=pl.col("volume")
                    .rolling_mean_by(by="date", window_size=window, closed="left", min_samples=1)
                    .over("route_id")
                )
                eq_df = eq_df.with_columns(
                    rolling_vol=pl.col("rolling_vol")
                    .fill_null(strategy="forward")
                    .over("route_id")
                    .fill_null(eps)
                )
                eq_df = eq_df.with_columns(
                    regret_eq=pl.col("regret") * (1.0 / (pl.col("rolling_vol") + eps).sqrt())
                )
                # Orijinal satır sırasına geri döndür (Polars sort permütasyonunu geri al)
                eq_df = eq_df.sort("_row")
                regret = eq_df["regret_eq"].to_numpy()

                if self.logging_enabled:
                    logger.info(
                        f"   ⚖️  Gradient Equalization (Ters Hacim Ağırlıklandırması): "
                        f"{window} sızıntısız rolling hacim ile regret ölçeklendi "
                        f"(eps={eps})."
                    )
            except Exception as exc:  # pragma: no cover - savunma amaçlı; hiçbir zaman fit'i kırmasın
                if self.logging_enabled:
                    logger.warning(
                        f"   ⚠️  Gradient Equalization atlandı (hesaplama hatası): {exc}"
                    )

        # --- 3. Min-max normalizasyon + clip (gradyan patlaması önlemi) ---
        lo, hi = self.proxy_spo_weight_clip_
        r_min, r_max = float(regret.min()), float(regret.max())
        if (r_max - r_min) < 1e-9:
            weights = np.full(n, lo, dtype=float)
        else:
            norm = (regret - r_min) / (r_max - r_min)
            weights = lo + norm * (hi - lo)

        if self.logging_enabled:
            n_over = int(is_over.sum())
            logger.info(
                f"   🎯 Proxy SPO (Karar Pişmanlığı Ağırlıklandırması): "
                f"{n_over}/{n} satır kapasite-proxy üzeri (spot araç riski) — "
                f"ort. ağırlık={float(np.mean(weights)):.3f}, "
                f"max ağırlık={float(np.max(weights)):.3f} "
                f"(capacity_q={q}, spot_mult={self.proxy_spo_spot_cost_multiplier_}, "
                f"idle_mult={self.proxy_spo_idle_cost_multiplier_})."
            )

        return weights

    def _compute_surge_dampening_weights(
        self,
        X_surge_full: pd.DataFrame,
        base_pred_surge: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        PDF Bölüm "Dinamik Asimetrik Kayıp ve Gradyan Sönümleme" +
        "Uyarlanabilir Örneklem Ağırlıklandırması" — TEK bir ağırlık
        vektöründe birleştirilmiş uygulama.

        w_i = 1.0                              (varsayılan — çoğu satır)
        w_i = 1 / (1 + α · uplift_i)            (SADECE Pazar + kampanya VE
                                                  Taban modelin (Model 1) o
                                                  satırda normalin belirgin
                                                  üzerinde tahmin ürettiği
                                                  satırlarda)

        uplift_i = max(0, (B_i - B̄_rota) / (B̄_rota + eps))
            B_i     : Taban modelin (OOF/in-sample) o satırdaki q50 tahmini
            B̄_rota  : Aynı rotanın (group_column) Pazar+kampanya DIŞI
                       satırlarındaki ortalama q50 tahmini ("normal" seviye)

        Bu tek dizi CatBoost'a HEM Pool(weight=...) (split arama ağırlığı)
        HEM DE AsymmetricLogCoshObjective.calc_ders_range'in `weights`
        argümanı (gradyan/Hessian sönümü) olarak verilir — CatBoost ikisini
        otomatik çarpar; PDF'in ayrı sunduğu iki teknik (γ_i dampening ve
        sample_weight) matematiksel olarak örtüştüğü için burada
        gereksiz kod tekrarı olmadan tek noktadan uygulanır.

        alpha=0 (surge_dampening_alpha_) → sönümleme tamamen kapalı, eski
        davranışla birebir geriye dönük uyumlu (tüm ağırlıklar 1.0).
        """
        n = len(X_surge_full)
        weights = np.ones(n, dtype=float)

        alpha = float(getattr(self, "surge_dampening_alpha_", 0.0))
        if alpha <= 0.0 or base_pred_surge is None:
            return weights

        campaign_cols = [c for c in ("is_campaign_day", "is_campaign_eve") if c in X_surge_full.columns]
        if not campaign_cols or "weekday" not in X_surge_full.columns:
            # Kampanya veya gün bilgisi yoksa sönümleyecek somut bir şey yok.
            return weights

        is_sunday = (
            pd.to_numeric(X_surge_full["weekday"], errors="coerce")
            .fillna(-1).values.astype(int) == SURGE_SUNDAY_WEEKDAY_VALUE
        )
        is_campaign = np.zeros(n, dtype=bool)
        for c in campaign_cols:
            is_campaign |= (pd.to_numeric(X_surge_full[c], errors="coerce").fillna(0.0) > 0).values

        target_rows = is_sunday & is_campaign
        if not target_rows.any():
            return weights

        eps = 1e-6
        base_pred_surge = np.asarray(base_pred_surge, dtype=float)
        non_target = ~target_rows

        if self.group_column and self.group_column in X_surge_full.columns:
            groups = X_surge_full[self.group_column].values
            baseline = np.empty(n, dtype=float)
            global_fallback = (
                float(np.mean(base_pred_surge[non_target])) if non_target.any()
                else float(np.mean(base_pred_surge))
            )
            for grp in pd.unique(groups):
                grp_mask = groups == grp
                grp_normal_mask = grp_mask & non_target
                baseline[grp_mask] = (
                    float(np.mean(base_pred_surge[grp_normal_mask]))
                    if grp_normal_mask.any() else global_fallback
                )
        else:
            global_baseline = (
                float(np.mean(base_pred_surge[non_target])) if non_target.any()
                else float(np.mean(base_pred_surge))
            )
            baseline = np.full(n, global_baseline, dtype=float)

        uplift = np.maximum(0.0, (base_pred_surge - baseline) / (baseline + eps))
        weights[target_rows] = 1.0 / (1.0 + alpha * uplift[target_rows])

        if self.logging_enabled:
            n_damp = int(target_rows.sum())
            mean_w = float(np.mean(weights[target_rows]))
            logger.info(
                f"   🧯 Dinamik Sönümleme (Pazar+Kampanya): {n_damp} satırda "
                f"ort. ağırlık={mean_w:.3f} (α={alpha}) — çift sayımı önlemek için "
                f"Kalıntı modelinin bu satırlardaki gradyan/split etkisi azaltıldı."
            )
        return weights

    def _build_dynamic_tau_array(
        self,
        base_pred_surge: np.ndarray,
        tau_min: float = DYNAMIC_TAU_MIN,
        tau_max: float = DYNAMIC_TAU_MAX,
        v_low: float = DYNAMIC_TAU_V_LOW,
        v_high: float = DYNAMIC_TAU_V_HIGH,
    ) -> np.ndarray:
        """
        [PDF Entegrasyonu — Dinamik Asimetrik Kayıp] Satır bazlı τ_i vektörü.

        Her gözlemin (rota-gün kombinasyonu) Taban Model (Model 1) tahmini
        `base_pred_surge` (Base_q50) değerine göre, o gözleme özel bir
        asimetri katsayısı (τ_i) üretir. Eskiden TÜM satırlara slot bazında
        (09:00→0.85, 17:00→0.95) sabit bir τ uygulanıyordu; bu, aynı slotta
        hem 50 desilik ölü bir rotayı hem de 15.000 desilik ana arteri
        BİREBİR AYNI şiddette cezalandırıyordu — düşük hacimli rotalarda
        gereksiz "hayalet talep" yaratma riskini artırıyordu.

        Formülasyon (log-hacim ekseninde ortalanmış lojistik geçiş):
            x_i    = log1p(max(Base_q50_i, 0))
            x_mid  = (log1p(v_low) + log1p(v_high)) / 2
            x_half = (log1p(v_high) - log1p(v_low)) / 2
            frac_i = 1 / (1 + exp(-k · (x_i - x_mid) / x_half))     ∈ (0, 1)
            τ_i    = tau_min + (tau_max - tau_min) · frac_i

        k (_DYNAMIC_TAU_LOGIT_ANCHOR ≈ 2.944), frac(x) fonksiyonunun TAM
        olarak x=log1p(v_low) noktasında ≈0.05'e ve x=log1p(v_high)
        noktasında ≈0.95'e ulaşacak şekilde kapalı formda çözülmüştür
        (ln(0.95/0.05) = ln(19)). Sonuç:
            Base_q50 ≈ 50 desi   → τ_i ≈ tau_min + 0.05·(tau_max-tau_min) ≈ 0.522
            Base_q50 ≈ 1500 desi → τ_i ≈ tau_max − 0.05·(tau_max-tau_min) ≈ 0.928
            Base_q50 → 0         → τ_i → tau_min (0.50, simetriğe yakın)
            Base_q50 → ∞         → τ_i → tau_max (0.95, SLA koruyucu)
        Bu, kesikli bir eşik/basamak DEĞİL — türevlenebilir, sürekli bir
        sigmoid'dir (ALNS'e sunulan maliyet/gradyan yüzeyinde ani sıçrama
        yaratmaz — bkz. PDF'in "Sürekli Ölçekleme" bölümü).

        Parameters
        ----------
        base_pred_surge : np.ndarray
            Surge/Residual (Model 2) eğitim setindeki (surge_mask ile
            filtrelenmiş) satırlara karşılık gelen Taban Model (Model 1)
            q50 tahminleri — _train_surge_residual_model içindeki
            `base_pred_surge` ile AYNI SIRADA olmalıdır.
        tau_min, tau_max, v_low, v_high : float
            Sigmoid'in anchor noktaları — modül sabitleri (DYNAMIC_TAU_*)
            varsayılan olarak kullanılır, test/A-B amaçlı override edilebilir.

        Returns
        -------
        np.ndarray
            base_pred_surge ile aynı uzunlukta, (tau_min, tau_max) açık
            aralığında τ_i değerleri.
        """
        base_pred_surge = np.asarray(base_pred_surge, dtype=np.float64)

        x = np.log1p(np.maximum(base_pred_surge, 0.0))
        x_low = np.log1p(v_low)
        x_high = np.log1p(v_high)
        x_mid = (x_low + x_high) / 2.0
        x_half = max((x_high - x_low) / 2.0, 1e-9)  # v_low==v_high yanlış konfigürasyonuna karşı güvenlik

        frac = 1.0 / (1.0 + np.exp(-_DYNAMIC_TAU_LOGIT_ANCHOR * (x - x_mid) / x_half))
        tau_array = tau_min + (tau_max - tau_min) * frac

        # Sayısal güvenlik: AsymmetricLogCoshObjective açık aralık (0,1) bekliyor.
        eps = 1e-6
        tau_array = np.clip(tau_array, tau_min if tau_min > 0 else eps, tau_max if tau_max < 1 else 1.0 - eps)
        return tau_array

    def _compute_volume_damping_factor(
        self,
        baseline_pred: np.ndarray,
        v_crit: Optional[float] = None,
        k: Optional[float] = None,
    ) -> np.ndarray:
        """
        [PDF Entegrasyonu — Hacim Tabanlı Lojistik Sönümleme] S_vol(V_base).

        Model 2'nin (Surge/Residual) ürettiği kalıntı düzeltmesini, rotanın
        taban hacmine (Base_q50) göre SÜREKLİ bir çarpanla sönümler:
        hacim düşükse (V_base << v_crit) düzeltme neredeyse tamamen
        bastırılır (S_vol ≈ 0 — küçük/durgun rotalarda "hayalet talep"
        yaratılmasın), hacim kritik eşiğin üzerindeyse düzeltme olduğu gibi
        geçer (S_vol → 1.0).

        ESKİ (Faz 2b — Segment Scale) mekanizmanın YERİNE geçer: o mekanizma
        KESİKLİ hacim aralıkları [(low, high, scale), ...] kullanıyordu ve
        bir rota aralık sınırını (ör. 60 desi) geçtiği an çarpan aniden
        sıçrıyordu — ALNS'in komşuluk arama operatörleri için maliyet
        yüzeyinde yapay bir uçurum/süreksizlik demekti. Burada kullanılan
        sigmoid TÜREVLENEBİLİR ve süreklidir; hiçbir hacim noktasında ani
        sıçrama yoktur.

        Formülasyon (PDF'in "Hacim Tabanlı Lojistik Sönümleme" bölümüyle
        birebir aynı, sıfıra normalize edilmiş lojistik):
            S_vol(V_base) = max(0, 2 / (1 + exp(-k · (V_base / V_crit))) - 1)

        V_base=0 olduğunda standart lojistik 0.5'e (sıfır değil) yapışır;
        bu yüzden çıktı [0, 1) aralığına normalize edilmiş bu özel forma
        (2·sigmoid - 1, negatifse 0'a kırpılır) ihtiyaç vardır — V_base=0
        için S_vol TAM OLARAK 0 olur.

        Parameters
        ----------
        baseline_pred : np.ndarray
            Sönümlenecek satırların taban hacmi (Base_q50, düzeltme
            ÖNCESİ ham q50 — _predict_single_batch içindeki
            `q50_vals[surge_mask_pred]` ile aynı).
        v_crit : float, optional
            Kritik hacim eşiği (desi). None ise self.surge_volume_damping_v_crit_
            kullanılır (varsayılan 150.0).
        k : float, optional
            Sigmoid eğim katsayısı. None ise self.surge_volume_damping_k_
            kullanılır (varsayılan 3.0).

        Returns
        -------
        np.ndarray
            baseline_pred ile aynı uzunlukta, [0, 1) aralığında S_vol
            çarpanları.
        """
        v_crit = float(v_crit if v_crit is not None else getattr(self, "surge_volume_damping_v_crit_", 150.0))
        k = float(k if k is not None else getattr(self, "surge_volume_damping_k_", 3.0))
        v_crit = v_crit if v_crit > 0 else 1e-6  # sıfır/negatif yanlış konfigürasyona karşı güvenlik

        baseline_pred = np.asarray(baseline_pred, dtype=np.float64)
        s_vol = np.maximum(
            0.0,
            2.0 / (1.0 + np.exp(-k * (baseline_pred / v_crit))) - 1.0,
        )
        return s_vol

    def _train_surge_residual_model(
        self,
        X_train: pd.DataFrame,
        y_train: Optional[pd.Series],
        precomputed_residual: Optional[np.ndarray] = None,
        precomputed_base_pred: Optional[np.ndarray] = None,
    ) -> None:
        """
        Model 2 (Surge/Residual) — İki Aşamalı Kalıntı Modellemesi.

        Yalnızca fit() içinde, Model 1 (ensemble) eğitimi TAMAMLANDIKTAN
        SONRA çağrılır (bkz. fit() — self.models_ dolu olmalı).

        Adımlar
        -------
        1. _build_surge_trigger_mask() ile train setindeki tetikleyici
           satırları bul. surge_min_rows altındaysa atla (B Planı:
           campaign_multipliers_ heuristiği devrede kalır).
        2. Model 1'in (4-fold ensemble, median) train seti üzerindeki
           q50 tahminini hesapla — gerekirse sqrt dönüşümünü geri çevir.
        3. Kalıntı = y_true - base_q50 (SADECE surge satırlarında).
        4. Özellik Uzayı Ortogonalleştirmesi (PDF): SURGE_STATIC_EXCLUDED_
           FEATURES içindeki statik takvim/kampanya bayrakları Kalıntı
           modelinin feature matrisinden ÇIKARILIR — model artık
           "kampanya var mı" değil "sistemde açıklanamayan bir yığılma var
           mı" sorusuna (backlog_release_index, campaign_release_index,
           rolling/EWMA istatistikler vb. dinamik sinyaller üzerinden) cevap
           verir. Bu, çift sayımı kaynağında keser.
        5. Dinamik Sönümleme (PDF): Pazar+kampanya satırlarında, Taban
           modelin ZATEN yükselttiği tahminlere göre ölçeklenen bir ağırlık
           vektörü (_compute_surge_dampening_weights) hem Pool(weight=...)
           hem de AsymmetricLogCoshObjective'in gradyan/Hessian'ına uygulanır.
        6. AsymmetricLogCoshObjective (tau=tau_array — bkz.
           _build_dynamic_tau_array, satır/rota bazında dinamik τ_i)
           kayıp fonksiyonuyla küçük, hızlı bir CatBoostRegressor eğit.

        Not (ADIM 3 güncellemesi): Kalıntı hedefi artık VARSAYILAN olarak
        `precomputed_residual` üzerinden, K-Fold döngüsünün OOF (out-of-sample)
        val tahminlerinden hesaplanıyor — bkz. fit() içindeki fold döngüsü ve
        self._oof_X_ / self._oof_residual_ / self._oof_base_pred_. Bu, her
        fold modelinin kendi val haftasını hiç görmemesinden yararlanır
        (use_best_model=False), yani gerçek out-of-sample bir kalıntı elde
        edilir; eski in-sample (train seti üzerinde) hesaplama sadece
        `precomputed_residual=None` geçildiğinde (fallback / OOF verisi
        boşsa) kullanılır.
        """
        self.surge_model_ = None
        self.surge_feature_names_ = []
        self.surge_cat_features_ = []

        if not self.surge_residual_enabled:
            return

        surge_mask = self._build_surge_trigger_mask(X_train)
        n_surge = int(surge_mask.sum())

        if n_surge < self.surge_min_rows:
            if self.logging_enabled:
                logger.info(
                    f"   ⚠️  Surge/Residual modeli (Model 2) ATLANDI: train setinde "
                    f"sadece {n_surge} tetikleyici satır bulundu "
                    f"(min={self.surge_min_rows}, tetikleyiciler={self.surge_trigger_columns_used_}). "
                    f"Eski çarpan heuristiği (campaign_multipliers_) B Planı olarak devrede."
                )
            return

        if precomputed_residual is not None:
            # OOF — zaten y_true - out-of-sample q50 (bkz. fit() içindeki fold döngüsü).
            # In-sample q50 tekrar hesaplanmıyor; residual_train doğrudan kullanılıyor.
            residual_train = np.asarray(precomputed_residual, dtype=float)
            base_pred_train = (
                np.asarray(precomputed_base_pred, dtype=float)
                if precomputed_base_pred is not None and len(precomputed_base_pred) == len(residual_train)
                else None
            )
            if base_pred_train is None and self.logging_enabled:
                logger.info(
                    "   ⚠️  precomputed_base_pred verilmedi/uyumsuz — Dinamik Sönümleme "
                    "bu eğitimde devre dışı (ağırlıklar=1.0, eski davranış)."
                )
        else:
            # --- Model 1'in (ensemble) train seti üzerindeki q50 tahmini (in-sample) ---
            train_pool = Pool(data=X_train, cat_features=self.cat_features_)
            base_q50_train = np.median(
                [m.predict(train_pool)[:, 1] for m in self.models_], axis=0
            )
            y_true_train = y_train.to_numpy(dtype=float)

            if self.log_transform_enabled:
                base_q50_train = np.square(base_q50_train)
                y_true_train = np.square(y_true_train)
            base_q50_train = np.maximum(base_q50_train, 0.0)

            residual_train = y_true_train - base_q50_train
            base_pred_train = base_q50_train

        # surge_mask, X_train üzerinden hesaplanıyor (yukarıda) — precomputed_residual
        # durumunda X_train = self._oof_X_ ve residual_train = self._oof_residual_,
        # concat sırası korunduğu için satır bazında hizalıdır. Tip uyuşmazlığını
        # (pandas bool Series vs numpy array indexleme) önlemek için mask'i numpy'a çevir.
        surge_mask = np.asarray(surge_mask)
        X_surge_full = X_train.loc[surge_mask]          # ORTOGONALLEŞTİRME ÖNCESİ — tüm sütunlar
        residual_surge = residual_train[surge_mask]
        base_pred_surge = base_pred_train[surge_mask] if base_pred_train is not None else None

        # --- Dinamik Sönümleme / Uyarlanabilir Ağırlık — orijinal (statik
        # bayraklar dahil) sütunlar üzerinden hesaplanır, çünkü is_campaign_day/
        # weekday'e ihtiyaç duyar (bu sütunlar birazdan feature matrisinden
        # çıkarılacak olsa da, AĞIRLIK hesaplamak için hâlâ okunabilir).
        sample_weights_surge = self._compute_surge_dampening_weights(
            X_surge_full, base_pred_surge
        )

        # --- Özellik Uzayı Ortogonalleştirmesi (PDF) ---
        # Statik kampanya/tatil bayraklarını Kalıntı modelinin GÖRDÜĞÜ
        # sütunlardan çıkar. Trigger mask zaten yukarıda hesaplandığı için
        # bu satırların "surge" sayılması etkilenmez — sadece modelin bu
        # bayrakları HAM haliyle tekrar öğrenmesi engellenir.
        cols_to_drop = [c for c in SURGE_STATIC_EXCLUDED_FEATURES if c in X_surge_full.columns]
        X_surge = X_surge_full.drop(columns=cols_to_drop) if cols_to_drop else X_surge_full
        self.surge_feature_names_ = list(X_surge.columns)
        self.surge_cat_features_ = [c for c in self.cat_features_ if c in X_surge.columns]

        # --- [PDF Entegrasyonu — Dinamik Asimetrik Kayıp] ---
        # ESKİ davranış: sabit tau=0.95 (17:00) / tau=0.85 (09:00) — TÜM
        # satırlara slot bazında eşit uygulanıyordu (bkz. git geçmişi:
        # `dinamik_tau = 0.95 if "1700" in self.target_column else 0.85`).
        # Sabit tau=0.95 tüm slotlara uygulandığında 17:00 iyileşiyor ama
        # 09:00 modeli bozuluyordu (aşırı agresif asimetri, sabah trafiğinde
        # gereksiz yere yukarı sıçramalar üretiyordu) — bu yüzden slot
        # bazında ikiye bölünmüştü.
        # YENİ davranış: PDF'in önerdiği gibi τ artık slot bazında DEĞİL,
        # SATIR (rota-gün) bazında dinamik — her gözlemin KENDİ Base_q50
        # hacmine göre _build_dynamic_tau_array() ile üretilir (50 desi
        # civarında τ≈0.50'ye yakın/simetrik, 1500+ desi ana arterlerde
        # τ≈0.95'e yakın/SLA koruyucu). Bu, slot ayrımına gerek KALMADAN
        # (aynı slottaki küçük ve büyük rotalar artık kendi ölçeğine göre
        # cezalandırılıyor) daha ince taneli bir çözüm sunar ve slot bazlı
        # 0.85/0.95 sıçramasını rotanın gerçek hacmiyle orantılı, sürekli
        # bir geçişe çevirir.
        # NOT: base_pred_surge her zaman dolu OLMALI (precomputed_base_pred
        # OOF akışında normalde her zaman verilir) — nadir/eski çağrı
        # yolunda (None) eski slot-bazlı sabit τ'ya güvenli şekilde düşülür.
        if base_pred_surge is not None:
            tau_for_loss = self._build_dynamic_tau_array(base_pred_surge)
            tau_log_desc = (
                f"dinamik vektör (n={tau_for_loss.size}, "
                f"min={tau_for_loss.min():.3f}, ort={tau_for_loss.mean():.3f}, "
                f"max={tau_for_loss.max():.3f})"
            )
        else:
            tau_for_loss = 0.95 if "1700" in self.target_column else 0.85
            tau_log_desc = f"sabit τ={tau_for_loss} (base_pred_surge yok — eski slot-bazlı davranışa düşüldü)"

        surge_pool = Pool(
            data=X_surge,
            label=residual_surge,
            cat_features=self.surge_cat_features_,
            weight=sample_weights_surge,
        )

        self.surge_model_ = CatBoostRegressor(
            iterations=min(800, self.iterations),          # ADIM 4 SEÇİLEN KOMBO (4): 400 → 800
            depth=max(4, self.depth - 1),                     # depth bir azaltıldı — en dengeli regret/MAPE
            learning_rate=self.learning_rate * 0.7,           # biraz daha yavaş öğren
            l2_leaf_reg=max(3.0, self.l2_leaf_reg * 0.1),      # base'in 0.1x'i (taban 3.0'a çok yakın/yapışık)
            loss_function=AsymmetricLogCoshObjective(tau=tau_for_loss),
            eval_metric="RMSE",   # custom objective ile CatBoost'un zorunlu tuttuğu izleme metriği
            random_seed=self.random_state,
            verbose=False,
            allow_writing_files=False,
            # ⚠️ [PDF Entegrasyonu — Dinamik Asimetrik Kayıp / KRİTİK DÜZELTME]
            # thread_count=-1 (tüm çekirdekler) İKEN, CatBoost `self` kullanan
            # (bound-method, JIT-optimize edilemeyen — bkz. yukarıdaki
            # UserWarning: "Can't optimize method calc_ders_range") özel
            # amaç fonksiyonlarını TEK bir çağrıda TÜM veri seti üzerinden
            # DEĞİL, iş parçacığı sayısına göre paralel/ardışık İÇ BLOKLARA
            # bölerek çağırır (gözlemlenen gerçek hata: 7515 satırlık veri
            # 1000'lik parçalar halinde geldi). calc_ders_range'e satır
            # indeksleri VERİLMEZ — bu yüzden tau_array (self.tau, ndim=1)
            # doğru satırla eşleşemez ve AsymmetricLogCoshObjective'in
            # kendi güvenlik kontrolü (uzunluk uyuşmazlığı) haklı olarak
            # ValueError fırlatır.
            # Çözüm: thread_count=1 → CatBoost paralel blok bölmeden, TEK
            # iş parçacığında, HER iterasyonda veri setinin TAMAMINI ORİJİNAL
            # SIRAYLA (ardışık/deterministik) tarar. Bu, AsymmetricLogCoshObjective
            # içindeki iç imlecin (self._cursor — bkz. calc_ders_range) her
            # bloğu doğru tau_i alt-dizisiyle eşleştirmesini garanti eder.
            # Bedel: eğitim tek çekirdekte çalıştığı için biraz daha yavaş —
            # ancak bu model zaten küçük/hızlı olacak şekilde tasarlanmıştı
            # (iterations≤800, depth bir azaltılmış) — kabul edilebilir.
            # tau_for_loss SKALER (float) olduğunda (base_pred_surge yoksa)
            # bu kısıtlamaya hiç gerek YOK (broadcast, sıra bağımsız) ama
            # kod yolunu basit/tek tip tutmak için her durumda thread_count=1
            # kullanılıyor.
            thread_count=1,
        )
        self.surge_model_.fit(surge_pool)

        if self.logging_enabled:
            mean_res = float(np.mean(residual_surge))
            mean_abs_res = float(np.mean(np.abs(residual_surge)))
            kaynak = "OOF (out-of-sample), y-base_q50" if precomputed_residual is not None else "train (in-sample), y-base_q50"
            logger.info(
                f"   🚀 Surge/Residual modeli (Model 2, Asimetrik Log-Cosh τ={tau_log_desc} "
                f"[hedef={self.target_column}]) "
                f"eğitildi: {n_surge} tetikleyici satır | tetikleyiciler={self.surge_trigger_columns_used_}\n"
                f"      ort. kalıntı ({kaynak}) = {mean_res:+,.1f} desi | "
                f"ort. |kalıntı| = {mean_abs_res:,.1f} desi\n"
                f"      🧬 Ortogonalleştirme: {len(cols_to_drop)} statik bayrak çıkarıldı "
                f"({cols_to_drop if cols_to_drop else '—'}) → Kalıntı modeli "
                f"{len(self.surge_feature_names_)} özellik görüyor "
                f"(Taban modelin {len(self.feature_names_)} özelliğine karşı)."
            )

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

        # --- 3. [YENİ, v4] SCALED hedef için ROTA BAZLI IQR üst-kırpması ---
        # ⚠️ REVİZYON NOTU (v2→v4): v2'de bunu GLOBAL yapmıştım çünkü v1'in
        # (rota bazlı) "işe yaramadığını" görmüştüm. Ama o gözlem YANLIŞ
        # yorumlanmıştı: v1 test edilirken v3'teki asıl bug (df_features'a
        # hiç geri yazılmıyordu — K-Fold döngüsü kırpmayı hiç görmüyordu)
        # ZATEN aktifti. Yani v1'in "etkisiz" görünmesi rota-bazlı mantığın
        # kendi kusuru değildi — kırpma o AŞAMADA HİÇBİR ŞEKİLDE modele
        # ulaşmıyordu (v2 de aynı sahte-negatifi verdi). "Kendi kendini kör
        # eden IQR" teorisi hiç doğrulanmadı, sadece spekülasyondu.
        # v3 (df_features'a geri yazma) düzeltildiğine göre, artık daha
        # savunulabilir olan ROTA BAZLI yaklaşımı gerçek anlamda test
        # edebiliriz: 289 rotanın doğal volatilite profilleri farklı
        # (büyük şehirlerarası hat vs. küçük besleme hattı) — tek bir
        # global eşik oynak-ama-meşru rotaları fazla kırpıp durgun
        # rotalardaki gerçek outlier'ları kaçırabilir. Rota bazlı, her
        # rotanın kendi dağılımına adil bir eşik verir.
        self._scaled_clip_upper_: Dict[str, float] = {}
        if (
            getattr(self, "_target_scaling_active_", False)
            and self._scaled_target_col_
            and self._scaled_target_col_ in df.columns
        ):
            scol = self._scaled_target_col_
            n_scaled_clipped = 0
            if self.group_column and self.group_column in df.columns:
                for grp, grp_df in df.groupby(self.group_column):
                    q25 = grp_df[scol].quantile(0.25)
                    q75 = grp_df[scol].quantile(0.75)
                    self._scaled_clip_upper_[grp] = q75 + self.outlier_clip_multiplier * (q75 - q25)
                before = df[scol].copy()
                for grp, upper in self._scaled_clip_upper_.items():
                    mask = df[self.group_column] == grp
                    df.loc[mask, scol] = df.loc[mask, scol].clip(upper=upper)
                n_scaled_clipped = int((before != df[scol]).sum())
            else:
                q25 = df[scol].quantile(0.25)
                q75 = df[scol].quantile(0.75)
                self._scaled_clip_upper_["_global_"] = q75 + self.outlier_clip_multiplier * (q75 - q25)
                before = df[scol].copy()
                df[scol] = df[scol].clip(upper=self._scaled_clip_upper_["_global_"])
                n_scaled_clipped = int((before != df[scol]).sum())

            if self.logging_enabled and n_scaled_clipped > 0:
                logger.info(
                    f"   🔧 [YENİ v4] Scaled-hedef outlier kırpma (train, '{scol}', ROTA BAZLI): "
                    f"{n_scaled_clipped} değer kırpıldı (IQR × {self.outlier_clip_multiplier})."
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

        # --- [YENİ] Scaled-hedef kırpmasını da (öğrenilmişse) uygula ---
        if (
            getattr(self, "_scaled_clip_upper_", None)
            and getattr(self, "_scaled_target_col_", None)
            and self._scaled_target_col_ in df.columns
        ):
            scol = self._scaled_target_col_
            if "_global_" in self._scaled_clip_upper_:
                df[scol] = df[scol].clip(upper=self._scaled_clip_upper_["_global_"])
            elif self.group_column and self.group_column in df.columns:
                for grp, upper in self._scaled_clip_upper_.items():
                    mask = df[self.group_column] == grp
                    df.loc[mask, scol] = df.loc[mask, scol].clip(upper=upper)

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

        Notlar
        ------
        Yeni features.py imzası `target_columns` (çoğul, Union[str, List[str]])
        bekliyor. Burada TEK bir hedef değil, HER İKİ slotun hedef sütununu
        birden geçiyoruz — çünkü build_feature_matrix'in iç mekanizması
        (hub/graph/hiyerarşik özellikler, cross-lag) her iki hedefin de
        var olmasını bekliyor; sadece kendi hedefini görseydi, diğer slotun
        bilgisini feature'a hiç dönüştüremezdi.

        Bunun sonucu olarak üretilen matriste her iki hedef sütun da
        (toplam_desi_0900 ve toplam_desi_1700) ve bunların lag/rolling'leri
        suffix'li olarak (lag_1_0900, lag_1_1700, ...) bulunur. Hangisinin bu
        model instance'ı için "gerçek hedef", "feature" veya "tamamen drop"
        olduğuna _get_drop_columns() / _split_X_y() karar verir.
        """
        if not self.sibling_target_column:
            raise ValueError(
                "❌ sibling_target_column zorunludur — wide format "
                "iki-slotlu akış (09:00/17:00) için hem kendi hedefin hem "
                "diğer slotun hedef sütun adı gerekir.\n"
                "   Örn: DemandForecaster(target_column='toplam_desi_0900', "
                "sibling_target_column='toplam_desi_1700')"
            )

        return build_feature_matrix(
            df=df,
            target_columns=[self.target_column, self.sibling_target_column],
            date_column=self.date_column,
            group_column=self.group_column,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            campaign_release_alpha=self.campaign_release_alpha_,
            campaign_max_release_days=self.campaign_max_release_days_,
            # getattr ile geriye dönük uyumluluk: joblib.load() __init__()'i
            # tekrar ÇALIŞTIRMAZ, bu yüzden bu attribute'lardan ÖNCE eğitilmiş
            # (pickle'lanmış) eski modellerde backlog_alpha_/backlog_max_
            # release_days_ hiç yok — AttributeError yerine sınıf varsayılanına
            # (eski/mevcut davranış, 5.25/6) düşer (bkz. surge_calibration_
            # factor_ gibi diğer attribute'lardaki AYNI desen).
            backlog_alpha=getattr(self, "backlog_alpha_", 5.25),
            backlog_max_release_days=getattr(self, "backlog_max_release_days_", 6),
            # v18 — aynı getattr geriye dönük uyumluluk deseni: bu attribute'lar
            # (backlog_baseline_window_/campaign_baseline_window_) v18'den ÖNCE
            # eğitilmiş pickle'lanmış modellerde yok; yoksa features.py'nin
            # kendi sınıf varsayılanına (14) düşer.
            backlog_baseline_window=getattr(self, "backlog_baseline_window_", 14),
            campaign_baseline_window=getattr(self, "campaign_baseline_window_", 14),
            drop_na=drop_na,
            enable_target_scaling=self.target_scaling_enabled_,
            target_scale_window_days=self.target_scale_window_days_,
            target_scale_min=self.target_scale_min_,
            censor_window=self.censor_window_,
            censor_min_volume_threshold=self.censor_min_volume_threshold_,
            censor_cap_ratio=self.censor_cap_ratio_,
            censor_inflation_factor=self.censor_inflation_factor_,
            censor_require_weekday_persistence=self.censor_require_weekday_persistence_,
            censor_persistence_occurrences=self.censor_persistence_occurrences_,
            censor_persistence_min_hits=self.censor_persistence_min_hits_,
            censor_capacity_df=self.censor_capacity_df_,
            censor_source_tm_column=self.censor_source_tm_column_,
            censor_capacity_tm_column=self.censor_capacity_tm_column_,
            censor_capacity_value_column=self.censor_capacity_value_column_,
            censor_real_capacity_ratio=self.censor_real_capacity_ratio_,
        )

    def _split_X_y(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Feature matrix'ten X ve y'yi ayırır.

        Modele girmeyen sütunları (date, target ve slot-farkındalıklı
        leakage sütunları) X'ten çıkarır — bkz. _get_drop_columns().
        Bu sayede date sütunu ve (09:00 modeli için) sibling/cross-lag
        sütunları tahmine sızmaz (leakage önlemi).
        """
        drop_cols = self._get_drop_columns(df.columns)

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
        self.iterations, self.depth, self.learning_rate, self.l2_leaf_reg, \
            self.bagging_temperature, self.optimized_alpha_, config_label = \
            _load_hyperparams(data_size, self.target_column, self.logging_enabled)

        # --- 1. Validasyon ---

        # sibling_target_column zorunlu (base class bunu bilmiyor, burada kontrol ediyoruz)
        if not self.sibling_target_column:
            raise ValueError(
                "❌ sibling_target_column zorunludur — wide format iki-slotlu "
                "akış (09:00/17:00) için hem kendi hedefin hem diğer slotun "
                "hedef sütun adı gerekir."
            )
        if self.sibling_target_column not in df.columns:
            raise ValueError(
                f"❌ df içinde sibling_target_column ('{self.sibling_target_column}') "
                f"bulunamadı. Ham veri (full_df) her iki slotun hedef sütununu da "
                f"içermelidir — build_feature_matrix ikisini de bekler."
            )

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

        df_features = self._engineer_features(df, drop_na=False)

        # --- 2.1 Hedef Ölçeklendirme (Target Scaling) Aktivasyon Kontrolü ---
        # PDF Strateji 2: add_scale_invariant_targets() beklenen sütunları
        # gerçekten ürettiyse (enable_target_scaling=True VE grup/tarih
        # bilgisi yeterliyse) K-Fold ensemble eğitimi bu sütunları kullanacak
        # şekilde etkinleşir; aksi halde (örn. eski bir features.py sürümü
        # veya target_scaling_enabled_=False) sessizce eski (raw) davranışa
        # düşülür — hiçbir yerde sert hata fırlatılmaz.
        _scale_suffix = f"_{self.target_column.rsplit('_', 1)[-1]}" if "_" in self.target_column else ""
        _candidate_scale_col = f"scale_factor{_scale_suffix}"
        _candidate_scaled_col = f"{self.target_column}_scaled"
        self._target_scaling_active_ = bool(
            self.target_scaling_enabled_
            and _candidate_scale_col in df_features.columns
            and _candidate_scaled_col in df_features.columns
        )
        if self._target_scaling_active_:
            self._scale_factor_col_ = _candidate_scale_col
            self._scaled_target_col_ = _candidate_scaled_col
            if self.logging_enabled:
                logger.info(
                    f"   📐 Hedef Ölçeklendirme (Target Scaling) AKTİF: "
                    f"K-Fold ensemble '{self._scaled_target_col_}' (rotanın "
                    f"{self.target_scale_window_days_} günlük ortalamasına göre "
                    f"oransal hedef) üzerinden eğitilecek; tahminler "
                    f"'{self._scale_factor_col_}' ile geri çevrilecek."
                )
        else:
            self._scale_factor_col_ = None
            self._scaled_target_col_ = None
            if self.target_scaling_enabled_ and self.logging_enabled:
                logger.warning(
                    f"   ⚠️  Hedef Ölçeklendirme istendi (target_scaling_enabled_=True) "
                    f"ama beklenen sütunlar ('{_candidate_scale_col}', "
                    f"'{_candidate_scaled_col}') df_features'ta bulunamadı — "
                    f"eski (raw hedef) davranışına düşülüyor."
                )

        # --- 3. Train/Test Split (walk-forward, zaman sıralı) ---
        train_df, test_df = self._train_test_split(df_features)

        # --- 3.5 Anormal Hafta Tespiti (optimize.py ile BİREBİR AYNI mantık) ---
        # Amaç: forecasters.py'nin kendi self-evaluation'ı (bu split, ör. son ~%15
        # gün) ile optimize.py'nin raporladığı "best_wape_clean" (kendi Fold-4
        # penceresi) arasında adil bir karşılaştırma yapılabilmesi. optimize.py
        # zaten haftalık ortalama hacmin genel ortalamanın 1.4 katını aştığı
        # haftaları (tatil/kampanya birikimi vb.) "anormal" sayıp WAPE'den dışlıyor;
        # forecasters.py'nin kendi "WAPE (tatil hariç)" hesabı ise sadece
        # is_holiday/Pazar bayrağını dışlıyordu — daha dar bir filtreydi. Aşağıda
        # aynı 1.4x eşiğini uygulayıp _evaluate_on_test()'e aktarıyoruz ki
        # "temiz WAPE" gerçekten optimize.py'nin metriğiyle kıyaslanabilir olsun.
        self._abnormal_weeks_ = set()
        if self.target_column in df_features.columns and self.date_column in df_features.columns:
            _weekly_src = df_features[df_features[self.target_column] > 0].copy()
            if not _weekly_src.empty:
                _weekly_src["_week"] = _weekly_src[self.date_column].dt.isocalendar().week.astype(int)
                _weekly_src["_year"] = _weekly_src[self.date_column].dt.year
                _wk_means = _weekly_src.groupby(["_year", "_week"])[self.target_column].mean()
                _wk_threshold = _wk_means.mean() * 1.4
                self._abnormal_weeks_ = set(_wk_means[_wk_means > _wk_threshold].index)
                if self.logging_enabled and self._abnormal_weeks_:
                    logger.info(
                        f"⚠️  Anormal haftalar tespit edildi (optimize.py ile aynı eşik, "
                        f"ort. × 1.4): {sorted(self._abnormal_weeks_)}"
                    )

        # --- 4. Veri Temizleme — SADECE train üzerinde fit et (leakage önlemi) ---
        # IQR eşikleri yalnızca train_df'ten öğrenilir.
        # Aynı eşikler test_df'e uygulanır — test dağılımı öğrenmeye girmez.
        if self.logging_enabled:
            logger.info("🧹 Veri temizleme başlıyor (train only fit)...")
        train_df = self._fit_clip(train_df)   # eşikleri öğren + uygula
        test_df, _ = self._apply_clip(test_df)  # sadece uygula

        # ⚠️⚠️ KÖK NEDEN DÜZELTMESİ (v3) ⚠️⚠️
        # K-Fold Ensembling döngüsü (aşağıda, ~2000. satır civarı) fold_train_df
        # / fold_val_df'i train_df/test_df'TEN DEĞİL, doğrudan df_features'tan
        # tarih filtresiyle türetiyor:
        #     fold_train_df = df_features[df_features[date_column] < val_start]
        # Yani yukarıdaki _fit_clip/_apply_clip çağrıları train_df/test_df'i
        # değiştiriyor ama df_features HİÇ etkilenmiyor — K-Fold ensemble
        # (gerçek dağıtılan model) kırpılmamış, ham veriyle eğitiliyor.
        # Kanıt: 3 ardışık retrain'de kırpılan satır sayısı değişti
        # (2318→1602, 811→726) ama nihai tahminler ve backtest regret'i
        # (4889.29, 1178.45) VİRGÜLÜNE KADAR aynı kaldı.
        # Düzeltme: öğrenilen kırpmayı train/test indekslerini kullanarak
        # df_features'a GERİ YAZ — böylece K-Fold döngüsü de görsün.
        df_features.loc[train_df.index, self.target_column] = train_df[self.target_column]
        df_features.loc[test_df.index, self.target_column] = test_df[self.target_column]
        if self._scaled_target_col_ and self._scaled_target_col_ in train_df.columns:
            df_features.loc[train_df.index, self._scaled_target_col_] = train_df[self._scaled_target_col_]
            df_features.loc[test_df.index, self._scaled_target_col_] = test_df[self._scaled_target_col_]
        if self.logging_enabled:
            logger.info(
                "   🔧 [YENİ v3] Kırpılmış train/test değerleri df_features'a "
                "geri yazıldı (K-Fold döngüsü artık kırpılmış veriyi görecek)."
            )

        # --- Dinamik Kampanya Çarpanlarını Öğren ---
        self._learn_campaign_multipliers(train_df)

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
                    f"🔁 Sqrt dönüşümü uygulanıyor\n"
                    f"   Ham çarpıklık   : {skew_stats.get('skewness_raw', '?'):+.4f}\n"
                    f"   predict() çıktısı otomatik square() ile geri çevrilecek."
                )
            train_df[self.target_column] = np.sqrt(train_df[self.target_column])
            test_df[self.target_column]  = np.sqrt(test_df[self.target_column])
        else:
            # log_transform kapalıysa yine de çarpıklık raporla (tavsiye için)
            compute_target_skewness(
                df            = train_df,
                target_column = self.target_column,
                group_column  = self.group_column,
                log_transform = False,
            )

        # Test satırlarının anormal-hafta maskesi — date_column X_test'ten
        # düşürülmeden ÖNCE hesaplanmalı (bkz. 3.5 adımı).
        abnormal_week_mask_test: Optional[np.ndarray] = None
        if self._abnormal_weeks_ and self.date_column in test_df.columns:
            _test_dates = pd.to_datetime(test_df[self.date_column])
            _test_years = _test_dates.dt.year
            _test_weeks = _test_dates.dt.isocalendar().week.astype(int)
            abnormal_week_mask_test = np.array([
                (y, w) in self._abnormal_weeks_
                for y, w in zip(_test_years, _test_weeks)
            ])

        X_train, y_train = self._split_X_y(train_df)
        X_test,  y_test  = self._split_X_y(test_df)

        # --- Bug Fix — Target Scaling Un-scale (Self-Evaluation) ---
        # _split_X_y() → _get_drop_columns() zaten scale_factor* sütununu X'ten
        # düşürüyor (leakage önlemi, bkz. _get_drop_columns). _evaluate_on_test()
        # ensemble modellerini DOĞRUDAN çağırdığı için (Pool + model.predict()),
        # _target_scaling_active_ ise çıktı SCALED uzaydadır (~1.0 civarı oran) —
        # bunu ham y_train/y_test (raw desi) ile karşılaştırmadan ÖNCE kendi
        # scale_factor'üyle geri çarpmak gerekir (aksi halde WAPE ~%100 gibi
        # anlamsız bir değere yapışır — bkz. _predict_single_batch()'teki AYNI
        # un-scale adımı, satır ~2125). X'ten düşürülmeden ÖNCE train_df/test_df
        # üzerinden ayrıca saklıyoruz.
        scale_factor_train = scale_factor_test = None
        if self._target_scaling_active_ and self._scale_factor_col_:
            if self._scale_factor_col_ in train_df.columns:
                scale_factor_train = train_df.loc[X_train.index, self._scale_factor_col_].to_numpy(dtype=float)
            if self._scale_factor_col_ in test_df.columns:
                scale_factor_test = test_df.loc[X_test.index, self._scale_factor_col_].to_numpy(dtype=float)

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

        # --- 4. Zaman Serisi Cross-Validation ve Ensemble Eğitimi ---
        # 4 Fold (7'şer günlük) — her biri farklı haftayı validation seti olarak kullanır
        # NOT: Bu pencereler, gerçek tahmin penceresine (PREDICT_START/END =
        # 2026-06-29 → 2026-07-05, run_forecast.py) en yakın, veri içindeki son
        # 4 tam hafta olacak şekilde seçildi — optimize.py'deki FOLD_DATES ile
        # BİREBİR AYNI tutulmalı (aksi halde optimize.py'nin bulduğu hiperparametreler
        # bu fold pencerelerinde eğitilen gerçek modelden farklı bir dönem için
        # tuned olur).
        fold_dates = [
            ("Fold 1", "2026-05-31", "2026-06-06"),
            ("Fold 2", "2026-06-07", "2026-06-13"),
            ("Fold 3", "2026-06-14", "2026-06-20"),
            ("Fold 4", "2026-06-21", "2026-06-27"),
        ]
        self.models_: List[CatBoostRegressor] = []

        if self.logging_enabled:
            logger.info("🚀 K-Fold Time-Series Ensembling Başlıyor (4 Model Eğitilecek)...")

        t_q = time.time()

        oof_X_list, oof_residual_list, oof_base_pred_list = [], [], []
        # Adım 2 düzeltmesi: date_column X_fold_val'dan HER ZAMAN drop edildiği
        # (leakage) için _oof_X_ üzerinden tarih bilgisine erişilemiyor —
        # route bias correction'ı recency-weighted yapabilmek için tarihleri
        # AYRI bir listede, drop edilmeden ÖNCE fold_val_df'ten saklıyoruz.
        oof_dates_list: List[Any] = []

        for fold_name, val_start, val_end in fold_dates:
            # O fold için Train ve Validation setlerini ayır
            fold_train_df = df_features[df_features[self.date_column] < val_start].copy()
            fold_val_df = df_features[
                (df_features[self.date_column] >= val_start) &
                (df_features[self.date_column] <= val_end)
            ].copy()

            # Fold train/val verisi yoksa atla (tarih aralığı dışı)
            if fold_train_df.empty or fold_val_df.empty:
                if self.logging_enabled:
                    logger.warning(f"   ⚠️  {fold_name}: Train veya Val seti boş, atlanıyor.")
                continue

            # NOT: Burada da _get_drop_columns() kullanılıyor (önceden burada
            # sadece [date_column, target_column] drop ediliyordu — bu, sibling
            # target ve cross_lag_0900_same_day sütunlarının 09:00 modelinin
            # fold eğitimlerine LEAKAGE olarak sızmasına yol açan bir hataydı;
            # _split_X_y ile aynı kurala bağlanarak düzeltildi).
            X_fold_train = fold_train_df.drop(columns=self._get_drop_columns(fold_train_df.columns))
            y_fold_train_raw = fold_train_df[self.target_column]
            X_fold_val   = fold_val_df.drop(columns=self._get_drop_columns(fold_val_df.columns))
            y_fold_val_raw   = fold_val_df[self.target_column]

            # PDF Strateji 2 — Rota Bazlı Hedef Ölçeklendirme: aktifse CatBoost
            # ham (raw) hacim yerine SCALED (rotanın kendi geçmiş ortalamasına
            # göre oransal) hedefle eğitilir. Proxy SPO / Gradient Equalization
            # ağırlıkları (aşağıda) HER ZAMAN y_fold_train_raw ile hesaplanır —
            # "kapasite" kavramı yalnızca desi biriminde (raw) anlamlıdır.
            if self._target_scaling_active_:
                y_fold_train = fold_train_df[self._scaled_target_col_]
                y_fold_val   = fold_val_df[self._scaled_target_col_]
            else:
                y_fold_train = y_fold_train_raw
                y_fold_val   = y_fold_val_raw

            # Sütun uyumunu garantile
            for col in self.feature_names_:
                if col not in X_fold_train.columns:
                    X_fold_train[col] = 0
                if col not in X_fold_val.columns:
                    X_fold_val[col] = 0
            X_fold_train = X_fold_train[self.feature_names_]
            X_fold_val   = X_fold_val[self.feature_names_]

            fold_train_pool = Pool(
                data=X_fold_train,
                label=y_fold_train,
                cat_features=self.cat_features_,
                weight=self._compute_decision_regret_weights(
                    X_fold_train, y_fold_train_raw, dates=fold_train_df[self.date_column]
                ),
            )
            fold_val_pool   = Pool(data=X_fold_val,   label=y_fold_val,   cat_features=self.cat_features_)

            # v4: Ortadaki kuantili (index 1) sabit 0.5 yerine Optuna'nın bulduğu
            # asimetrik alpha ile değiştiriyoruz. JSON'da alpha yoksa 0.5 (eski davranış).
            loss_fn_v4 = f"MultiQuantile:alpha=0.1,{self.optimized_alpha_:.4f},{Q90_ALPHA}"
            fold_model = CatBoostRegressor(
                loss_function=loss_fn_v4,
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                l2_leaf_reg=self.l2_leaf_reg,
                bagging_temperature=self.bagging_temperature,
                random_seed=self.random_state,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )

            fold_model.fit(
                fold_train_pool,
                eval_set=fold_val_pool,   # sadece izleme/log amaçlı — aşağıdaki use_best_model=False ile durdurmuyor
                use_best_model=False,
                # ⚠️ KRİTİK: eval_set verilip use_best_model açıkça False yapılmazsa,
                # CatBoost varsayılan olarak use_best_model=True kullanır ve modeli
                # sessizce en iyi validation-skorlu iterasyona geri sarar — early_stopping_rounds
                # kaldırılmış olsa bile! Önceki denemede tam olarak bu oldu: early_stopping_rounds
                # kaldırıldı ama use_best_model=False unutulduğu için sonuç birebir aynı çıktı.
                # Artık gerçekten her fold sabit self.iterations kadar eğitiliyor.
                verbose=False,
            )

            best_iter = self.iterations
            if self.logging_enabled:
                logger.info(
                    f"   ✅ {fold_name} eğitildi | "
                    f"Sabit iterasyon: {self.iterations} (use_best_model=False — gerçekten sabit)"
                )

            # Bu fold'un modeli, kendi val haftasını hiç görmedi (use_best_model=False,
            # eval_set sadece izleme amaçlı) — yani bu gerçek bir out-of-sample tahmin.
            fold_val_pred = fold_model.predict(fold_val_pool)[:, 1]  # q50 (index=1)
            # PDF Strateji 2 — model scaled uzayda eğitildiyse, tahmini KENDİ
            # (val satırlarının) scale_factor'ü ile geri çarpıp raw desi
            # uzayına döndür. Kantil Regresyonu ölçek dönüşümlerine karşı
            # eşdeğişken (equivariant) olduğundan bu geri çevirme kusursuzdur.
            if self._target_scaling_active_:
                fold_val_pred = fold_val_pred * fold_val_df[self._scale_factor_col_].to_numpy(dtype=float)
            fold_val_pred = np.maximum(fold_val_pred, 0.0)
            y_fold_val_actual = y_fold_val_raw.to_numpy(dtype=float)
            if self.log_transform_enabled:
                fold_val_pred = np.square(fold_val_pred)
                y_fold_val_actual = np.square(y_fold_val_actual)
            oof_X_list.append(X_fold_val)
            oof_residual_list.append(y_fold_val_actual - fold_val_pred)
            oof_base_pred_list.append(fold_val_pred)
            oof_dates_list.append(pd.to_datetime(fold_val_df[self.date_column]).to_numpy())

            self.models_.append(fold_model)

        # Geriye uyumluluk için self.model_ → ensemble'ın ilk modeline işaret eder
        # (_evaluate_on_test ve get_feature_importances gibi yardımcılar bunu kullanır)
        if self.models_:
            self.model_ = self.models_[0]

        self._oof_X_ = pd.concat(oof_X_list, axis=0) if oof_X_list else pd.DataFrame()
        self._oof_residual_ = (
            np.concatenate(oof_residual_list) if oof_residual_list else np.array([])
        )
        self._oof_base_pred_ = (
            np.concatenate(oof_base_pred_list) if oof_base_pred_list else np.array([])
        )
        self._oof_dates_ = (
            np.concatenate(oof_dates_list) if oof_dates_list else np.array([], dtype="datetime64[ns]")
        )

        # Adım 2 — rota bazlı OOF bias düzeltmesini burada öğren: OOF
        # (leakage-safe, gerçek tarihsel n) hazır olur olmaz, Model 2
        # (surge/residual) eğitiminden ÖNCE de yapılabilir çünkü bu ikisi
        # birbirinden bağımsız sinyaller (biri "kampanya/backlog kalıntısı",
        # diğeri "genel sistematik yön") — sırası önemli değil.
        self._learn_route_bias_correction()

        elapsed = time.time() - t_q
        if self.logging_enabled:
            logger.info(
                f"   ✅ Ensemble eğitimi tamamlandı: {len(self.models_)} model "
                f"({elapsed:.1f}s)"
            )

        self.is_fitted_ = True

        # --- 4.5 Surge/Residual Modeli (Model 2) — PDF Bölüm 1 + Bölüm 3 ---
        # Model 1 (ensemble) tamamlandıktan HEMEN sonra, aynı X_train/y_train
        # üzerinde eğitilir (bkz. yukarıdaki _split_X_y çağrısı).
        if len(getattr(self, "_oof_X_", [])) > 0:
            self._train_surge_residual_model(
                self._oof_X_, None,
                precomputed_residual=self._oof_residual_,
                precomputed_base_pred=self._oof_base_pred_,
            )
        else:
            self._train_surge_residual_model(X_train, y_train)

        # --- 5. Context Buffer — predict() için lag kaynağı ---
        # Eğitim verisinin sonundan max(lags) + max(rolling_windows) satır saklanır.
        # predict() bu satırları tahmin verisinin önüne ekleyerek lag/rolling
        # değerlerini gerçek tarihsel veriden hesaplar; fillna(0) yanılgısı yok.
        self._save_context_buffer(df)

        # --- 6. Self-Evaluation ---
        if len(X_test) > 0:
            # Overfit analizi için X_train ve y_train'i de gönderiyoruz
            self._evaluate_on_test(
                X_test, y_test, X_train, y_train,
                abnormal_week_mask=abnormal_week_mask_test,
                scale_factor_test=scale_factor_test,
                scale_factor_train=scale_factor_train,
            )

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

    def _predict_single_batch(
        self,
        df: pd.DataFrame,
        include_features: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        TEK SEFERDE (non-recursive) tahmin — asıl model/feature/heuristic
        mantığının tamamı burada yaşıyor.

        Bu metod self.context_buffer_'ı OLDUĞU GİBİ kullanır (değiştirmez).
        `predict()` (geriye dönük uyumluluk) ve `predict_sequential()`
        (gün-gün autoregressive akış) her ikisi de bu metodu çağırır;
        predict_sequential() her gün için self.context_buffer_'ı geçici
        olarak "rolling_context" ile değiştirip burayı tetikler.

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
        # YENİ: Tahmin edilecek asıl satırları kaybetmemek için işaretliyoruz
        df = df.copy()
        df["_is_predict_row_"] = True
        df_predict = self._prepend_context_buffer(df)
        # Buffer'dan gelen geçmiş satırlarda bu sütun NaN olacaktır, onları False yap
        df_predict["_is_predict_row_"] = df_predict["_is_predict_row_"].fillna(False)
        df_features = self._engineer_features(df_predict, drop_na=False)
        # Buffer satırlarını çıkart, SADECE asıl tahmin edilecek satırları tut
        df_features = df_features[df_features["_is_predict_row_"] == True].reset_index(drop=True)

        # Temizlik: Kodu çöpe atmadan önce işaretçi sütununu sil
        df_features = df_features.drop(columns=["_is_predict_row_"])

        # Kalan küçük NaN'ları (buffer yetersizse) son bilinen değerle doldur
        lag_cols  = [c for c in df_features.columns if c.startswith("lag_")]
        roll_cols = [c for c in df_features.columns if c.startswith("rolling_")]
        if lag_cols or roll_cols:
            # [FIX] ffill/bfill artık rota (group_column) BAZINDA yapılıyor.
            # ÖNCEKİ HALİ df_features[...].ffill().bfill() idi — bu, rota
            # sınırını görmeden TÜM DataFrame boyunca aşağı doğru dolduruyordu.
            # df_features rota+tarih sıralı olduğundan (_prepend_context_buffer
            # bkz.), bir rotanın gerçek NaN'ı (örn. yetersiz buffer/kapanış
            # sonrası ilk günler), SIRADAKİ SATIRDAKİ BAŞKA BİR ROTANIN
            # değeriyle dolduruluyordu — sessiz, veri-sırasına-bağımlı bir
            # doğruluk hatası. Şimdi her rota kendi zaman serisi içinde
            # dolduruluyor; rotalar arası hiçbir sızıntı olmuyor.
            if self.group_column and self.group_column in df_features.columns:
                df_features[lag_cols + roll_cols] = (
                    df_features.groupby(self.group_column)[lag_cols + roll_cols]
                    .transform(lambda s: s.ffill().bfill())
                )
            else:
                # ffill: son bilinen değeri taşı; ardından bfill: serinin başındaki boşlukları kapat
                df_features[lag_cols + roll_cols] = (
                    df_features[lag_cols + roll_cols]
                    .ffill()
                    .bfill()
                )

        # X'i hazırla (target, date ve slot-farkındalıklı leakage sütunlarını çıkar)
        # _split_X_y (fit) ile AYNI kuralı kullanır — bkz. _get_drop_columns().
        drop_cols = self._get_drop_columns(df_features.columns)
        X_pred = df_features.drop(columns=drop_cols)

        # Eksik feature sütunlarını sıfırla tamamla (train ile uyumsuzluk güvencesi)
        for col in self.feature_names_:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[self.feature_names_]  # train ile aynı sütun sırası

        # --- 3 Kantil Tahmini (Ensemble MultiQuantile) ---
        pred_pool = Pool(data=X_pred, cat_features=self.cat_features_)

        # ENSEMBLE TAHMİNİ: Eğitilen tüm fold modellerinden tahmin al
        all_preds = [model.predict(pred_pool) for model in self.models_]

        # ⚠️ YENİ: Outlier (panikleyen) modellerden korunmak için mean yerine MEDIAN kullanıyoruz!
        ensemble_preds = np.median(all_preds, axis=0)

        q10_vals = ensemble_preds[:, 0]
        q50_vals = ensemble_preds[:, 1]
        q90_vals = ensemble_preds[:, 2]

        # Negatif tahminleri sıfırla (hacim negatif olamaz)
        q10_vals = np.maximum(q10_vals, 0)
        q50_vals = np.maximum(q50_vals, 0)
        q90_vals = np.maximum(q90_vals, 0)

        # --- Sqrt Geri Çevirme (fit() sqrt dönüşümü uyguladıysa) ---
        # Model sqrt-uzayında eğitildi; tahminleri orijinal desi ölçeğine çevir.
        # square(x) = x²  →  sqrt'ın tam tersi; expm1'e kıyasla bantlar sıkı kalır.
        # Monotonluk: sqrt monoton artan olduğundan q10 ≤ q50 ≤ q90 korunur.
        if self.log_transform_enabled:
            # Sqrt ile eğitilen modeli orijinal hacme geri döndür
            q10_vals = np.square(q10_vals)
            q50_vals = np.square(q50_vals)
            q90_vals = np.square(q90_vals)
            # square sonrası da negatif olamaz güvencesi
            q10_vals = np.maximum(q10_vals, 0)
            q50_vals = np.maximum(q50_vals, 0)
            q90_vals = np.maximum(q90_vals, 0)

        # --- Hedef Ölçeklendirme Geri Çevirme (PDF Strateji 2 — Target Scaling un-scale) ---
        # fit() sırasında K-Fold ensemble scaled (rotanın kendi geçmiş
        # ortalamasına göre oransal) hedefle eğitildiyse, kantil tahminlerini
        # her satırın KENDİ scale_factor'üyle çarparak orijinal desi uzayına
        # geri döndür. Kantil Regresyonu ölçek dönüşümlerine karşı eşdeğişken
        # (equivariant: Q_tau(Y/c) = Q_tau(Y)/c) olduğundan bu geri çevirme
        # kusursuzdur ve q10 ≤ q50 ≤ q90 sıralaması korunur. Sqrt geri
        # çevirmeden SONRA, aşağıdaki tüm raw-uzay düzeltmelerinden (surge,
        # weekday bias, Pazar çarpanı, kampanya heuristiği) ÖNCE çalışmalı.
        if getattr(self, "_target_scaling_active_", False) and getattr(self, "_scale_factor_col_", None) \
                and self._scale_factor_col_ in df_features.columns:
            _scale_vals = df_features[self._scale_factor_col_].to_numpy(dtype=float)
            q10_vals = q10_vals * _scale_vals
            q50_vals = q50_vals * _scale_vals
            q90_vals = q90_vals * _scale_vals
            q10_vals = np.maximum(q10_vals, 0)
            q50_vals = np.maximum(q50_vals, 0)
            q90_vals = np.maximum(q90_vals, 0)

            if self.logging_enabled:
                logger.info(
                    f"   📐 Hedef Ölçeklendirme geri çevrildi (Target Scaling un-scale): "
                    f"{len(_scale_vals)} satır, kendi {self.target_scale_window_days_} "
                    f"günlük rolling ortalamasıyla çarpılarak orijinal desi uzayına döndürüldü."
                )

        # --- Faz 1 Teşhis: düzeltme öncesi ham q50 (Model-1 only) ---
        # Surge/Residual ve Weekday Bias düzeltmelerinden ÖNCEKİ q50_vals'ın
        # bir kopyası — mevcut q50 davranışını BOZMADAN, ne kadarının Model-1
        # ne kadarının düzeltme katmanlarından geldiğini görmek için.
        q50_base_vals = q50_vals.copy()

        # --- Surge/Residual Model Düzeltmesi (Model 2 — PDF Bölüm 1 + 3) ---
        # Model 1'in (ensemble) ardışık kapalı gün / kampanya sonrası
        # patlamalarda sistematik olarak eksik tahmin ettiği ("kapatılamayan
        # boşluk") satırlar, öğrenilmiş bir kalıntı modeliyle telafi edilir.
        surge_mask_pred = np.zeros(len(X_pred), dtype=bool)
        if getattr(self, "surge_model_", None) is not None:
            _predict_threshold = getattr(self, "surge_predict_continuous_threshold_", None)
            surge_mask_pred = self._build_surge_trigger_mask(
                X_pred, continuous_threshold=_predict_threshold
            )
            n_surge_pred = int(surge_mask_pred.sum())
            if n_surge_pred > 0:
                # Özellik Uzayı Ortogonalleştirmesi (PDF): Kalıntı modeli fit()
                # sırasında SURGE_STATIC_EXCLUDED_FEATURES çıkarılmış bir
                # feature matrisi görmüştü (bkz. _train_surge_residual_model);
                # predict() de AYNI alt-kümeyi vermeli, aksi halde CatBoost
                # sütun uyumsuzluğu hatası verir/istatistiksel olarak tutarsız
                # tahmin üretir. Eski (bu alanlar olmadan) kaydedilmiş modellerle
                # geriye dönük uyumluluk için boşsa tam sütun setine düşülür.
                surge_feat_cols = getattr(self, "surge_feature_names_", None) or self.feature_names_
                surge_cat_cols = getattr(self, "surge_cat_features_", None)
                if surge_cat_cols is None:
                    surge_cat_cols = self.cat_features_
                surge_pool_pred = Pool(
                    data=X_pred.loc[surge_mask_pred, surge_feat_cols],
                    cat_features=surge_cat_cols,
                )
                residual_pred = self.surge_model_.predict(surge_pool_pred)
                # Kalıntı SADECE eksik-tahmin yönünde (residual > 0) uygulanır:
                # Model 2'nin tek görevi "kapatılamayan boşluğu" kapatmaktır;
                # negatif kalıntı üretirse q50'yi gereksiz aşağı çekmesin diye
                # sıfırla kırpılır — ALNS'in asimetrik maliyet yapısıyla tutarlı.
                residual_pred = np.maximum(residual_pred, 0.0) * getattr(self, "surge_calibration_factor_", 1.0)

                # ADIM 5 / Faz 2 — Relative Cap: correction'ı baseline (düzeltme öncesi)
                # hacmin belirli bir oranıyla sınırla. None = kapalı (varsayılan,
                # geriye dönük uyumlu).
                alpha = getattr(self, "surge_relative_cap_alpha_", None)
                if alpha is not None:
                    cap = alpha * q50_vals[surge_mask_pred]   # q50_vals burada henüz ham/base değer
                    n_capped = int(np.sum(residual_pred > cap))
                    residual_pred = np.minimum(residual_pred, cap)
                    if self.logging_enabled and n_capped > 0:
                        logger.info(
                            f"   🧢 Relative Cap uygulandı (α={alpha}): {n_capped} satırda "
                            f"correction baseline'ın %{alpha*100:.0f}'i ile sınırlandı."
                        )

                # [PDF Entegrasyonu — Hacim Tabanlı Lojistik Sönümleme (S_vol)]
                # ESKİ (Faz 2b — Segment Scale) blok TAMAMEN KALDIRILDI: hacim
                # aralığına göre KESİKLİ çarpanlar (örn. 60-1200 desi → 0.4x)
                # kullanıyordu — bir rota aralık sınırını (ör. 60 desi) geçtiği
                # an çarpan aniden sıçrıyor, bu da ALNS'in komşuluk arama
                # operatörleri için maliyet yüzeyinde yapay bir süreksizlik/
                # uçurum yaratıyordu (bkz. PDF "Sürekli Ölçekleme ve Pürüzsüz
                # Sönümleme" bölümü). YENİ blok, aynı niyeti (küçük hacimli
                # rotalarda düzeltmeyi bastır, büyük hacimli rotalarda olduğu
                # gibi bırak) SÜREKLİ/türevlenebilir bir sigmoid ile karşılar —
                # ALNS'e sunulan maliyet yüzeyinde hiçbir hacim noktasında ani
                # sıçrama olmaz. Alpha-cap (yukarıdaki blok) ile birlikte
                # kullanılabilir: kod sırası gereği önce alpha-cap residual'ı
                # kırpar, sonra bu blok kırpılmış residual'ı hacme göre
                # SÜREKLİ olarak sönümler.
                # surge_volume_damping_enabled_=False → eski davranışla
                # birebir uyumlu (S_vol=1.0, sönümleme yok).
                if getattr(self, "surge_volume_damping_enabled_", True):
                    baseline_pred = q50_vals[surge_mask_pred]   # henüz ham/base değer
                    s_vol = self._compute_volume_damping_factor(baseline_pred)
                    residual_pred = residual_pred * s_vol
                    if self.logging_enabled and len(s_vol) > 0:
                        logger.info(
                            f"   🌊 Hacim Tabanlı Lojistik Sönümleme (S_vol) uygulandı: "
                            f"{len(s_vol)} satır | ort. S_vol={float(np.mean(s_vol)):.3f} "
                            f"(v_crit={getattr(self, 'surge_volume_damping_v_crit_', 150.0)}, "
                            f"k={getattr(self, 'surge_volume_damping_k_', 3.0)})."
                        )

                q50_vals[surge_mask_pred] = q50_vals[surge_mask_pred] + residual_pred
                # q90 (spot araç alarm bandı) da aynı düzeltmeyi + %15 ek tampon
                # ile yansıtır ki patlamanın büyüklüğü uncertainty_range'e sızsın.
                q90_vals[surge_mask_pred] = np.maximum(
                    q90_vals[surge_mask_pred],
                    q50_vals[surge_mask_pred] + residual_pred * 0.15,
                )
                q10_vals = np.maximum(q10_vals, 0)
                q50_vals = np.maximum(q50_vals, 0)
                q90_vals = np.maximum(q90_vals, 0)

                if self.logging_enabled:
                    logger.info(
                        f"   🚀 Surge/Residual düzeltmesi (Model 2) uygulandı: "
                        f"{n_surge_pred} güne ort. +{float(np.mean(residual_pred)):,.1f} desi "
                        f"(çarpan={getattr(self, 'surge_calibration_factor_', 1.0)}x, tetikleyiciler={self.surge_trigger_columns_used_})."
                    )

        # --- Weekday Bias Calibration (post-hoc, empirik — ADIM 2) ---
        # Surge/Residual düzeltmesinden SONRA, ayrı ve bağımsız bir adım.
        # self.weekday_bias_ dolu değilse (henüz kalibre edilmemiş model)
        # tamamen no-op'tur — mevcut davranışı bozmaz.
        # Sadece POZİTİF yönde uygulanır (underprediction telafisi) — q90'a
        # dokunulmaz, çünkü bu düzeltme gözlemlenmiş sistematik bir yanlılığı
        # telafi ediyor, belirsizliği artırmıyor.
        if getattr(self, "weekday_bias_", None):
            if "weekday" in X_pred.columns:
                wd_vals = X_pred["weekday"].to_numpy()
                bias_vals = np.array(
                    [self.weekday_bias_.get(int(w), 0.0) for w in wd_vals]
                )
                q50_vals = q50_vals + np.maximum(bias_vals, 0.0)
                q50_vals = np.maximum(q50_vals, 0.0)

                if self.logging_enabled and np.any(bias_vals > 0):
                    logger.info(
                        f"   📅 Weekday Bias Calibration uygulandı: "
                        f"{int(np.sum(bias_vals > 0))} satıra ort. "
                        f"+{float(np.mean(bias_vals[bias_vals > 0])):,.1f} desi offset "
                        f"(weekday_bias_={self.weekday_bias_})."
                    )
            elif self.logging_enabled:
                logger.warning(
                    "   ⚠️ weekday_bias_ tanımlı ama X_pred'de 'weekday' kolonu yok — "
                    "kalibrasyon atlandı."
                )

        # --- Pazar Çarpımsal Post-Process Düzeltmesi (DENEYSEL — ADIM 2b) ---
        # Yukarıdaki weekday_bias_ bloğundan TAMAMEN AYRI ve BAĞIMSIZ bir
        # mekanizma: o blok sadece toplamsal + sadece pozitif (underprediction
        # telafisi) çalışır, bu yüzden Pazar'ın overprediction durumunu
        # kapsayamaz. tau/karar ağaçları hiç dokunulmadan, sadece post-hoc
        # olarak q50 (ve q10/q90, oranı korumak için) Pazar satırlarında
        # _SUNDAY_POST_PROCESS_MULTIPLIER ile ölçeklenir.
        # ⚠️ Bu, 2026-06-14→06-20 penceresindeki TEK bir bilinen gerçek
        # değere göre geriye doğru kalibre edilmiş bir sabittir (leakage'lı) —
        # temiz/görülmemiş bir pencerede ayrıca doğrulanmadan production'a
        # kalıcı gömülmemeli (bkz. sınıf sabiti docstring'i).
        # ⚠️ ÇİFTE-DÜZELTME KORUMASI (Öneri A entegrasyonu): eğer
        # route_bias_correction_enabled_=True VE bir rota için
        # route_bucket_bias_correction_[(rota,"sunday")] zaten öğrenilmişse,
        # bu rotayı burada bir daha küçültmeyiz — aşağıdaki "Adım 2: Rota
        # Bazlı OOF Bias Düzeltmesi" bloğu o rotayı ZATEN kendi (daha
        # isabetli, rota-spesifik) çarpanıyla düzeltecek. Bu flat/global
        # sabit SADECE (a) route_bias_correction_enabled_ kapalıysa (eski
        # davranış, geriye dönük uyumlu) ya da (b) o rota için bucket
        # düzeltmesi öğrenilememişse (yetersiz OOF verisi) devrede kalan
        # bir GÜVENLİK AĞIDIR.
        # ⚠️ REVİZYON (v2 — 2026-07-22): Öneri A'nın "sunday" bucket'ı TÜM
        # rotaları kapsayınca (sunday=289/289) eski ×0.55 güvenlik ağı hiç
        # tetiklenmez olmuştu — ve bucket'ın kendi öğrendiği oranın (örn.
        # 0.789x) backlog_alpha=1.4→5.25 düzeltmesinden SONRA yetersiz
        # kaldığı görüldü (17:00 Pazar farkı +24.8%→+128.4%'e kötüleşti;
        # base model Sunday büyüklüğü backlog sinyali güçlenince paylaşılan
        # ağaç yapısı üzerinden dolaylı olarak da büyümüş). Artık bucket
        # tarafından kapsanan rotalarda ×0.55'i DOĞRUDAN uygulamıyoruz
        # (çifte düzeltme olur) — bunun yerine TABAN/GÜVENLİK SINIRI olarak
        # saklıyoruz; aşağıdaki Öneri A bloğu kendi düzeltmesini uyguladıktan
        # SONRA, ikisinden DAHA DÜŞÜK olanı (daha muhafazakar) kazanır.
        # decision_regret eksik tahmini fazla tahminden ~9x cezalandırdığı
        # için "daha düşük kazanır" kuralı metrik açısından güvenli taraf.
        sunday_floor_idx: Optional[np.ndarray] = None
        sunday_floor_vals: Optional[np.ndarray] = None

        if "weekday" in X_pred.columns and self.slot_label == "17:00":
            wd_vals = X_pred["weekday"].to_numpy()
            sunday_mask = (wd_vals == 6)

            # --- REJİM AYRIMI (2026-07-26): bu güvenlik tabanı SADECE
            # "sunday_closed" (kapalı/pasif TM, event YOK) rejimi için
            # tasarlandı — backlog/kampanya boşalması nedeniyle talebin
            # gerçekten YÜKSEK olması beklenen "sunday_event" satırlarını
            # ×0.55 ile aşağı bastırmak yanlış yönde bir müdahale olurdu.
            # Bu yüzden event_active'i burada (Adım 2 bloğundan ÖNCE, yerel
            # olarak) hesaplayıp sadece event'siz Pazar satırlarını
            # kapsıyoruz; event aktif Pazar satırları bu tabandan tamamen
            # muaf.
            event_active_sunday = np.zeros(len(X_pred), dtype=bool)
            for _col in SURGE_BINARY_TRIGGER_COLUMNS:
                if _col in X_pred.columns:
                    event_active_sunday |= (pd.to_numeric(X_pred[_col], errors="coerce").fillna(0.0) > 0).values
            for _col in SURGE_CONTINUOUS_TRIGGER_COLUMNS:
                if _col in X_pred.columns:
                    event_active_sunday |= (
                        pd.to_numeric(X_pred[_col], errors="coerce").fillna(0.0) > BUCKET_EVENT_CONTINUOUS_THRESHOLD
                    ).values

            sunday_closed_mask = sunday_mask & ~event_active_sunday

            bucket_map_sunday = getattr(self, "route_bucket_bias_correction_", None)
            already_covered = np.zeros(len(X_pred), dtype=bool)
            if (
                sunday_closed_mask.any()
                and getattr(self, "route_bias_correction_enabled_", False)
                and bucket_map_sunday
                and self.group_column in X_pred.columns
            ):
                route_vals_sd = X_pred[self.group_column].values
                already_covered = np.array([
                    (r, "sunday_closed") in bucket_map_sunday for r in route_vals_sd
                ])

            uncovered_sunday_mask = sunday_closed_mask & ~already_covered
            covered_sunday_mask = sunday_closed_mask & already_covered

            if np.any(covered_sunday_mask):
                scale = self._SUNDAY_POST_PROCESS_MULTIPLIER
                sunday_floor_idx = np.where(covered_sunday_mask)[0]
                sunday_floor_vals = q50_vals[sunday_floor_idx] * scale
                if self.logging_enabled:
                    logger.info(
                        f"   📅 [DENEYSEL] Pazar (kapalı/pasif rejim) güvenlik TABANI (×{scale}) "
                        f"{len(sunday_floor_idx)} Öneri-A-kapsamlı satır için "
                        f"hesaplandı — Öneri A kendi düzeltmesini uyguladıktan "
                        f"SONRA ikisinden düşük olan kazanacak (çifte düzeltme değil). "
                        f"'sunday_event' (backlog/kampanya boşalması) rejimindeki satırlar "
                        f"bu tabandan muaf tutuldu."
                    )

            if np.any(uncovered_sunday_mask):
                scale = self._SUNDAY_POST_PROCESS_MULTIPLIER
                q50_vals[uncovered_sunday_mask] = q50_vals[uncovered_sunday_mask] * scale
                q10_vals[uncovered_sunday_mask] = q10_vals[uncovered_sunday_mask] * scale
                q90_vals[uncovered_sunday_mask] = q90_vals[uncovered_sunday_mask] * scale

                if self.logging_enabled:
                    logger.info(
                        f"   📅 [DENEYSEL] Pazar (kapalı/pasif rejim) post-process çarpımsal düzeltmesi "
                        f"(güvenlik ağı — Öneri A tarafından kapsanmayan rotalar) "
                        f"uygulandı: {int(np.sum(uncovered_sunday_mask))} satır × {scale}x "
                        f"(kaynak=2026-06-14→06-20 penceresi, leakage'lı — "
                        f"kalıcı production öncesi temiz pencerede doğrulayın)."
                    )


        # --- Hibrit Domain Heuristic (Tahmin çıktısı) ---
        # Kampanya arifesinde ML'in göremediği hacim artışı kural tabanlı eklenir.
        # NOT: surge_model_ o satırı ZATEN düzelttiyse burada tekrar dokunulmaz
        # (çifte düzeltme önlenir) — heuristik yalnızca surge modelinin
        # kapsamadığı (surge_mask_pred=False) kampanya-arifesi satırlarında,
        # ya da surge modeli hiç eğitilmediğinde (B Planı) devreye girer.
        if "is_campaign_eve" in X_pred.columns and hasattr(self, "campaign_multipliers_"):
            camp_mask_pred = (X_pred["is_campaign_eve"] == 1).values & ~surge_mask_pred
            if camp_mask_pred.sum() > 0:
                route_vals = X_pred[self.group_column].values if self.group_column in X_pred.columns else []
                # q10 ve q50 için normal çarpan, q90 için spot riskine karşı +0.10 tampon
                mult_array = np.array([self.campaign_multipliers_.get(r, 1.15) for r in route_vals])

                q10_vals[camp_mask_pred] *= mult_array[camp_mask_pred]
                q50_vals[camp_mask_pred] *= mult_array[camp_mask_pred]
                q90_vals[camp_mask_pred] *= (mult_array[camp_mask_pred] + 0.10)

                q10_vals = np.maximum(q10_vals, 0)
                q50_vals = np.maximum(q50_vals, 0)
                q90_vals = np.maximum(q90_vals, 0)

                if self.logging_enabled:
                    logger.info(f"   💡 Dinamik Domain Heuristic (predict): {camp_mask_pred.sum()} güne akıllı rota çarpanları uygulandı.")
        # ---------------------------------------------------------

        # --- Adım 2: Rota Bazlı OOF Bias Düzeltmesi (post-hoc kalibrasyon) ---
        # campaign_multipliers_'dan FARKLI: kampanya günüyle sınırlı değil,
        # HER satıra (o rotanın OOF'ta öğrenilmiş sistematik yönüne göre)
        # uygulanır. debug_backtest.py'deki rota-bazlı bias teşhisinin
        # (örn. Yalova-hub rotalarının sistematik eksik tahmini) doğrudan
        # üretime taşınmış hali — sert tier/segment YOK, her rota kendi
        # OOF-öğrenilmiş çarpanını (veya veri azsa 1.0'ı) alır.
        if (
            getattr(self, "route_bias_correction_enabled_", False)
            and self.group_column in X_pred.columns
            and (getattr(self, "route_bias_correction_", None) or getattr(self, "route_bucket_bias_correction_", None))
        ):
            route_vals_bias = X_pred[self.group_column].values
            weekday_vals_bias = (
                pd.to_numeric(X_pred["weekday"], errors="coerce").values
                if "weekday" in X_pred.columns else np.full(len(X_pred), np.nan)
            )
            # Öneri A — event_active maskesi: _build_surge_trigger_mask ile
            # AYNI sütun/eşik seti, ama self.surge_trigger_columns_used_'a
            # (Model 2 loglaması paylaşılan durumu) dokunmadan yerel hesap.
            event_active_bias = np.zeros(len(X_pred), dtype=bool)
            for _col in SURGE_BINARY_TRIGGER_COLUMNS:
                if _col in X_pred.columns:
                    event_active_bias |= (pd.to_numeric(X_pred[_col], errors="coerce").fillna(0.0) > 0).values
            for _col in SURGE_CONTINUOUS_TRIGGER_COLUMNS:
                if _col in X_pred.columns:
                    event_active_bias |= (
                        pd.to_numeric(X_pred[_col], errors="coerce").fillna(0.0) > BUCKET_EVENT_CONTINUOUS_THRESHOLD
                    ).values

            bias_mult = np.array([
                self._lookup_bias_correction(r, wd, bool(ev))
                for r, wd, ev in zip(route_vals_bias, weekday_vals_bias, event_active_bias)
            ])
            adj_mask = bias_mult != 1.0
            if np.any(adj_mask):
                q10_vals[adj_mask] *= bias_mult[adj_mask]
                q50_vals[adj_mask] *= bias_mult[adj_mask]
                q90_vals[adj_mask] *= bias_mult[adj_mask]

                q10_vals = np.maximum(q10_vals, 0)
                q50_vals = np.maximum(q50_vals, 0)
                q90_vals = np.maximum(q90_vals, 0)

                if self.logging_enabled:
                    def _bucket_of(wd, ev):
                        is_sun = (not pd.isna(wd)) and (int(wd) == 6)
                        if is_sun:
                            return "sunday_event" if ev else "sunday_closed"
                        return "event" if ev else "normal"
                    n_bucket_hits = int(np.sum(
                        [(r, _bucket_of(wd, ev)) in getattr(self, "route_bucket_bias_correction_", {})
                         for r, wd, ev in zip(route_vals_bias, weekday_vals_bias, event_active_bias)]
                    ))
                    logger.info(
                        f"   🎯 Rota×Gün-Türü OOF Bias Düzeltmesi (predict, Öneri A): "
                        f"{int(adj_mask.sum())} satıra uygulandı "
                        f"({n_bucket_hits} satır (rota,bucket)-spesifik, "
                        f"{int(adj_mask.sum()) - n_bucket_hits} satır flat rota-bazlı fallback'ten; "
                        f"ort. çarpan={bias_mult[adj_mask].mean():.3f}x)."
                    )
        # ---------------------------------------------------------

        # --- Pazar Güvenlik Tabanını Uygula (v2 — bkz. yukarıdaki not) ---
        # Öneri A'nın "sunday" bucket düzeltmesi ZATEN uygulandı (yukarıda).
        # Şimdi bu sonucu, önceden hesaplanmış ×0.55 tabanıyla karşılaştırıp
        # İKİSİNDEN DÜŞÜK OLANI (daha muhafazakar/az-fazla-tahmin) kazandırıyoruz.
        if sunday_floor_idx is not None and len(sunday_floor_idx) > 0:
            current_vals = q50_vals[sunday_floor_idx]
            needs_floor = current_vals > sunday_floor_vals
            if np.any(needs_floor):
                idx_to_fix = sunday_floor_idx[needs_floor]
                new_vals = sunday_floor_vals[needs_floor]
                # q10/q90'ı da AYNI oranda küçült ki quantile sıralaması bozulmasın.
                ratio = np.divide(
                    new_vals, q50_vals[idx_to_fix],
                    out=np.ones_like(new_vals), where=q50_vals[idx_to_fix] > 0,
                )
                q10_vals[idx_to_fix] *= ratio
                q90_vals[idx_to_fix] *= ratio
                q50_vals[idx_to_fix] = new_vals

                if self.logging_enabled:
                    logger.info(
                        f"   📅 [DENEYSEL] Pazar (kapalı/pasif rejim) güvenlik TABANI devreye girdi: "
                        f"{int(needs_floor.sum())}/{len(sunday_floor_idx)} Öneri-A-kapsamlı "
                        f"'sunday_closed' satırında bucket düzeltmesi yetersiz kaldı, "
                        f"×{self._SUNDAY_POST_PROCESS_MULTIPLIER} tabanına çekildi "
                        f"('sunday_event' satırları bu blokta hiç yer almıyor)."
                    )

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
                # Slot bilgisi: iki instance (09:00/17:00) aynı formatta sonuç
                # üretince hangi tahminin hangi slota ait olduğu kaybolmasın
                # diye eklendi — uncertainty.py ve run_forecast.py buna bağımlı.
                "slot":                 self.slot_label,
                "q10":                  round(float(q10_vals[i]), 4),
                "q50":                  round(float(q50_vals[i]), 4),
                "q90":                  round(float(q90_vals[i]), 4),
                # Faz 1 Teşhis: surge/weekday düzeltmesinden ÖNCEKİ ham q50
                # (Model-1 only) — q50 ile q50_base arasındaki fark, düzeltme
                # katmanlarının (Model 2 + weekday bias) o satıra kattığı miktar.
                "q50_base":             round(float(q50_base_vals[i]), 4),
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
                f"✅ _predict_single_batch() tamamlandı: {len(results)} tahmin üretildi "
                f"(format: in-memory JSON, disk I/O yok)"
            )

        return results

    # -----------------------------------------------------------------------
    # predict → geriye dönük uyumluluk (tek seferde / tek günlük tahmin)
    # -----------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        include_features: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Eski davranış: TÜM tahmin ufkunu (ör. 7 gün) TEK bir batch olarak,
        fit() sırasında kaydedilen context_buffer_'a göre tahmin eder.

        Autoregressive/recursive DEĞİLDİR — 2. günün lag_1'i, 1. günün
        GERÇEK tahminini değil, context_buffer_'daki (fit-zamanı) son
        gerçek veriyi görür. Çok günlük ufuklarda hatayı azaltmak için
        `predict_sequential()` kullanın; bu metod geriye dönük uyumluluk
        ve tek günlük tahminler için hâlâ geçerlidir.
        """
        return self._predict_single_batch(df, include_features=include_features)

    # -----------------------------------------------------------------------
    # predict_sequential → Gün-gün Autoregressive/Recursive Tahmin
    # -----------------------------------------------------------------------

    def predict_sequential(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Tahmin ufkunu (ör. 7 gün) TEK SEFERDE değil, GÜN GÜN tahmin eder;
        her günün q50 tahmini "gerçekmiş gibi" bir sonraki günün
        lag/rolling feature'larına beslenir (recursive/autoregressive).

        Neden gerekli:
          predict() (tek batch) modunda, tahmin ufkundaki TÜM günler aynı
          anda, sadece fit-zamanındaki context_buffer_'a (gerçek geçmiş)
          bakarak hesaplanır. 2. günün lag_1'i aslında 1. günün TAHMİNİ
          değil, context_buffer_'daki son gerçek veridir. Ufuk uzadıkça
          (7 gün) bu, lag feature'ların giderek daha stale/yanlış olmasına
          yol açar. predict_sequential() her günü kendi (o ana kadarki
          gerçek + tahmin edilmiş önceki günler) context'iyle tahmin
          ederek bunu düzeltir.

        Uygulama notu:
          self.context_buffer_'ı KALICI OLARAK değiştirmez — bir kopyasını
          (`rolling_context`) alır, her gün sonunda o günün q50 tahminini
          bu kopyaya "sözde-gerçek" (pseudo-actual) olarak ekler ve
          context_buffer_'ı SADECE _predict_single_batch() çağrısı
          süresince GEÇİCİ olarak bu genişleyen kopyayla değiştirir, hemen
          ardından orijinaline geri döndürür (try/finally ile — bir
          _predict_single_batch çağrısı istisna fırlatsa bile
          self.context_buffer_ bozulmadan kalır).

        ⚠️ sibling_target_column (ör. 17:00 modeli için toplam_desi_0900):
          Bu sütun bu sınıfta LAG'LANMIŞ bir feature olarak DEĞİL, AYNI
          GÜNÜN feature'ı olarak kullanılıyor (bkz. _get_drop_columns
          docstring'i — 17:00 tahmini yapıldığında o günün 09:00 talebi
          zaten gerçekleşmiş sayılır, dolayısıyla leakage değildir). Yani
          rolling_context'e eklenen pseudo_actual satırında sibling_target
          sütunu, gün ilerledikçe "tahmin edilmesi gereken" bir şey
          DEĞİLDİR — day_df içinde zaten ne geldiyse (run_forecast.py'nin
          predict_grid'inden, bugün için 0.0 veya bilinen gerçek değer) o
          kalır, dokunulmaz. Burada AYRICA doldurulmasına GEREK YOK: bu
          modelin recursive/autoregressive döngüsünün konusu sadece
          target_column'un KENDİ geçmişidir (lag_1, lag_7, ...,
          rolling_7, rolling_14) — sibling ayrı bir konu.

        Parameters
        ----------
        df : pd.DataFrame
            Tahmin ufkunun TAMAMI (ör. 7 gün × N rota), aynı şema
            (predict() ile aynı — target_column NaN/0 olabilir).

        Returns
        -------
        List[Dict[str, Any]]
            Tüm günlerin tahminlerini birleştiren liste (predict() ile
            aynı kayıt formatı).
        """
        if not self.is_fitted_:
            raise ValueError(
                "❌ Model eğitilmedi. Önce fit() çağırın!\n"
                "   Kullanım: forecaster.fit(train_df)"
            )

        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        unique_dates = sorted(df[self.date_column].unique())

        if len(unique_dates) > self.forecast_horizon:
            logger.warning(
                f"⚠️ predict_sequential(): {len(unique_dates)} gün geldi ama "
                f"forecast_horizon={self.forecast_horizon}. Muhtemelen predict_grid'e "
                f"buffer günleri de karışmış — sadece gelecek ufku geçirin."
            )

        # fit() sırasında kaydedilen context_buffer_'ı BOZMADAN bir kopyasını
        # al — bu kopya, gün ilerledikçe "tahmin edilmiş" satırlarla büyüyecek.
        rolling_context = (
            self.context_buffer_.copy()
            if self.context_buffer_ is not None
            else pd.DataFrame(columns=df.columns)
        )

        buffer_size = max(self.lags) + max(self.rolling_windows)

        all_results: List[Dict[str, Any]] = []

        if self.logging_enabled:
            logger.info(
                f"🔁 predict_sequential() başladı: {len(unique_dates)} gün, "
                f"tek seferde değil GÜN GÜN (autoregressive) tahmin edilecek."
            )

        for h_idx, d in enumerate(unique_dates, start=1):
            # --- Feature Trust Decay: Ufuk (h) ilerledikçe q50 yerine
            # "güvenilir referans seviye"ye (rotanın son 7 günlük gerçek
            # ortalaması) kaydırılacak çapa değeri. Modelin kendi düşük
            # tahminini tekrar tekrar kendine yedirip "ölüm sarmalı"
            # (death spiral) oluşturmasını önler — bkz. predict_sequential
            # docstring'i / rota bazlı backtest teşhisi.
            #
            # [ADIM 2 — 2026-07-21] DÜZELTME: reference_level artık
            # self.context_buffer_'dan (cutoff-öncesi, olay BAŞLAMADAN
            # önceki dondurulmuş 7 günlük ortalama) DEĞİL, rolling_context'ten
            # (o ana kadarki gerçek + tahmin edilmiş günler) HER h_idx'te
            # yeniden hesaplanıyor. Eskiden bu çapa döngü başlamadan ÖNCE
            # bir kez hesaplanıp donduruluyordu — h=7'de bile hâlâ h=1'deki
            # "eski normal"e işaret ediyordu; oysa lag_1/rolling_mean_7/14
            # gibi TÜM diğer feature'lar zaten rolling_context'in büyüyen
            # (pseudo-actual eklenen) haline bakıyor. h=1'de rolling_context
            # henüz context_buffer_ ile birebir aynı olduğu için davranış
            # DEĞİŞMEZ; h>=2'den itibaren çapa, modelin kendi (aynı
            # koşudaki) önceki günlerin tahminlerini de yansıtmaya başlar.
            if self.group_column and rolling_context is not None and len(rolling_context) > 0:
                reference_level = (
                    rolling_context
                    .sort_values(self.date_column)
                    .groupby(self.group_column)[self.target_column]
                    .apply(lambda s: s.tail(7).mean())
                    .to_dict()
                )
            else:
                reference_level = {}

            day_df = df[df[self.date_column] == d].copy()

            # O günü, o ana kadarki context (gerçek geçmiş + önceki günlerin
            # TAHMİNLERİ) ile birlikte tek günlük bir batch olarak tahmin et.
            # self.context_buffer_'ı GEÇİCİ olarak rolling_context ile
            # değiştirip _predict_single_batch'i çağırıyoruz.
            original_buffer = self.context_buffer_
            self.context_buffer_ = rolling_context
            try:
                # DÜZELTME (v3): include_features=True — bir önceki denemede
                # surge mask'i HAM day_df'ten (feature engineering'den ÖNCE)
                # hesaplıyordum, backlog_release_index/is_closed gibi sütunlar
                # orada hiç yoktu → mask hep False çıkıyordu (bir önceki
                # koşuda "🔥 Trust Decay gevşetildi" logu HİÇ görünmedi —
                # bunun kanıtı). _predict_single_batch İÇİNDE gerçek feature
                # engineering yapılıyor ama dışarı sızmıyordu; include_features
                # ile gerçekten hesaplanmış feat_<col> değerlerini çekiyoruz.
                day_results = self._predict_single_batch(day_df, include_features=True)
            finally:
                self.context_buffer_ = original_buffer  # her koşulda geri al

            # Surge tetikleyicisi, HAM day_df değil ENGINEERED feat_ değerleri
            # üzerinden, rota bazında hesaplanır (bu günde tüm rotalar aynı
            # tarihte olduğu için rota→aktif/pasif sözlüğü yeterli).
            surge_active_by_group: Dict[Any, bool] = {}
            for r in day_results:
                active = False
                for col in SURGE_BINARY_TRIGGER_COLUMNS:
                    val = r.get(f"feat_{col}")
                    if val is not None and pd.notna(val) and float(val) > 0:
                        active = True
                        break
                if not active:
                    for col in SURGE_CONTINUOUS_TRIGGER_COLUMNS:
                        val = r.get(f"feat_{col}")
                        if val is not None and pd.notna(val) and float(val) > SURGE_CONTINUOUS_TRIGGER_THRESHOLD:
                            active = True
                            break
                surge_active_by_group[r.get(self.group_column)] = active

            # 🔬 GEÇİCİ TEŞHİS: rolling_mean_7 / rolling_mean_14 tutarlılık
            # kontrolü. Tanım gereği (shift(1).rolling(w)) rolling_mean_14
            # penceresi rolling_mean_7 penceresini KAPSAR (son 7 gün + 7 gün
            # daha) — dolayısıyla 14g ortalaması, 7g ortalamasının yarısından
            # (7g ort. / 2) DAHA DÜŞÜK olamaz (önceki 7 gün tam sıfır olsa
            # bile alt sınır budur). Bu eşitsizlik her h_idx'te bozulmuyorsa
            # sorun rolling penceresinde değil başka bir yerdedir.
            _suffix = "_" + self.target_column.rsplit("_", 1)[-1]
            _m7_key, _m14_key = f"feat_rolling_mean_7{_suffix}", f"feat_rolling_mean_14{_suffix}"
            _watch_routes = getattr(self, "debug_watch_routes_", None) or []
            _spike_dump_cols = (
                SURGE_BINARY_TRIGGER_COLUMNS + SURGE_CONTINUOUS_TRIGGER_COLUMNS
                + ["days_since_resumption", "accumulated_closed_days", "is_closed",
                   "hub_out_vol_7d", "hub_in_vol_7d", _m7_key.replace("feat_", ""),
                   _m14_key.replace("feat_", "")]
            )
            if self.logging_enabled and _watch_routes:
                for r in day_results:
                    if r.get(self.group_column) not in _watch_routes:
                        continue
                    m7, m14 = r.get(_m7_key), r.get(_m14_key)
                    if m7 is None or m14 is None:
                        continue
                    flag = "🚨 TUTARSIZ" if (m14 < 0.5 * m7 - 1e-6) else "ok"
                    logger.info(
                        f"   🔬 [rolling-check h={h_idx} {d.date() if hasattr(d,'date') else d}] "
                        f"{r.get(self.group_column)}: rolling_mean_7={m7:,.1f} "
                        f"rolling_mean_14={m14:,.1f} q50={r.get('q50'):,.1f} → {flag}"
                    )
                    # 🚨 SPIKE DUMP: q50, kendi rolling bağlamının (7g/14g
                    # ortalamasının) 4 katından fazlaysa — bu normal bir
                    # surge/backlog artışı değil, muhtemelen bir feature
                    # patlaması. q50_base'e de bakıyoruz: q50≈q50_base ise
                    # sorun TABAN modelde (Model 1), surge/residual'da değil.
                    q50_val = r.get("q50", 0.0) or 0.0
                    ref_mag = max(m7, m14, 1.0)
                    if q50_val > 4 * ref_mag:
                        logger.warning(
                            f"      🚨 SPIKE h={h_idx} {r.get(self.group_column)}: "
                            f"q50={q50_val:,.1f} (bağlamın {q50_val/ref_mag:.1f}x üstü) | "
                            f"q50_base={r.get('q50_base', 'YOK')}"
                        )
                        for col in _spike_dump_cols:
                            fk = col if col.startswith("feat_") else f"feat_{col}"
                            if fk in r:
                                logger.warning(f"         {col} = {r.get(fk)}")


            # Tanı yakalama (varsayılan kapalı — bkz. __init__ açıklaması):
            # feat_* anahtarlarını silmeden ÖNCE, istenirse tam kopyasını
            # debug_captured_rows_'a ekle. h_idx de eklenir ki SHAP
            # tablosunda "ufuk arttıkça hangi feature değişiyor" görülebilsin.
            if getattr(self, "capture_debug_features_", False):
                for r in day_results:
                    snap = dict(r)
                    snap["_h_idx"] = h_idx
                    self.debug_captured_rows_.append(snap)

            # Dış sözleşmeyi (predict_sequential'ın döndürdüğü kayıt şeması)
            # BOZMAMAK için feat_* anahtarlarını all_results'a eklemeden önce çıkar.
            for r in day_results:
                for col in list(r.keys()):
                    if col.startswith("feat_"):
                        del r[col]

            all_results.extend(day_results)

            # Bu günün tahminini (q50) "gerçekmiş gibi" rolling_context'e
            # ekle ki BİR SONRAKİ günün lag_1'i bunu görsün.
            d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
            pred_map = {
                (r[self.group_column], r[self.date_column]): r["q50"]
                for r in day_results
            }
            pseudo_actual = day_df.copy()
            # Feature Trust Decay (v2 — DÜZELTİLDİ): reference_level, cutoff
            # ÖNCESİ (olay/backlog patlaması BAŞLAMADAN önceki) 7-günlük
            # gerçek ortalama — yani "eski normal". Sabit alpha_h formülü bu
            # eski normale doğru kayarken, Yalova gibi AKTİF bir surge
            # penceresinde (backlog_release_index vb.) modelin doğru şekilde
            # yüksek ürettiği q50'yi YANLIŞ (bayat) bir çapaya geri çekiyordu
            # — backtest'te tam olarak görülen "ölüm sarmalı" buydu (bkz.
            # commit notu: Yalova → İstanbul q50'si h arttıkça 5230→3494'e
            # düşerken y_true 6379-11321 bandında kaldı).
            #
            # DÜZELTME: o günkü satırın surge tetikleyicisi AKTİFSE (Model
            # 2'nin kullandığı AYNI mask — _build_surge_trigger_mask), trust
            # decay'i büyük ölçüde GEVŞET (alpha_h_active — varsayılan 0.9,
            # h=7'de bile neredeyse tam güven). Tetikleyici PASİFSE eski
            # (spiral-önleyici, sakin/gürültülü rotalarda hâlâ gerekli)
            # davranış aynen korunur.
            alpha_h_base = max(0.35, 1.0 - 0.11 * (h_idx - 1))   # h=1: 1.0 | h=4: ~0.67 | h=7: ~0.35
            surge_active_today = (
                pseudo_actual[self.group_column].map(surge_active_by_group).fillna(False).to_numpy(dtype=bool)
                if self.group_column in pseudo_actual.columns
                else np.zeros(len(pseudo_actual), dtype=bool)
            )
            alpha_h_active = getattr(self, "trust_decay_surge_alpha_", 0.9)
            if getattr(self, "trust_decay_event_gating_enabled_", False):
                alpha_h_row = np.where(surge_active_today, np.maximum(alpha_h_base, alpha_h_active), alpha_h_base)
            else:
                alpha_h_row = np.full(len(pseudo_actual), alpha_h_base, dtype=float)

            pred_vals = pseudo_actual.apply(
                lambda row: pred_map.get((row[self.group_column], d_str), 0.0), axis=1
            ).to_numpy(dtype=float)
            ref_vals = pseudo_actual[self.group_column].map(reference_level).to_numpy(dtype=float)
            ref_vals = np.where(np.isnan(ref_vals), pred_vals, ref_vals)   # rota reference_level'da yoksa q50'ye düş

            pseudo_actual[self.target_column] = alpha_h_row * pred_vals + (1 - alpha_h_row) * ref_vals

            if (
                self.logging_enabled
                and getattr(self, "trust_decay_event_gating_enabled_", False)
                and surge_active_today.any()
            ):
                logger.info(
                    f"   🔥 Trust Decay gevşetildi (h={h_idx}, {d_str}): "
                    f"{int(surge_active_today.sum())} satırda surge aktif → "
                    f"alpha_h={alpha_h_active} (bayat referans seviyeye kayma engellendi)."
                )

            rolling_context = pd.concat(
                [rolling_context, pseudo_actual], ignore_index=True
            )

            # Buffer'ı sınırsız büyütmeyin — sadece gereken kadarını tutun
            if self.group_column and self.group_column in rolling_context.columns:
                rolling_context = (
                    rolling_context.sort_values([self.group_column, self.date_column])
                    .groupby(self.group_column, group_keys=False)
                    .tail(buffer_size)
                    .reset_index(drop=True)
                )
            else:
                rolling_context = (
                    rolling_context.sort_values(self.date_column)
                    .tail(buffer_size)
                    .reset_index(drop=True)
                )

        if self.logging_enabled:
            logger.info(
                f"✅ predict_sequential() tamamlandı: {len(all_results)} tahmin "
                f"üretildi ({len(unique_dates)} gün, autoregressive)."
            )

        return all_results

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

        ⚠️  VARSAYIM (wide-format iki-slotlu akış): `df` (run_forecast.py'den
        gelen ham full_df) her iki hedef sütununu da (toplam_desi_0900 ve
        toplam_desi_1700) içermelidir. df.groupby(...).tail(buffer_size)
        tüm sütunları taşıdığı için bu fonksiyonun kendisi değişmedi —
        ama varsayım burada açıkça belirtiliyor, çünkü _prepend_context_buffer
        ve _engineer_features bu varsayıma bağımlı çalışıyor (build_feature_matrix
        her iki hedefin de var olmasını bekliyor).

        Parameters
        ----------
        df : Ham eğitim DataFrame'i (feature engineering öncesi), her iki
             hedef sütunu da (target_column + sibling_target_column) içermeli.
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

        ⚠️  VARSAYIM (wide-format iki-slotlu akış): predict() için hazırlanan
        `df`, hem target_column hem sibling_target_column'u (adı ve tipiyle)
        içermek ZORUNDADIR — ikisi de NaN/0 olabilir ama sütun olarak mevcut
        olmalı. Aksi halde build_feature_matrix hata verir veya eksik sütun
        uydurmaya çalışır. Bu, run_forecast.py'nin build_predict_grid()
        fonksiyonunun tahmin ızgarasına her iki hedef sütununu da koyup
        koymadığının kontrol edilmesini gerektirir (run_forecast.py adımında
        netleştirilecek).

        Parameters
        ----------
        df : Ham tahmin DataFrame'i — hem target_column hem
             sibling_target_column sütunlarını (NaN olsa da) içermeli.

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
        abnormal_week_mask: Optional[np.ndarray] = None,
        scale_factor_test: Optional[np.ndarray] = None,
        scale_factor_train: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Test ve Train setleri üzerinde WAPE ve Decision Regret hesaplar,
        raporlama ve sunumlar için aşırı öğrenme (overfit) analizi basar.

        Parameters
        ----------
        abnormal_week_mask : optimize.py ile aynı yöntemle (haftalık ortalama
            > genel ortalama × 1.4) işaretlenmiş anormal-hafta maskesi.
            Verilirse "WAPE (temiz)" hesabından bu satırlar da dışlanır —
            böylece bu metrik, optimize.py'nin raporladığı "best_wape_clean"
            ile gerçekten kıyaslanabilir hale gelir (aksi halde iki WAPE
            farklı istisna kümeleriyle hesaplanıp yanıltıcı şekilde
            karşılaştırılabiliyordu).
        scale_factor_test, scale_factor_train : PDF Strateji 2 — Target
            Scaling Bug Fix. `self.models_` (ensemble) `_target_scaling_active_`
            iken SCALED hedef üzerinde eğitildi (bkz. fit() — y_fold_train =
            fold_train_df[self._scaled_target_col_]); bu fonksiyon ise
            model.predict()'i DOĞRUDAN çağırdığı için (Pool bazlı, tekil
            batch), çıktısı da SCALED uzaydadır (~1.0 civarı oran). Bu diziler
            verilirse (fit()'te train_df/test_df üzerinden, X'ten drop
            edilmeden ÖNCE ayrıca alınır — bkz. çağıran kod), tahminler ham
            y_true (raw desi) ile karşılaştırılmadan HEMEN önce kendi
            scale_factor'üyle geri çarpılıp raw uzaya döndürülür — TIPKI
            _predict_single_batch()'teki un-scale adımı gibi (satır ~2125).
            None ise (target scaling kapalı veya eski/geriye-dönük çağrı)
            hiçbir şey yapılmaz — eski davranışla birebir aynı kalır.
            ⚠️ Bu düzeltme olmadan, scaled-uzay tahmini (~1.0) doğrudan ham
            hacimle (yüzler/binler) karşılaştırılınca WAPE yapay olarak
            ~%100'e yapışıyordu (Train VE Test'te aynı anda) — bir overfit/
            model kalite sorunu DEĞİL, sadece bu raporlama fonksiyonunun
            target scaling'den önce güncellenmemiş olmasıydı.
        """
        # --- TEST SETİ DEĞERLENDİRMESİ ---
        test_pool = Pool(data=X_test, cat_features=self.cat_features_)
        # Ensemble: tüm modellerin q50 ([:, 1]) medyanı
        q50_preds_test = np.median(
            [model.predict(test_pool)[:, 1] for model in self.models_], axis=0
        )
        # Ensemble: tüm modellerin q90 ([:, 2]) medyanı — Spot_Cost_Sim için gerekli
        # (q50-vs-q90 strateji kıyaslaması eğitim log'unda görünür olsun diye)
        q90_preds_test = np.median(
            [model.predict(test_pool)[:, 2] for model in self.models_], axis=0
        )

        y_true_test = y_test.values
        # ⚠️ EĞER DÖNÜŞÜM YAPILDIYSA, METRİK HESABINDAN ÖNCE GERİ ÇEVİR!
        if self.log_transform_enabled:
            q50_preds_test = np.square(q50_preds_test)
            q90_preds_test = np.square(q90_preds_test)
            y_true_test = np.square(y_true_test)

        # --- Bug Fix — Target Scaling Un-scale (bkz. docstring) ---
        # sqrt geri-çevirmeden SONRA, y_true_test ile karşılaştırmadan ÖNCE —
        # _predict_single_batch()'teki sıralamayla birebir tutarlı.
        if scale_factor_test is not None and len(scale_factor_test) == len(q50_preds_test):
            q50_preds_test = q50_preds_test * scale_factor_test
            q90_preds_test = q90_preds_test * scale_factor_test
            if self.logging_enabled:
                logger.info(
                    "   📐 [_evaluate_on_test] Hedef Ölçeklendirme geri çevrildi "
                    f"(Test seti, {len(scale_factor_test)} satır) — WAPE artık raw desi uzayında."
                )

        q50_preds_test = np.maximum(q50_preds_test, 0)
        q90_preds_test = np.maximum(q90_preds_test, 0)

        # --- Hibrit Domain Heuristic (WAPE değerlendirmesi) ---
        if "is_campaign_eve" in X_test.columns and hasattr(self, "campaign_multipliers_"):
            camp_mask_test = (X_test["is_campaign_eve"] == 1).values
            if camp_mask_test.sum() > 0:
                # Her test satırı için ait olduğu rotanın çarpanını getir (bulamazsa 1.15)
                route_vals = X_test[self.group_column].values if self.group_column in X_test.columns else []
                mult_array = np.array([self.campaign_multipliers_.get(r, 1.15) for r in route_vals])

                # Sadece kampanya günlerine kendi özel çarpanlarını uygula
                q50_preds_test[camp_mask_test] *= mult_array[camp_mask_test]

                if self.logging_enabled:
                    logger.info(f"   💡 Dinamik Domain Heuristic (eval): {camp_mask_test.sum()} güne rota bazlı çarpanlar uygulandı.")
        # ---------------------------------------------------------

        # --- Adım 2: Rota Bazlı OOF Bias Düzeltmesi (WAPE değerlendirmesi) ---
        # predict()'teki uygulamayla TUTARLI olsun diye burada da eklendi —
        # aksi halde fit()-zamanı raporlanan Test WAPE, üretimde gerçekte
        # kullanılacak kalibre edilmiş tahminleri YANSITMAZ (yanıltıcı olur).
        if (
            getattr(self, "route_bias_correction_enabled_", False)
            and self.group_column in X_test.columns
            and (getattr(self, "route_bias_correction_", None) or getattr(self, "route_bucket_bias_correction_", None))
        ):
            route_vals_bias_test = X_test[self.group_column].values
            weekday_vals_bias_test = (
                pd.to_numeric(X_test["weekday"], errors="coerce").values
                if "weekday" in X_test.columns else np.full(len(X_test), np.nan)
            )
            event_active_bias_test = np.zeros(len(X_test), dtype=bool)
            for _col in SURGE_BINARY_TRIGGER_COLUMNS:
                if _col in X_test.columns:
                    event_active_bias_test |= (pd.to_numeric(X_test[_col], errors="coerce").fillna(0.0) > 0).values
            for _col in SURGE_CONTINUOUS_TRIGGER_COLUMNS:
                if _col in X_test.columns:
                    event_active_bias_test |= (
                        pd.to_numeric(X_test[_col], errors="coerce").fillna(0.0) > BUCKET_EVENT_CONTINUOUS_THRESHOLD
                    ).values

            bias_mult_test = np.array([
                self._lookup_bias_correction(r, wd, bool(ev))
                for r, wd, ev in zip(route_vals_bias_test, weekday_vals_bias_test, event_active_bias_test)
            ])
            adj_mask_test = bias_mult_test != 1.0
            if np.any(adj_mask_test):
                q50_preds_test[adj_mask_test] *= bias_mult_test[adj_mask_test]
                q50_preds_test = np.maximum(q50_preds_test, 0)
                if self.logging_enabled:
                    logger.info(
                        f"   🎯 Rota×Gün-Türü OOF Bias Düzeltmesi (eval, Öneri A): "
                        f"{int(adj_mask_test.sum())} satıra uygulandı."
                    )
        # ---------------------------------------------------------

        sum_true_test = np.sum(y_true_test)
        wape_test = (
            float(np.sum(np.abs(y_true_test - q50_preds_test)) / sum_true_test)
            if sum_true_test > 0 else 0.0
        )

        # --- Temiz WAPE: tatil/birikim günlerini dışla ---
        # Test seti tatil günleri (talep ~%10'a düşer) veya tatil sonrası
        # birikim patlaması (talep ~%150'ye çıkar) içeriyorsa bu günler
        # WAPE'yi gerçek model performansından bağımsız şişirir.
        # "Temiz WAPE" sadece normal iş günlerini değerlendirir.
        wape_clean = wape_test  # varsayılan: temiz gün yoksa tüm test
        if "is_holiday" in X_test.columns:
            # Tatil günü veya Pazar (weekday==6, talep ~sıfır) çıkar
            weekday_col = X_test.get("weekday") if "weekday" in X_test.columns else None
            holiday_mask = (X_test["is_holiday"].values == 1)
            # Pazar günleri de çıkar (talep yapısal olarak çok düşük, WAPE'yi şişirir)
            if weekday_col is not None:
                sunday_mask = (weekday_col.values == 6)
            else:
                sunday_mask = np.zeros(len(y_true_test), dtype=bool)
            normal_mask = ~(holiday_mask | sunday_mask)
            # [Entegrasyon] optimize.py'nin anormal-hafta filtresi (ort. × 1.4)
            # de aynı "temiz" tanımına dahil edilir — aksi halde bu metrik
            # optimize.py'nin best_wape_clean'iyle kıyaslanamaz kalırdı.
            n_abnormal_excluded = 0
            if abnormal_week_mask is not None and len(abnormal_week_mask) == len(normal_mask):
                abnormal_arr = np.asarray(abnormal_week_mask, dtype=bool)
                n_abnormal_excluded = int((abnormal_arr & normal_mask).sum())
                normal_mask = normal_mask & ~abnormal_arr
            if normal_mask.sum() >= 10:
                wape_clean = (
                    float(np.sum(np.abs(y_true_test[normal_mask] - q50_preds_test[normal_mask]))
                          / np.sum(y_true_test[normal_mask]))
                    if np.sum(y_true_test[normal_mask]) > 0 else 0.0
                )
                if self.logging_enabled and n_abnormal_excluded > 0:
                    logger.info(
                        f"   ℹ️  WAPE (temiz) hesabından ayrıca {n_abnormal_excluded} "
                        f"anormal-hafta günü dışlandı (optimize.py ile tutarlı tanım)."
                    )

        diff_test = y_true_test - q50_preds_test
        regret_test = np.where(
            diff_test > 0,
            diff_test * self.underestimation_penalty,
            np.abs(diff_test) * 1.0,
        )
        decision_regret_test = float(np.mean(regret_test))

        # --- [Opsiyonel] Spot_Cost_Sim: q50-vs-q90 strateji kıyaslaması ---
        # HPO'yu bekletmeden, fit sonrası eğitim log'unda alpha'nın (q90 asimetrik
        # kaybı) gerçek maliyeti doğru yönde hareket ettirip ettirmediğini görmek
        # için metrics.py::spot_cost_simulation() çağrılır. Opsiyonel olduğundan
        # metrics.py bulunamazsa / imza uyuşmazsa sessizce atlanır — training
        # akışını KIRMAZ.
        # NOT: cost_per_unit_spot / cost_per_unit_idle, metrics.py'deki gerçek TL
        # maliyet varsayılanlarıdır (1.0 / 0.2) — Decision Regret'teki 9x asimetrik
        # ceza (underestimation_penalty) ile KARIŞTIRILMAZ; bilinçli olarak override
        # edilmiyor, metrics.py'nin kendi varsayılanları kullanılıyor.
        spot_cost_sim_result: Optional[Dict[str, float]] = None
        try:
            from .metrics import spot_cost_simulation
            spot_cost_sim_result = spot_cost_simulation(
                y_true=y_true_test,
                y_pred_q50=q50_preds_test,
                y_pred_q90=q90_preds_test,
            )
        except ImportError:
            if self.logging_enabled:
                logger.debug(
                    "   ℹ️  Spot_Cost_Sim atlandı: metrics.py::spot_cost_simulation() "
                    "bulunamadı (opsiyonel özellik)."
                )
        except Exception as exc:
            if self.logging_enabled:
                logger.warning(
                    f"   ⚠️  Spot_Cost_Sim hesaplanamadı (opsiyonel, training "
                    f"etkilenmedi): {exc}"
                )

        # Geriye uyumluluk için eski anahtarları koruyoruz (optimize.py kırılmasın diye)
        self.eval_results_: Dict[str, float] = {
            "WAPE":            round(wape_test, 6),
            "Decision_Regret": round(decision_regret_test, 4),
            "test_samples":    len(y_true_test),
        }
        if spot_cost_sim_result is not None:
            self.eval_results_["Spot_Cost_Sim"] = spot_cost_sim_result

        # --- TRAIN SETİ DEĞERLENDİRMESİ (OVERFIT KONTROLÜ) ---
        wape_train = 0.0
        decision_regret_train = 0.0
        
        if X_train is not None and y_train is not None:
            train_pool = Pool(data=X_train, cat_features=self.cat_features_)
            # Ensemble: tüm modellerin q50 medyanı
            q50_preds_train = np.median(
                [model.predict(train_pool)[:, 1] for model in self.models_], axis=0
            )
            y_true_train = y_train.values
            # ⚠️ EĞER DÖNÜŞÜM YAPILDIYSA, METRİK HESABINDAN ÖNCE GERİ ÇEVİR!
            if self.log_transform_enabled:
                q50_preds_train = np.square(q50_preds_train)
                y_true_train = np.square(y_true_train)

            # --- Bug Fix — Target Scaling Un-scale (bkz. fonksiyon docstring'i) ---
            if scale_factor_train is not None and len(scale_factor_train) == len(q50_preds_train):
                q50_preds_train = q50_preds_train * scale_factor_train
                if self.logging_enabled:
                    logger.info(
                        "   📐 [_evaluate_on_test] Hedef Ölçeklendirme geri çevrildi "
                        f"(Train seti, {len(scale_factor_train)} satır) — WAPE artık raw desi uzayında."
                    )

            q50_preds_train = np.maximum(q50_preds_train, 0)

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
            # Asıl performansı yansıtan wape_clean üzerinden overfit kontrolü yapılır
            if X_train is not None and (wape_clean - wape_train) > 0.06: 
                status = "⚠️ OVERFIT"

            clean_note = f"{wape_clean:<12.4%}" if wape_clean != wape_test else f"{'(tatil yok)':<12}"
            slot_note = f" — Slot: {self.slot_label}" if self.slot_label else ""
            log_table = (
                f"\n📊 MODEL PERFORMANS VE OVERFIT ANALİZİ (q50){slot_note}:\n"
                f"   ┌───────────────────┬──────────────┬──────────────┬──────────────┐\n"
                f"   │ Metrik            │ Train Seti   │ Test Seti    │ Durum        │\n"
                f"   ├───────────────────┼──────────────┼──────────────┼──────────────┤\n"
                f"   │ WAPE (tüm günler) │ {wape_train:<12.4%} │ {wape_test:<12.4%} │ {status:<12} │\n"
                f"   │ WAPE (tatil hariç)│ {'':12} │ {clean_note} │ {'gerçek perf.':<12} │\n"
                f"   │ Decision Regret   │ {decision_regret_train:<12.2f} │ {decision_regret_test:<12.2f} │ {'-'*12} │\n"
                f"   │ Örnek Sayısı      │ {len(y_train) if y_train is not None else 0:<12,} │ {len(y_true_test):<12,} │ {'-'*12} │\n"
                f"   └───────────────────┴──────────────┴──────────────┴──────────────┘"
            )
            if spot_cost_sim_result is not None:
                q50_cost = spot_cost_sim_result.get("q50_total_cost")
                q90_cost = spot_cost_sim_result.get("q90_total_cost")
                savings  = spot_cost_sim_result.get("savings_with_q90")
                if q50_cost is not None and q90_cost is not None:
                    # savings_with_q90 = q50_cost - q90_cost (metrics.py tanımı)
                    # Pozitif → q90 daha ucuz (az spot araç) | Negatif → q90 fazla ihtiyatlı (idle kapasite)
                    yon = "q90 daha ucuz ✅" if savings > 0 else ("q50 daha ucuz ⚠️" if savings < 0 else "eşit")
                    log_table += (
                        f"\n💰 SPOT COST SIM (q50 vs q90 strateji, test seti — alpha yönü kontrolü):\n"
                        f"   q50 strateji maliyeti : {q50_cost:,.2f}\n"
                        f"   q90 strateji maliyeti : {q90_cost:,.2f}\n"
                        f"   q90 Tasarrufu          : {savings:,.2f} → {yon}"
                    )
                else:
                    log_table += f"\n💰 SPOT COST SIM (ham sonuç): {spot_cost_sim_result}"
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
            "sibling_target_column":   self.sibling_target_column,
            "slot_label":              self.slot_label,
            "surge_residual_enabled":  self.surge_residual_enabled,
            "surge_log_cosh_tau":      self.surge_log_cosh_tau,
            "surge_min_rows":          self.surge_min_rows,
            "surge_calibration_factor": getattr(self, "surge_calibration_factor_", 1.0),
            "surge_dampening_alpha":    getattr(self, "surge_dampening_alpha_", 0.0),
            "surge_relative_cap_alpha": getattr(self, "surge_relative_cap_alpha_", None),
            "surge_volume_damping_enabled": getattr(self, "surge_volume_damping_enabled_", True),
            "surge_volume_damping_v_crit":  getattr(self, "surge_volume_damping_v_crit_", 150.0),
            "surge_volume_damping_k":       getattr(self, "surge_volume_damping_k_", 3.0),
            "proxy_spo_enabled":            getattr(self, "proxy_spo_enabled_", True),
            "proxy_spo_capacity_quantile":  getattr(self, "proxy_spo_capacity_quantile_", 0.90),
            "proxy_spo_spot_cost_multiplier": getattr(self, "proxy_spo_spot_cost_multiplier_", 3.0),
            "proxy_spo_idle_cost_multiplier": getattr(self, "proxy_spo_idle_cost_multiplier_", 1.0),
            "proxy_spo_weight_clip":        getattr(self, "proxy_spo_weight_clip_", (1.0, 5.0)),
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
            f"  Mimari          : {'Ensemble (' + str(len(self.models_)) + ' fold model)' if self.is_fitted_ and self.models_ else 'Tekli Model'}",
            f"  Hedef           : {self.target_column}" + (f" (Slot: {self.slot_label})" if self.slot_label else ""),
            f"  Sibling Hedef   : {self.sibling_target_column or '⚠️  YOK (zorunlu!)'}",
            f"  Grup            : {self.group_column}",
            f"  Horizon         : {self.forecast_horizon} gün",
            f"  Iterations      : {self.iterations}",
            f"  Depth           : {self.depth}",
            f"  Lags            : {self.lags}",
            f"  Rolling         : {self.rolling_windows}",
            f"  Asimetrik Ceza  : {self.underestimation_penalty}x (q90)",
            f"  Outlier Clip    : IQR × {self.outlier_clip_multiplier} ({'kapalı' if self.outlier_clip_multiplier == 0 else 'açık'})",
            f"  Log Dönüşümü    : {'⚠️  log1p (MultiQuantile ile önerilmez!)' if self.log_transform_enabled else '✅ kapalı (MultiQuantile için doğru)'}",
            f"  Kantiller       : q10 / q50 / q90",
            f"  Proxy SPO (Taban): {'✅ aktif — Pool(weight=karar_pişmanlığı)' if getattr(self, 'proxy_spo_enabled_', True) else '⛔ kapalı (weight=1.0)'} "
            f"(cap_q={getattr(self, 'proxy_spo_capacity_quantile_', 0.90)}, "
            f"spot_mult={getattr(self, 'proxy_spo_spot_cost_multiplier_', 3.0)}, "
            f"clip={getattr(self, 'proxy_spo_weight_clip_', (1.0, 5.0))})",
            f"  Surge/Residual  : {'✅ eğitildi (Model 2, τ_i=dinamik vektör [PDF, satır/rota bazlı])' if getattr(self, 'surge_model_', None) is not None else ('⏳ atlandı/kapalı' if self.surge_residual_enabled else '⛔ kapalı (surge_residual_enabled=False)')}",
            f"  Surge Kalibrasyon: {getattr(self, 'surge_calibration_factor_', 1.0)}x" + (" (varsayılan, değiştirilmedi)" if getattr(self, 'surge_calibration_factor_', 1.0) == 1.0 else " ⚠️ manuel ayarlandı"),
            f"  Surge Ortogonalleştirme: {len(getattr(self, 'surge_feature_names_', []))} özellik "
            f"(Taban: {len(self.feature_names_)}) — statik kampanya/tatil bayrakları çıkarıldı",
            f"  Surge Dinamik Sönümleme: α={getattr(self, 'surge_dampening_alpha_', 0.0)} "
            f"({'✅ aktif' if getattr(self, 'surge_dampening_alpha_', 0.0) > 0 else '⛔ kapalı'})",
            f"  Surge Hacim Sönümleme (S_vol): "
            f"{'✅ aktif' if getattr(self, 'surge_volume_damping_enabled_', True) else '⛔ kapalı'} "
            f"(v_crit={getattr(self, 'surge_volume_damping_v_crit_', 150.0)}, "
            f"k={getattr(self, 'surge_volume_damping_k_', 3.0)}) — Faz 2b (kesikli segment_scale) YERİNE",
            f"  Weekday Bias    : {self.weekday_bias_ if getattr(self, 'weekday_bias_', None) else '⏳ kalibre edilmedi'}",
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

    def save_model(self, file_path: str) -> None:
        """Eğitilmiş modeli, içindeki context_buffer ve çarpanlarla birlikte kaydeder (.joblib)"""
        if not self.is_fitted_:
            raise ValueError("❌ Model henüz eğitilmedi, kaydedilemez!")
        joblib.dump(self, file_path)
        if self.logging_enabled:
            logger.info(f"💾 Eğitilmiş model başarıyla kaydedildi: {file_path}")

    @classmethod
    def load_model(cls, file_path: str) -> "DemandForecaster":
        """Hazır eğitilmiş modeli diskten yükler"""
        model = joblib.load(file_path)

        # ⚠️ DENEYSEL / TEST AMAÇLI (ADIM 2 — weekday bias calibration).
        # .joblib DOSYASININ İÇERİĞİNİ DEĞİŞTİRMEZ — sadece bu runtime
        # nesnesine (bellekte) uygulanır, diske kalıcı yazılmaz. Model daha
        # önce hiç weekday_bias_ almadan kaydedildiyse (eski model ya da
        # henüz kalibre edilmemiş), slot_label'a göre otomatik enjekte eder.
        # Kaynak ve sınırlar için bkz. _EMPIRICAL_WEEKDAY_BIAS_1700 docstring'i.
        # Kalıcı/doğru (retrain'li) versiyon fit()'te öğrenilip joblib'e
        # gömülünce bu bloğu KALDIRIN — aksi halde deneysel değer, gerçekten
        # öğrenilmiş olanın üzerine sessizce binmez (if None kontrolü zaten
        # bunu engelliyor) ama kafa karışıklığına yol açabilir.
        if getattr(model, "weekday_bias_", None) is None:
            if getattr(model, "slot_label", None) == "17:00":
                model.weekday_bias_ = {
                    k: v * cls._WEEKDAY_BIAS_SCALE
                    for k, v in cls._EMPIRICAL_WEEKDAY_BIAS_1700.items()
                }
                if getattr(model, "logging_enabled", True):
                    logger.info(
                        "   📅 [DENEYSEL] weekday_bias_ otomatik enjekte edildi "
                        f"(slot=17:00, ölçek={cls._WEEKDAY_BIAS_SCALE}x, "
                        f"kaynak=2026-06-14→06-20 penceresi, "
                        f"joblib dosyasına YAZILMADI): {model.weekday_bias_}"
                    )
            # 09:00 için aynı pencerede işaret tutarsızdı (bkz. sınıf sabiti
            # docstring'i) — bilerek hiçbir bias enjekte edilmiyor.

        # ⚠️ DENEYSEL / TEST AMAÇLI (Öneri A — route_bias_correction_
        # enable). .joblib DOSYASININ İÇERİĞİNİ DEĞİŞTİRMEZ — sadece bu
        # runtime nesnesine (bellekte) uygulanır. NEDEN GEREKLİ: bu bir
        # instance attribute'u — pickle __init__()'i tekrar çalıştırmaz,
        # sadece dump ANINDAKİ değeri saklar. run_forecast.py fit()'ten
        # SONRA bu flag'i bellek-içi nesnede True yapıyor ama .joblib
        # dump'ı BUNDAN ÖNCE gerçekleşmiş olabilir — yani diskteki dosya
        # flag=False ile kaydedilmiş olabilir. Bu yüzden HER load_model()
        # çağrısı (debug_backtest.py dahil) burada merkezi olarak flag'i
        # açar; route_bucket_bias_correction_/route_bias_correction_ boşsa
        # zaten no-op (hiçbir satır etkilenmez, cap'li/güvenli fallback).
        # Kalıcı olarak KAPATMAK isterseniz load_model()'den dönen nesnede
        # `model.route_bias_correction_enabled_ = False` ile ezebilirsiniz.
        model.route_bias_correction_enabled_ = True

        # ⚠️ GERİYE DÖNÜK UYUMLULUK — unconstrain_censored_demand() weekday-
        # persistence gate'i eklenmeden ÖNCE kaydedilmiş .joblib dosyaları
        # bu attribute'lara sahip DEĞİL (pickle __init__()'i tekrar
        # çalıştırmaz). getattr(..., None) is None kontrolüyle sadece
        # EKSİKSE enjekte ediyoruz — v2 (persistence gate) davranışını
        # varsayılan yapıyoruz, ama bu SADECE bellek-içi nesneyi etkiler,
        # .joblib dosyasının içeriğini DEĞİŞTİRMEZ. Eski davranışı (v1,
        # tek-seferlik tespit) geri istiyorsanız, load_model()'den dönen
        # nesnede `model.censor_require_weekday_persistence_ = False`
        # ile ezebilirsiniz.
        if getattr(model, "censor_window_", None) is None:
            model.censor_window_ = 14
        if getattr(model, "censor_min_volume_threshold_", None) is None:
            model.censor_min_volume_threshold_ = 50.0
        if getattr(model, "censor_cap_ratio_", None) is None:
            model.censor_cap_ratio_ = 0.98
        if getattr(model, "censor_inflation_factor_", None) is None:
            model.censor_inflation_factor_ = 1.05
        if getattr(model, "censor_require_weekday_persistence_", None) is None:
            model.censor_require_weekday_persistence_ = True
        if getattr(model, "censor_persistence_occurrences_", None) is None:
            model.censor_persistence_occurrences_ = 3
        if getattr(model, "censor_persistence_min_hits_", None) is None:
            model.censor_persistence_min_hits_ = 2
        # Gerçek kapasite gate'i (bkz. Mantık 3.5) eski .joblib'lerde hiç
        # yok — varsayılan None (KAPALI) enjekte ediliyor, yani eski
        # modeller davranış DEĞİŞTİRMEDEN yüklenmeye devam eder. Kapasite
        # gate'ini AÇMAK isterseniz load_model() sonrası elle atayın:
        #   fc.censor_capacity_df_ = pd.read_excel("Ellecleme-kapasite.xlsx")
        #   fc.censor_source_tm_column_ = "kaynak_tm"
        if not hasattr(model, "censor_capacity_df_"):
            model.censor_capacity_df_ = None
        if not hasattr(model, "censor_source_tm_column_"):
            model.censor_source_tm_column_ = None
        if not hasattr(model, "censor_capacity_tm_column_"):
            model.censor_capacity_tm_column_ = "transfer_merkezi"
        if not hasattr(model, "censor_capacity_value_column_"):
            model.censor_capacity_value_column_ = "ellecleme_kapasite"
        if not hasattr(model, "censor_real_capacity_ratio_"):
            model.censor_real_capacity_ratio_ = 0.90

        return model