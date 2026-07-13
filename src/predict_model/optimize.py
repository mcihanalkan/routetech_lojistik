"""
optimize.py — Hiperparametre Optimizasyon Scripti
===================================================
Ayrı çalıştırılır (run_forecast.py ile değil).
Gerçek veri üzerinde Optuna çalıştırır, en iyi parametreleri
hyperparams_map.json'a yazar.

Kullanım:
    python optimize.py                            # varsayılan: 09:00 VE 17:00 SIRAYLA (tek çalıştırma)
    python optimize.py --slot 0900                # sadece 09:00 modeli
    python optimize.py --slot 1700                # sadece 17:00 modeli
    python optimize.py --data baska_veri.xlsx     # farklı veriyle (ikisini de tune eder)
    python optimize.py --trials 50 --timeout 1200
    python optimize.py --trials 50 --timeout 0    # timeout yok, sadece trial sayısı sınırı
    python optimize.py --gamma 0.6                # spot/atıl maliyet oranı: (C_spot/C_atil) - 1
    python optimize.py --gamma 0                  # asimetrik ceza kapalı (eski davranış)
    python optimize.py --alpha-sweep               # her iki slot için de alpha knee taraması

--slot verilmezse (ya da --slot both) tek komutla 09:00 ve 17:00 SIRAYLA
tune edilir (bkz. v12 notu aşağıda) — toplam süre tek slota göre ~2 katı
olur. Tek bir slotu hızlıca tune etmek istersen --slot 0900 / --slot 1700.

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

--- v12 Değişiklikleri (Slot-Farkındalı Entegrasyon — run_forecast.py / forecasters.py ile tam senkron) ---
optimize.py artık run_forecast.py'nin GERÇEKTEN çalıştığı wide/slot formatına
(09:00 / 17:00, tek satır = rota×tarih, iki hedef sütun) ve forecasters.py'nin
gerçek eğitim mantığına birebir bağlı:
  1. load_data() KALDIRILDI — run_forecast.py::load_dataset() import edilip
     kullanılıyor (tek kaynak, kod tekrarı ve kopma riski yok). Varsayılan
     --data artık teknofest26_gelismis.xlsx.
  2. build_feature_matrix çağrıları artık target_columns=[target_column,
     sibling_target_column] (liste) kullanıyor — forecasters.py::
     _engineer_features ile birebir aynı.
  3. optimize() ve alpha_sweep() artık --slot {0900,1700} argümanıyla
     çalışıyor; TARGET_COL sabit kaldırıldı, her yerde
     target_column/sibling_target_column parametre olarak taşınıyor.
     n_rows artık (full_df[target_column] > 0).sum() ile SADECE o slotun
     gerçek kayıtları üzerinden hesaplanıyor.
  4. drop_cols naif [TARGET_COL, DATE_COL] listesi yerine forecasters.py::
     _get_drop_columns ile BİREBİR AYNI kural uygulanıyor (bkz.
     _get_drop_columns() fonksiyonu aşağıda): 09:00 modeli için sibling
     (toplam_desi_1700) ve cross_lag_0900_same_day KESİNLİKLE drop edilir;
     17:00 modeli için sibling (toplam_desi_0900) BİLEREK tutulur (leakage
     değil, meşru feature).
  5. FOLD_DATES artık forecasters.py::fit() ile BİREBİR AYNI:
     2026-05-31 → 2026-06-27 (önceki 2026-04-14 → 2026-05-10 pencereleri
     eski/kopmuş veriye aitti — bu dosyanın kendi docstring uyarısının
     ihlal edildiği tam nokta buydu, artık düzeltildi).
  6. [v13'te ÇÖZÜLDÜ — bkz. v13 notu aşağıda] ALPHA_MIN/MAX bandı artık
     slot-bazlı (_SLOT_ALPHA_BANDS / _alpha_band_for()).
  7. --slot argümanı: varsayılan "both" ile TEK komutta 09:00 (n≈24.906 →
     "small" bucket) ve 17:00 (n≈41.118 → "medium" bucket) SIRAYLA tune
     edilir. İstenirse --slot 0900 / --slot 1700 ile tek tek de çalıştırılabilir.
       python optimize.py                 # ikisi de, tek komut
       python optimize.py --slot 0900     # sadece 09:00
       python optimize.py --slot 1700     # sadece 17:00
     _bucket_name() otomatik ayrım yapar; doğru n_rows hesaplandığı sürece
     (madde 3) iki farklı bucket kendiliğinden JSON'da oluşur.

--- v13 Değişiklikleri (rel_width Bug Düzeltmesi + Slot-Bazlı Alpha Bandı) ---
  1. BUG DÜZELTMESİ — _correct_and_score() içindeki rel_width formülü
     eskiden (q90-q10)/max(preds,1.0) idi; preds = MultiQuantile'ın
     alpha-kuantili, yani ALPHA İLE BİRLİKTE KAYAN bir "mid" tahmini
     (alpha kendisi Optuna'nın arama uzayında). alpha büyüdükçe preds de
     sistematik olarak büyüdüğü için, bant GENİŞLİĞİ hiç değişmese bile
     rel_width sadece alpha seçimi yüzünden küçülüyordu — Optuna gerçek
     bandı daraltmadan, salt yüksek alpha seçerek bu cezadan kaçabiliyordu.
     Bu, alpha seçimini objective üzerinden güvenilmez kılan asıl sebepti.
     Düzeltme: rel_width artık Σ(q90c-q10c)/Σy_true ile normalize ediliyor
     — WAPE'nin paydasıyla aynı, alpha'dan tamamen bağımsız SABİT bir
     referans. Aynı düzeltme alpha_sweep() ve optimize()'daki üç ayrı
     rel_width hesaplamasında da (kod tekrarı — _fit_fold_ensemble_and_score
     zaten _correct_and_score()'u kullanıyordu, ama alpha_sweep() ve
     optimize()'ın final raporlama bloğu formülü inline tekrarlıyordu)
     tutarlı şekilde uygulandı.
  2. Slot-bazlı ALPHA_MIN/ALPHA_MAX — v12'nin kendi uyarısı (madde 6, eski)
     doğrulandı: yeni slot verisinde eski [0.60, 0.78] bandı 09:00 için
     çok yüksekti. `python optimize.py --alpha-sweep --slot 0900` sonucuna
     göre 09:00 için ~0.50-0.60 bandı çok daha makul (WAPE ~%46-48,
     Underpred ~%32-37, hâlâ kabul edilebilir). Artık _SLOT_ALPHA_BANDS =
     {"0900": (0.50, 0.60), "1700": (ALPHA_MIN, ALPHA_MAX)} — 17:00 kendi
     --alpha-sweep --slot 1700 kanıtı gelene kadar eski bantta kalıyor.
     optimize(), _alpha_band_for(target_column) ile doğru bandı otomatik
     seçiyor; CLI/çağıran kod değişikliği gerekmiyor.
  3. rel_width formülü değiştiği için (mean-of-ratio → sum-of-ratio) sayısal
     ölçek bir miktar kayabilir — BETA_WIDTH (0.03) yeniden kalibre
     edilmesi gerekebilir; ilk birkaç çalıştırmadan sonra gözlemleyip
     ayarlayın (bkz. BETA_WIDTH yorumu).
  ÖNEMLİ: Bu değişiklikler sonrası (ve yeni feature.py backlog/extreme-event
  düzeltmeleriyle birlikte) 09:00 bucket'ı için TAM HPO'nun (sadece
  --alpha-sweep değil) yeniden çalıştırılması gerekir:
      python optimize.py --slot 0900 --trials 50 --timeout 1800
  Bu, hyperparams_map.json'daki "small" (09:00) bucket'ını hem yeni
  feature'lara hem de düzeltilmiş rel_width/alpha bandına göre günceller.

--- v14 Değişiklikleri (Decision-Regret Sütunu — Sert Ceza) ---
alpha_sweep() artık her alpha için metrics.py::decision_regret() ile
(spot_multiplier=9.0) doğrudan operasyonel maliyet biriminde bir
"decision_regret" sütunu da hesaplıyor ve print_alpha_sweep_table() bunu
ayrı bir sütun olarak basıyor. Gerekçe: hybrid_score = WAPE + γ·underpred
+ β·rel_width + ... bir VEKİL (proxy); WAPE'yi minimize eden alpha ile
decision_regret'i minimize eden alpha genelde FARKLI noktalarda knee
yapar — bu ikisi arasındaki farkı görmeden hibrit skor minimumuna göre
alpha seçmek de, salt WAPE'ye göre seçmekle aynı kör atış hatasına düşer.
  ÖNEMLİ — bu değişikliğin ardından hâlâ YAPILMASI GEREKENLER (bu dosyanın
  kodu bunları OTOMATİK yapmaz, elle çalıştırılıp sonuç gözlenmeli):
    1. python optimize.py --alpha-sweep --slot 0900
       python optimize.py --alpha-sweep --slot 1700
       yeni decision_regret sütununa bakarak knee noktasını (saf minimum
       değil, WAPE'nin hâlâ makul olduğu, regret kazancının düzleştiği
       nokta) belirleyin.
    2. _SLOT_ALPHA_BANDS'i bulunan knee'nin etrafında dar bir bantla
       güncelleyin (örn. knee=0.65 → (0.62, 0.70)).
    3. python optimize.py --slot 0900 --trials 50 --timeout 1800
       python optimize.py --slot 1700 --trials 50 --timeout 1800
       ile hyperparams_map.json'daki "small"/"medium" bucket'larını yeni
       bant + v14 decision_regret kanıtına göre yeniden üretin — JSON'daki
       09:00 girdisi hâlâ eski (alpha=0.5058, underpred=%40) kalibrasyona
       ait olduğu sürece güncel değildir.

  [0900 SONUÇLANDI] python optimize.py --alpha-sweep --slot 0900 gerçek veriyle
  çalıştırıldı (21 alpha × 4 fold, varsayılan sweep parametreleriyle). Sonuç:
  knee=0.66 (WAPE ~%47.94, decision_regret=866.2 — bkz. _SLOT_ALPHA_BANDS
  yorumu). _SLOT_ALPHA_BANDS["0900"] = (0.62, 0.70) olarak güncellendi.
  Sıradaki adım: python optimize.py --slot 0900 --trials 50 --timeout 1800
  ile "small" bucket'ını bu yeni banda göre yeniden üretmek.
  [1700 SONUÇLANDI] python optimize.py --alpha-sweep --slot 1700 gerçek veriyle
  çalıştırıldı (21 alpha × 4 fold). Sonuç: knee=0.70 (WAPE ~%27.37,
  decision_regret=3565.7 — bkz. _SLOT_ALPHA_BANDS yorumu).
  _SLOT_ALPHA_BANDS["1700"] = (0.66, 0.74) olarak güncellendi.

  [SIRADAKİ ADIM] Her iki slot için de tam HPO'nun yeni bantlarla yeniden
  çalıştırılması gerekiyor — hyperparams_map.json'daki "small" (0900) ve
  "medium" (1700) bucket'ları hâlâ eski (0900: alpha=0.5058, underpred=%40;
  1700: alpha=0.6821) kalibrasyona ait, bu değişiklikten HENÜZ etkilenmedi:
      python optimize.py --slot 0900 --trials 50 --timeout 1800
      python optimize.py --slot 1700 --trials 50 --timeout 1800
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
# v14: decision_regret() metrics.py'den import ediliyor — alpha_sweep()
# artık sadece hibrit skoru değil, gerçek operasyonel maliyeti (spot araç
# çarpanı ile ağırlıklandırılmış regret) de raporluyor. Bkz. alpha_sweep()
# docstring'i ve print_alpha_sweep_table().
from src.metrics import decision_regret
# v12: load_dataset() artık run_forecast.py'den import ediliyor — iki dosyanın
# birbirinden kopması (bkz. FOLD_DATES kopması, v11'de düzeltildi) tam olarak
# bu kod tekrarından geliyordu. Slot sabitleri de aynı yerden alınıyor ki
# "toplam_desi_0900" / "toplam_desi_1700" string'i iki dosyada AYRI AYRI
# yazılıp birbirinden sapmasın.
from run_forecast import (
    load_dataset,
    DATE_COL,
    GROUP_COL,
    KAYNAK_COL,
    VARIS_COL,
    TARGET_COL_0900,
    TARGET_COL_1700,
)

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
# v12: TARGET_COL / DATE_COL / GROUP_COL / KAYNAK_COL / VARIS_COL artık
# module-level sabit DEĞİL — DATE_COL/GROUP_COL/KAYNAK_COL/VARIS_COL yukarıda
# run_forecast.py'den import edildi. Tek bir "desi_hacmi" hedefi de artık
# yok — wide/slot formatında iki hedef var (TARGET_COL_0900/TARGET_COL_1700,
# import edildi), hangisinin kullanılacağı --slot CLI argümanıyla seçilir ve
# optimize()/alpha_sweep() içinde target_column/sibling_target_column olarak
# taşınır (bkz. aşağıdaki _get_drop_columns() ve _SLOT_TARGETS).

# --- Slot seçimi (forecasters.py::DemandForecaster ile aynı 09:00/17:00 mantığı) ---
_SLOT_TARGETS = {
    "0900": (TARGET_COL_0900, TARGET_COL_1700),
    "1700": (TARGET_COL_1700, TARGET_COL_0900),
}
# forecasters.py::_get_drop_columns'daki cross-lag sütunu — 09:00 modeli için
# kendi hedefinin trivial kopyası olduğundan drop edilmesi gerekiyor.
_CROSS_LAG_0900_COL = "cross_lag_0900_same_day"


def get_drop_columns(target_column: str, sibling_target_column: str, available_columns) -> list:
    """
    forecasters.py::DemandForecaster._get_drop_columns ile BİREBİR AYNI kural
    (bkz. forecasters.py satır ~287-339 — doğrulandı):

      - date_column   : HER ZAMAN drop (leakage — model tarihi görmemeli).
      - target_column : HER ZAMAN drop (y olarak ayrılır).
      - sibling_target_column:
          * 09:00 modeliyse (target_column == TARGET_COL_0900): DAİMA drop
            edilir — 17:00 talebi, 09:00 tahmini anında henüz gerçekleşmemiş
            (kesin leakage).
          * 17:00 modeliyse: DROP EDİLMEZ — 17:00 tahmini anında sabahki
            (09:00) talep zaten gerçekleşmiştir, meşru bir feature'dır.
      - cross_lag_0900_same_day:
          * 09:00 modeli için: kendi hedefinin trivial kopyası → drop.
          * 17:00 modeli için: zararsız (toplam_desi_0900 zaten feature
            olarak tutuluyor) → tutulur.

    Bu olmadan (eski naif [TARGET_COL, DATE_COL] listesiyle) optimize.py,
    üretimde asla göremeyeceği bir "gelecek bilgisi" ile tune edilir —
    bulunan hiperparametreler gerçek modelde işe yaramaz, hatta yanıltıcı
    olabilir.
    """
    is_0900_model = target_column == TARGET_COL_0900
    drop_cols = [DATE_COL, target_column]

    if sibling_target_column and is_0900_model:
        drop_cols.append(sibling_target_column)

    if is_0900_model:
        drop_cols.append(_CROSS_LAG_0900_COL)

    available = set(available_columns)
    return [c for c in drop_cols if c in available]

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
# v12 UYARI: Bu bant [0.60, 0.78] ESKİ tek-serili Desi_talep.xlsx verisinde
# (skewness ≈1.3) kalibre edildi. Yeni teknofest26_gelismis.xlsx/slot verisi
# ÇOK daha çarpık: 09:00 hedefi için skewness ≈+4.71, 17:00 için ≈+3.80.
# Bandın hâlâ doğru olduğunu VARSAYMAK RİSKLİ — Adım 1-5 (bu dosyadaki v12
# değişiklikleri) tamamlandıktan sonra HER SLOT için ayrı ayrı çalıştırın:
#     python optimize.py --alpha-sweep --slot 0900
#     python optimize.py --alpha-sweep --slot 1700
# ve knee noktasını yeniden bulup gerekirse ALPHA_MIN/ALPHA_MAX'ı (slot
# başına farklı olabilir — bu durumda _SLOT_TARGETS'e benzer bir
# {"0900": (min,max), "1700": (min,max)} sözlüğüne geçirin) güncelleyin.
# v13 UYARI ÇÖZÜLDÜ: v12'de bu bant [0.60, 0.78] ESKİ tek-serili Desi_talep.xlsx
# verisinde (skewness ≈1.3) kalibre edilmiş, yeni slot verisinde (09:00
# skewness ≈+4.71, 17:00 ≈+3.80) doğrulanmamıştı. --alpha-sweep --slot 0900
# çalıştırıldı ve sonuç NET: 09:00 için knee noktası eski bandın (0.60-0.78)
# İÇİNDE değil, daha DÜŞÜK bir aralıkta — ~0.50-0.60 bandında WAPE ~%46-48,
# Underpred ~%32-37 ile en dengeli takas burada (0.60'ın ötesinde WAPE hızla
# kötüleşiyor, Underpred kazancı ise düzleşiyor). 17:00 için henüz yeni bir
# --alpha-sweep --slot 1700 kanıtı YOK — o slot şimdilik eski [0.60, 0.78]
# bandında kalıyor (yeni veriyle doğrulanana kadar).
#
# Bu yüzden ALPHA_MIN/ALPHA_MAX artık TEK bir global bant değil, slot bazlı:
# _SLOT_ALPHA_BANDS = {"0900": (min,max), "1700": (min,max)}. ALPHA_MIN/MAX
# modül sabitleri, slot eşleşmezse (örn. ileride yeni bir slot eklenirse)
# kullanılacak GERİYE DÖNÜK UYUMLU fallback olarak bırakıldı — optimize()
# artık bunları DOĞRUDAN kullanmıyor, _alpha_band_for(target_column) ile
# doğru slotun bandını seçiyor (bkz. aşağıda).
#
# Bu bantları periyodik olarak (özellikle veri belirgin şekilde değiştiğinde)
# ilgili slot için --alpha-sweep ile yeniden doğrulayın:
#     python optimize.py --alpha-sweep --slot 0900
#     python optimize.py --alpha-sweep --slot 1700
ALPHA_MIN = 0.60
ALPHA_MAX = 0.78

# Slot-bazlı alpha bandı — bkz. yukarıdaki v13 notu. 09:00, v14'te GERÇEK
# --alpha-sweep --slot 0900 sonucuna (decision_regret sütunlu, 21 alpha × 4
# fold) göre yeniden güncellendi; 17:00 de kendi --alpha-sweep --slot 1700
# sonucuna göre güncellendi.
_SLOT_ALPHA_BANDS = {
    # v14 — gerçek sweep sonucu: decision_regret 0.50→0.66 arası 1143.6→866.2'ye
    # (%24) düşerken WAPE bu aralıkta ~%47-50 arasında kalıp 0.66'da yeniden
    # ~%47.94 lokal minimuma dönüyor. 0.66'dan sonra WAPE hızla tırmanıyor
    # (0.80'de %58, 0.90'da %78) ama regret'in marjinal kazancı küçülüyor —
    # knee=0.66. Saf decision_regret minimumu sınırın ucunda (0.90) çıkıyor
    # çünkü spot_multiplier=9 çok agresif; model WAPE'yi hiç önemsemeden "hep
    # yüksek tahmin et" stratejisiyle regret'i sürekli azaltabiliyor — knee
    # kararı bu yüzden saf minimumdan değil, WAPE/regret ödünleşiminden verildi.
    "0900": (0.62, 0.70),
    # v14 — gerçek --alpha-sweep --slot 1700 sonucu: 0.50→0.58 arası WAPE zaten
    # düşerken (25.38%→24.64%, hybrid_score minimumu da tam burada) regret
    # %16 düşüyor — bedava kazanç. 0.58→0.70 arası WAPE ölçülü artıyor
    # (+2.73 puan) ama regret ek %13 düşmeye devam ediyor (4105.6→3565.7) ve
    # underpred_rate iyileşiyor (13.82%→11.26%) — hâlâ iyi bir takas.
    # 0.70'ten sonra getiri hızla küçülüyor (0.70→0.74: sadece 50 birim regret
    # kazancı için WAPE +1.99 puan). knee=0.70. Mevcut üretim parametresi
    # (hyperparams_map.json medium bucket, alpha=0.6821) zaten bu bandın
    # içinde — ek bir tutarlılık sinyali.
    "1700": (0.66, 0.74),
}


def _alpha_band_for(target_column: str) -> tuple:
    """
    target_column'a (TARGET_COL_0900 / TARGET_COL_1700) göre doğru slotun
    (alpha_min, alpha_max) bandını döndürür. Bilinmeyen bir target_column
    gelirse (örn. ileride yeni bir slot eklenirse) modül sabiti
    (ALPHA_MIN, ALPHA_MAX)'a düşer — sessizce yanlış banda düşmek yerine
    güvenli, dokümante edilmiş bir varsayılan.
    """
    if target_column == TARGET_COL_0900:
        return _SLOT_ALPHA_BANDS.get("0900", (ALPHA_MIN, ALPHA_MAX))
    if target_column == TARGET_COL_1700:
        return _SLOT_ALPHA_BANDS.get("1700", (ALPHA_MIN, ALPHA_MAX))
    return (ALPHA_MIN, ALPHA_MAX)

# Bant genişliği Σ(q90-q10)/Σy_true cezası — Optuna sadece WAPE+underprediction'a
# bakınca, WAPE'i minik bir miktar iyileştirmek için bandı aşırı genişletebiliyordu
# (gözlemlenen: rel_width=3.075, hiçbir cezası yokken). Bu ceza olmadan HPO,
# hangi config'in üretimde dar/geniş bant üreteceğini önemsemiyordu.
# v13: rel_width, alpha ile birlikte kayan "mid" tahmine (preds[:,1]) göre
# DEĞİL, Σy_true (WAPE'nin paydasıyla aynı, alpha'dan bağımsız sabit referans)
# ile normalize ediliyor — bkz. _correct_and_score() docstring'i.
# ⚠️ Bu ağırlık İLK TAHMİN — kesin kalibre edilmiş bir değer değil. rel_width
# tipik olarak 0.5-3.0 aralığında, WAPE ise 0.2-0.3 civarında; 0.03 ile
# rel_width=1.0 → skora +0.03 eklenir (WAPE'nin ~%10-15'i kadar, anlamlı ama
# baskın değil). v13'teki formül değişikliği rel_width'in SAYISAL ölçeğini de
# bir miktar değiştirebilir (artık mean yerine sum-oranı) — sonucu görüp
# gerekirse BETA_WIDTH'i yeniden kalibre edin.
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


def _resolve_cost_file(project_root: Path) -> str:
    """
    v16: Maliyet dosyasının yolu proje boyunca zaten 2 kez değişti
    (kök → sonra _arsiv_faz1/ altına arşivlendi, kökte onun yerine
    Araç_Kapasite_Maliyet_Saat.xlsx belirdi). Tek bir sabit yol yerine,
    BULUNAN İLK dosyayı kullanan bir öncelik listesi — klasör yapısı
    tekrar değişirse optimize.py sessizce fallback gamma'ya düşüp
    yanlışlıkla yanlış γ ile eğitim yapmak yerine, en azından gerçek
    veriden türetilmiş bir γ bulma şansını en üst düzeye çıkarır.
    Sıra: (1) yeni "_Saat" dosyası [en güncel/aktif olduğu varsayılır],
          (2) arşivlenmiş orijinal, (3) eski kök konum (geri gelirse).
    NOT: derive_gamma_from_costs() sütun adları uyuşmazsa zaten açıkça
    uyarıp fallback'e düşüyor — yani _Saat dosyasının sütun yapısı
    farklıysa bu SESSİZCE yanlış sonuç ÜRETMEZ, sadece uyarır.
    """
    candidates = [
        project_root / "data" / "raw" / "Araç_Kapasite_Maliyet_Saat.xlsx",
        project_root / "data" / "raw" / "_arsiv_faz1" / "Araç_Kapasite_Maliyet.xlsx",
        project_root / "data" / "raw" / "Araç_Kapasite_Maliyet.xlsx",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Hiçbiri yoksa ilkini döndür — derive_gamma_from_costs() zaten
    # "bulunamadı" uyarısını verip fallback gamma'ya düşecek.
    return str(candidates[0])


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
# Veri yükleme
# v12: load_data() KALDIRILDI. Artık run_forecast.py::load_dataset() DOĞRUDAN
# import edilip kullanılıyor (yukarıda) — rota×tarih×slot tam grid, eksikleri
# 0 ile doldurma, wide pivot (toplam_desi_0900/toplam_desi_1700) mantığı tek
# kaynaktan geliyor, iki dosya birbirinden kopamıyor.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fold tarihleri — forecasters.py::DemandForecaster.fit() ile BİREBİR AYNI
# (bkz. forecasters.py satır ~745-750 — doğrulandı). Bu pencereler, gerçek
# tahmin penceresine (PREDICT_START/END = 2026-06-29 → 2026-07-05,
# run_forecast.py) en yakın, veri içindeki son 4 tam hafta.
# v12: Önceki 2026-04-14 → 2026-05-10 pencereleri eski/kopmuş bir veri
# dönemine aitti — dosyanın kendi docstring'inde uyardığı "v5 aynı kalmalı,
# aksi halde farklı bir dönem için tuned olur" hatası tam olarak buydu.
# Biri değişirse diğeri MANUEL güncellenmeli.
FOLD_DATES = [
    ("Fold 1", "2026-05-31", "2026-06-06"),
    ("Fold 2", "2026-06-07", "2026-06-13"),
    ("Fold 3", "2026-06-14", "2026-06-20"),
    ("Fold 4", "2026-06-21", "2026-06-27"),
]


def _fit_fold_ensemble(feat_df, feature_cols, cat_features, params, target_column, fold_dates=FOLD_DATES):
    models = []
    for fold_name, val_start, val_end in fold_dates:
        fold_train = feat_df[feat_df[DATE_COL] < val_start]
        if fold_train.empty:
            logger.warning(f"   ⚠️  {fold_name}: train seti boş, atlanıyor.")
            continue
        X_ft = fold_train[feature_cols]
        y_ft = fold_train[target_column]
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

    v13 BUG DÜZELTMESİ — rel_width artık HAREKETLİ "mid" tahminine göre
    normalize edilmiyor:
      Eskiden: rel_width = mean((q90c-q10c) / max(preds, 1.0))
      preds = raw_preds[:, 1] = MultiQuantile'ın alpha-kuantili — ve alpha
      kendisi Optuna'nın arama uzayında (ALPHA_MIN..ALPHA_MAX). alpha
      büyüdükçe preds de (daha yüksek bir kuantil hedeflendiği için)
      sistematik olarak büyür — bant GENİŞLİĞİ (q90-q10) hiç değişmese
      bile rel_width SADECE alpha'nın kendisi büyüdüğü için küçülüyordu.
      Sonuç: Optuna, gerçekte bandı daraltmadan, sırf yüksek alpha seçerek
      rel_width cezasından "ucuza" kaçabiliyordu — bu da alpha seçimini
      objective üzerinden güvenilmez kılan asıl mekanizmaydı.

      Yeni: rel_width = Σ(q90c-q10c) / Σy_true — WAPE'nin paydasıyla
      (gerçekleşen toplam hacim) AYNI, alpha'dan TAMAMEN bağımsız sabit
      bir referans. Bant genişliği artık sadece kendi büyüklüğüyle
      ölçülüyor, alpha'nın hangi kuantili hedeflediğiyle karışmıyor.
      (Alternatif olarak ayrı, sabit bir 0.5-kuantil tahmini de kullanılabilir
      ama bu, ek bir model/tahmin gerektirir; Σy_true zaten elde olan,
      hesaplaması bedava ve WAPE ile tutarlı bir referanstır.)

    Dönüş: (score, rel_width_corrected, crossing_rate)
    """
    preds = raw_preds[:, 1]
    q10c  = np.minimum(raw_preds[:, 0], preds)   # DemandBand: q10 = min(q10, q50)
    q90c  = np.maximum(raw_preds[:, 2], preds)   # DemandBand: q90 = max(q90, q50)

    crossing_rate = float(np.mean(raw_preds[:, 2] < raw_preds[:, 0]))

    sum_true  = float(np.sum(y_true))
    rel_width = float(np.sum(q90c - q10c) / sum_true) if sum_true > 0 else 0.0

    score = (
        _hybrid_score(y_true, preds, gamma=gamma)
        + beta_width * rel_width
        + CROSSING_PENALTY_WEIGHT * crossing_rate
    )
    return score, rel_width, crossing_rate


def _fit_fold_ensemble_and_score(
    feat_df, feature_cols, cat_features, params, gamma, beta_width, target_column,
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
        y_ft = fold_train[target_column]
        model = CatBoostRegressor(**params)
        model.fit(Pool(X_ft, y_ft, cat_features=cat_features), verbose=False)
        models.append(model)

        X_fv = fold_val[feature_cols]
        y_fv = fold_val[target_column].values
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
    target_column: str,
    sibling_target_column: str,
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

    target_column/sibling_target_column : hangi slotun (09:00/17:00) tune
        edildiğini belirler — n_rows, feature matrix ve leakage-safe
        drop_cols hep buna göre hesaplanır (bkz. optimize() ile aynı mantık).

    v14: decision_regret sütunu eklendi (metrics.py::decision_regret,
    spot_multiplier=9.0). WAPE'yi minimize eden alpha ile gerçek operasyonel
    regret'i minimize eden alpha genelde FARKLI noktalarda knee yapar —
    WAPE eksik tahminle fazla tahmini simetrik cezalandırırken,
    decision_regret spot araç maliyetinin atıl kapasiteden ~9× pahalı
    olduğunu doğrudan modele yansıtır. "En iyi alpha" kararı artık
    hybrid_score'un değil, decision_regret'in makul bir WAPE artışı
    karşılığında belirgin düştüğü noktanın (knee) esas alınmasıyla verilmeli.

    Returns
    -------
    List[Dict] : her alpha için {alpha, wape, underpred_rate, rel_width,
                 crossing_rate, hybrid_score, decision_regret} — knee'yi
                 görmek için sırayla yazdırılmaya hazır.
    """
    full_df = load_dataset(data_path)
    # v12: n_rows artık SADECE bu slotun gerçek kayıtları üzerinden
    # hesaplanıyor (run_forecast.py::_fit_or_load_forecaster ile aynı satır).
    n_rows  = int((full_df[target_column] > 0).sum())
    lags    = select_lags(n_rows)

    feat_df = build_feature_matrix(
        df=full_df, target_columns=[target_column, sibling_target_column],
        date_column=DATE_COL, group_column=GROUP_COL, lags=lags,
        rolling_windows=ROLLING_WINDOWS, drop_na=False,
    )
    drop_cols    = get_drop_columns(target_column, sibling_target_column, feat_df.columns)
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
                Pool(fold_train[feature_cols], fold_train[target_column], cat_features=cat_features),
                verbose=False,
            )
            raw = np.maximum(model.predict(fold_val[feature_cols]), 0.0)

            y_true_all.append(fold_val[target_column].values)
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
        # v13: rel_width artık preds (alpha-kuantili, alpha ile birlikte kayar)
        # yerine Σy_true (alpha'dan bağımsız, sabit) ile normalize edilir —
        # bkz. _correct_and_score() docstring'i. Bu düzeltme olmadan sweep
        # tablosundaki rel_width sütunu, yüksek alpha'yı gerçekte bandı
        # daraltmadan yapay olarak ödüllendiriyordu.
        rel_width = float(np.sum(q90c - q10c) / sum_true) if sum_true > 0 else 0.0
        crossing_rate = float(np.mean(q90 < q10))
        # v14: decision_regret — spot_multiplier=9.0, metrics.py::decision_regret
        # ile aynı hesap. hybrid_score gibi bir vekil (proxy) değil, doğrudan
        # operasyonel maliyet birimiyle ifade edilen regret; knee kararı buna
        # göre verilmeli (bkz. fonksiyon docstring'i).
        regret_val = float(decision_regret(y_true, preds, spot_multiplier=9.0))

        hybrid = (
            wape_val
            + gamma * underpred_rate
            + BETA_WIDTH * rel_width
            + CROSSING_PENALTY_WEIGHT * crossing_rate
        )

        row = {
            "alpha":            round(alpha, 4),
            "wape":             round(wape_val, 6),
            "underpred_rate":   round(underpred_rate, 6),
            "rel_width":        round(rel_width, 4),
            "crossing_rate":    round(crossing_rate, 4),
            "hybrid_score":     round(hybrid, 6),
            "decision_regret":  round(regret_val, 6),
        }
        results.append(row)
        logger.info(
            f"   alpha={alpha:.2f} | WAPE={wape_val:.4%} | Underpred={underpred_rate:.4%} | "
            f"rel_width={rel_width:.3f} | crossing={crossing_rate:.2%} | hibrit={hybrid:.4%} | "
            f"regret={regret_val:.4f}"
        )

    return results


def print_alpha_sweep_table(results: List[Dict[str, Any]]) -> None:
    """Sweep sonuçlarını okunaklı tablo + knee önerisi olarak yazdırır."""
    print("\n" + "=" * 92)
    print("  ALPHA SWEEP SONUÇLARI")
    print("=" * 92)
    print(
        f"{'alpha':>7} | {'WAPE':>9} | {'Underpred':>10} | {'rel_width':>10} | "
        f"{'crossing':>9} | {'hibrit':>9} | {'regret':>10}"
    )
    print("-" * 92)
    for r in results:
        print(
            f"{r['alpha']:>7.2f} | {r['wape']:>8.2%} | {r['underpred_rate']:>9.2%} | "
            f"{r['rel_width']:>10.3f} | {r['crossing_rate']:>8.2%} | {r['hybrid_score']:>8.2%} | "
            f"{r['decision_regret']:>10.4f}"
        )
    print("-" * 92)

    best_hybrid = min(results, key=lambda r: r["hybrid_score"])
    best_regret = min(results, key=lambda r: r["decision_regret"])
    print(f"\n💡 Hibrit skora göre en iyi: alpha={best_hybrid['alpha']:.2f}")
    print(f"💡 Decision_regret'e göre en iyi (spot_multiplier=9.0): alpha={best_regret['alpha']:.2f}")
    if best_hybrid["alpha"] != best_regret["alpha"]:
        print(
            "   ⚠️  Bu iki nokta FARKLI — hibrit skor minimumu WAPE + gamma·underpred\n"
            "   ağırlıklı bir vekil (proxy); decision_regret ise doğrudan spot araç\n"
            "   maliyeti biriminde. Alpha kararını hibrit minimumdan değil, aşağıdaki\n"
            "   knee mantığıyla decision_regret'ten verin."
        )
    print(
        "   NOT: Tabloyu gözle de inceleyin — 'knee' noktası (decision_regret'in\n"
        "   makul bir WAPE artışı karşılığında belirgin düştüğü, sonrasında ise\n"
        "   düzleştiği alpha) genelde saf minimum decision_regret'ten de 1-2\n"
        "   basamak farklı, daha dengeli bir üretim seçimi olabilir (örn. regret\n"
        "   minimumu alpha=0.74 ise ama 0.66-0.68 aralığında regret zaten çoğu\n"
        "   kazanımı yakalamışsa ve WAPE hâlâ makulse, 0.66-0.68 daha güvenli bir\n"
        "   seçim olabilir)."
    )
    print("=" * 92)


# ---------------------------------------------------------------------------
# Optuna Optimizasyonu
# ---------------------------------------------------------------------------

def optimize(
    data_path: str,
    target_column: str,
    sibling_target_column: str,
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

    target_column / sibling_target_column : hangi slotun (09:00/17:00)
        modeli tune ediliyor — run_forecast.py::_fit_or_load_forecaster
        çağrısıyla AYNI mantık. n_rows, feature matrix, drop_cols hep buna
        göre hesaplanır (bkz. get_drop_columns()).

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

    v12 değişiklikleri (slot-farkındalığı — run_forecast.py ile tam senkron):
      5. load_data() yerine run_forecast.py::load_dataset() (wide/slot format).
      6. build_feature_matrix artık target_columns=[target_column,
         sibling_target_column] (liste) kullanıyor.
      7. n_rows SADECE bu slotun gerçek kayıtları üzerinden hesaplanıyor.
      8. drop_cols artık forecasters.py::_get_drop_columns ile birebir aynı
         (leakage-safe) — bkz. get_drop_columns().
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("🔍 Optuna Hiperparametre Optimizasyonu  (v6 — Walk-Forward CV + Pruning)")
    logger.info("=" * 60)

    # v13: alpha bandı artık slot-bazlı (bkz. _SLOT_ALPHA_BANDS / _alpha_band_for)
    # — tek global ALPHA_MIN/ALPHA_MAX yerine, her slot kendi kanıtlanmış
    # (--alpha-sweep) bandını kullanır.
    alpha_min, alpha_max = _alpha_band_for(target_column)

    if gamma > 0:
        logger.info(
            f"   γ (gamma) = {gamma:.2f}  →  spot maliyet ≈ {gamma + 1:.1f}× atıl maliyet\n"
            f"   Objective : WAPE + {gamma:.2f} × Underprediction_Penalty\n"
            f"   alpha arama aralığı: [{alpha_min}, {alpha_max}]  (slot='{target_column}')"
        )
    else:
        logger.info("   γ = 0 → Simetrik WAPE modu (asimetrik ceza devre dışı)")

    logger.info(f"   Slot: target_column='{target_column}' | sibling_target_column='{sibling_target_column}'")

    # Veri — v12: run_forecast.py::load_dataset() (wide/slot format)
    full_df = load_dataset(data_path)
    # v12: n_rows artık SADECE bu slotun gerçek kayıtları üzerinden
    # hesaplanıyor (run_forecast.py::_fit_or_load_forecaster ile birebir
    # aynı satır: n_real_rows = int((full_df[target_column] > 0).sum())).
    n_rows  = int((full_df[target_column] > 0).sum())
    bucket  = _bucket_name(n_rows)
    space   = _search_space(n_rows)

    logger.info(f"\n   Gerçek kayıt ({target_column}): {n_rows:,} | Bucket: {bucket}")

    # Feature engineering
    # lags → veri büyüklüğüne göre uyarlanır (bkz. select_lags() ve LAG_21_MIN_ROWS/LAG_30_MIN_ROWS).
    # rolling_windows → run_forecast.py (DemandForecaster) ile birebir eşit.
    lags = select_lags(n_rows)
    logger.info(f"   Kullanılan lag'ler: {lags}  (eşikler: lag_21≥{LAG_21_MIN_ROWS:,}, lag_30≥{LAG_30_MIN_ROWS:,})")
    # v12: target_columns artık LİSTE — forecasters.py::_engineer_features ile
    # birebir aynı (hub/graph/hiyerarşik/cross-lag feature'lar her iki slotun
    # da var olmasını gerektiriyor, bkz. get_drop_columns() ve build_feature_matrix).
    feat_df = build_feature_matrix(
        df             = full_df,
        target_columns = [target_column, sibling_target_column],
        date_column    = DATE_COL,
        group_column   = GROUP_COL,
        lags           = lags,
        rolling_windows= ROLLING_WINDOWS,
        drop_na        = False,   # fix #2: forecasters.py::fit() ile AYNI
    )

    # Anormal haftaları tespit et (tatil/birikim → temiz-WAPE hesabında dışlanır)
    weekly_avg  = full_df[full_df[target_column] > 0].copy()
    weekly_avg["week"] = weekly_avg[DATE_COL].dt.isocalendar().week.astype(int)
    weekly_avg["year"] = weekly_avg[DATE_COL].dt.year
    wk_means    = weekly_avg.groupby(["year", "week"])[target_column].mean()
    threshold   = wk_means.mean() * 1.4
    abnormal_wk = set(wk_means[wk_means > threshold].index)
    if abnormal_wk:
        logger.info(f"⚠️  Anormal haftalar dışlandı: {abnormal_wk}")

    # v12: leakage-safe drop — forecasters.py::_get_drop_columns ile birebir
    # aynı (bkz. get_drop_columns() yukarıda). Naif [TARGET_COL, DATE_COL]
    # yerine slot-farkındalıklı kural: 09:00 modeli için sibling
    # (toplam_desi_1700) ve cross_lag_0900_same_day KESİNLİKLE drop edilir.
    drop_cols    = get_drop_columns(target_column, sibling_target_column, feat_df.columns)
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
    y_val       = val_df[target_column].values
    clean_mask  = (~is_abnormal).values
    y_val_clean = val_df.loc[~is_abnormal, target_column].values

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
        # v13: artık slot-bazlı bant (alpha_min/alpha_max, yukarıda
        # _alpha_band_for(target_column) ile hesaplandı) kullanılıyor —
        # tek global ALPHA_MIN/ALPHA_MAX DEĞİL.
        # v9: alpha_min==alpha_max ise (--alpha-sweep ile knee bulunup
        # sabitlendiyse) Optuna'ya boş bir boyut açtırmıyoruz, direkt sabit
        # değeri kullanıyoruz — arama artık sadece diğer 5 parametrede.
        if alpha_min == alpha_max:
            alpha = alpha_min
        elif gamma > 0:
            alpha = trial.suggest_float("alpha", alpha_min, alpha_max)
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
            y_ft = fold_train[target_column]
            fold_model = CatBoostRegressor(**params)
            fold_model.fit(Pool(X_ft, y_ft, cat_features=cat_features), verbose=False)
            models.append(fold_model)

            X_fv = fold_val[feature_cols]
            y_fv = fold_val[target_column].values
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
    best_models    = _fit_fold_ensemble(feat_df, feature_cols, cat_features, best_params_full, target_column)
    best_raw       = _ensemble_predict(best_models, X_val)
    best_preds     = best_raw[:, 1]
    best_q10_corr  = np.minimum(best_raw[:, 0], best_preds)
    best_q90_corr  = np.maximum(best_raw[:, 2], best_preds)
    # v13: Σy_val ile normalize (bkz. _correct_and_score) — alpha'dan bağımsız sabit referans.
    _sum_y_val     = float(np.sum(y_val))
    best_rel_width = float(np.sum(best_q90_corr - best_q10_corr) / _sum_y_val) if _sum_y_val > 0 else 0.0
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
        f"   Bant genişliği    : Σ(q90-q10)/Σy_val ≈ {best_rel_width:.3f}  "
        f"(artık skora dahil - BETA_WIDTH={BETA_WIDTH}, monotonluk düzeltmeli, alpha'dan bağımsız sabit referans)\n"
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
        "target_column":             target_column,          # v12: hangi slot (0900/1700) tune edildi — izlenebilirlik
        "sibling_target_column":     sibling_target_column,   # v12: leakage kuralının hangi sibling'e uygulandığı
        "row_count":                 n_rows,
        "lags_used":                 lags,   # run_forecast.py'nin AYNI lag'leri kullanması gerekir — bkz. select_lags()
        "gamma":                     gamma,
        "gamma_source":              gamma_note,  # γ nereden geldi (dosya/sütun/formül) — izlenebilirlik için
        "best_hybrid_score":         round(study.best_value, 6),
        "best_wape":                 round(pure_wape, 6),
        "best_wape_clean":           round(wape_clean, 6),   # YENİ: anormal hafta hariç WAPE
        "underprediction_rate":      round(underpred_rt, 6),
        "rel_width_q90_q10":         round(best_rel_width, 4),  # v13: Σy_val ile normalize, alpha'dan bağımsız (monotonluk düzeltmeli, artık negatif çıkmaz)
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
        with open(HYPERPARAMS_PATH, encoding="utf-8") as f:
            hmap = json.load(f)
    else:
        hmap = {}

    # v12: Adım 7'nin varsayımı — 09:00 (n≈24.9k → "small") ve 17:00
    # (n≈41.1k → "medium") FARKLI bucket'lara düşer, bu yüzden normalde
    # birbirini ezmezler. Ama veri değiştikçe bu garanti değil; aynı
    # bucket'a farklı bir slot'un sonucu yazılmak üzereyse sessizce
    # ezmek yerine uyar (izlenebilirlik).
    prev = hmap.get(bucket)
    if prev and prev.get("target_column") not in (None, target_column):
        logger.warning(
            f"⚠️  Bucket '{bucket}' daha önce farklı bir slot için tune edilmişti "
            f"(önceki target_column='{prev.get('target_column')}', şimdiki='{target_column}'). "
            f"Üzerine yazılıyor — bu iki slotun veri boyutu artık aynı bucket'a düşüyor demektir."
        )

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
    # v12: varsayılan veri artık teknofest26_gelismis.xlsx (slot bazlı,
    # run_forecast.py::DATA_PATH ile aynı) — eski Desi_talep.xlsx tek-slotlu
    # akışına ait değildi.
    parser.add_argument("--data",    default=str(_PROJECT_ROOT / "data" / "raw" / "teknofest26_gelismis.xlsx"), help="Veri dosyası")
    # v12: Adım 7 — 09:00 ve 17:00 AYRI çalıştırmalar gerektiriyor (farklı
    # v12: --slot artık ZORUNLU DEĞİL. Verilmezse (varsayılan "both") tek
    # çalıştırmada 09:00 VE 17:00 SIRAYLA tune edilir (aşağıdaki döngüde) —
    # farklı n_rows → farklı bucket → _search_space zaten otomatik ayrışıyor,
    # bu yüzden aynı anda iki bucket'ı güvenle üretebiliyoruz. Tek bir slotu
    # tune etmek istersen --slot 0900 veya --slot 1700 ile de çalıştırabilirsin.
    parser.add_argument("--slot", default="both", choices=["0900", "1700", "both"],
                        help="Hangi slot tune edilsin: '0900', '1700' veya 'both' "
                             "(varsayılan — ikisini de SIRAYLA tek çalıştırmada tune eder). "
                             "run_forecast.py'nin iki ayrı modeliyle (09:00/17:00) eşleşir.")
    parser.add_argument("--trials",  type=int,   default=50,           help="Optuna trial sayısı")
    parser.add_argument("--timeout", type=int,   default=1800,
                        help="Timeout (saniye) — 0 = sınırsız.")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Optuna TPESampler seed — aynı seed + aynı veri = aynı trial dizisi (tekrarlanabilirlik).")
    parser.add_argument("--gamma",   type=float, default=None,
                        help="Asimetrik ceza katsayısı (C_spot/C_atıl - 1). "
                             "Verilmezse --cost-file'dan otomatik türetilir.")
    parser.add_argument("--cost-file", default=_resolve_cost_file(_PROJECT_ROOT),
                        help="Spot/atıl maliyet oranının türetileceği dosya (--gamma verilmezse kullanılır). "
                             "Varsayılan: Araç_Kapasite_Maliyet_Saat.xlsx > _arsiv_faz1/Araç_Kapasite_Maliyet.xlsx "
                             "> eski kök konum, sırayla bulunan ilki (bkz. _resolve_cost_file()).")
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

    # v12: --slot "both" (varsayılan) → tek çalıştırmada 09:00 VE 17:00
    # SIRAYLA işlenir (aşağıdaki döngü). Tek bir slot istenirse liste tek
    # elemanlı olur — kod yolu aynı, sadece döngü 1 kez döner.
    slots_to_run = ["0900", "1700"] if args.slot == "both" else [args.slot]
    if len(slots_to_run) > 1:
        logger.info(
            f"🔁 --slot both → 09:00 ve 17:00 SIRAYLA tune edilecek "
            f"(toplam süre tek slota göre ~2 katı olur; her biri kendi "
            f"--timeout/--trials sınırına tabidir)."
        )

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
        for slot in slots_to_run:
            target_column, sibling_target_column = _SLOT_TARGETS[slot]
            logger.info(f"\n{'='*60}\n🔬 Alpha Sweep — slot {slot} ({target_column})\n{'='*60}")
            sweep_results = alpha_sweep(
                args.data, fixed_params=fixed_params, gamma=gamma,
                alphas=alphas, target_column=target_column,
                sibling_target_column=sibling_target_column, hpo_folds=args.hpo_folds,
            )
            print_alpha_sweep_table(sweep_results)
        sys.exit(0)

    all_results = {}
    for slot in slots_to_run:
        target_column, sibling_target_column = _SLOT_TARGETS[slot]
        logger.info(f"\n{'='*60}\n🎯 Optimizasyon — slot {slot} ({target_column})\n{'='*60}")
        all_results[slot] = optimize(
            args.data, target_column=target_column, sibling_target_column=sibling_target_column,
            n_trials=args.trials, timeout=timeout,
            gamma=gamma, gamma_note=gamma_note, optuna_seed=args.seed,
            hpo_folds=args.hpo_folds,
        )

    if len(slots_to_run) > 1:
        print(f"\nJSON'a yazılan parametreler (her iki slot):\n{json.dumps(all_results, indent=2)}")
    else:
        print(f"\nJSON'a yazılan parametreler:\n{json.dumps(all_results[slots_to_run[0]], indent=2)}")