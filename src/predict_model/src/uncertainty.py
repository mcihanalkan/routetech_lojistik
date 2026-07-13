# src/uncertainty.py
"""
Belirsizlik Yönetimi ve ALNS Payload Modülü

Sorumluluk:
  DemandForecaster.predict() çıktısındaki ham kantil bantlarını alır,
  iş kurallarını uygular ve ALNS optimizasyon motorunun tüketeceği
  nihai in-memory payload'ı üretir.

Veri Akışı:
  Ham Veri
    → DemandForecaster.predict()     [List[Dict]: q10/q50/q90]
      → UncertaintyBand.from_json()  [band nesneleri + risk sınıfı]
        → to_alns_payload()          [ALNS'in beklediği final format]
          → ALNS Motoru              [araç ataması + rota optimizasyonu]

ALNS Payload Formatı (Tur 5 — Hacim Ağırlıklı Dinamik Sigmoid + Materyalite Ağırlığı):
  {
    "metadata": {
      "generated_at": ..., "n_records": ..., "horizon_days": ...,
      "risk_model": {"name": "volume_weighted_dynamic_sigmoid",
                      "tau_base": 0.50, "kappa": 5.0, "beta": 0.30,
                      "k_min": 2.0, "gamma": 1.0,
                      "materiality_floor": 750.0}
    },
    "demands":  [
      {
        "tarih":               "2026-01-08",
        "TM_ID":               "IST-01",
        "demand_low":          142.3,   ← q10 (kötümser alt sınır)
        "demand_base":         198.7,   ← q50 (operasyonel plan tahmini)
        "demand_high":         267.4,   ← q90 (spot araç alarm seviyesi)
        "uncertainty_range":   125.1,   ← q90 - q10
        "relative_uncertainty":0.630,   ← U_rel = uncertainty_range / max(q50,1)
        "dynamic_threshold":   0.271,   ← τ(V), hacme özel kabul edilebilir U_rel eşiği
        "dynamic_steepness":   14.59,   ← k(V), hacme özel sigmoid eğri katılığı
        "risk_score_raw":      0.93,    ← [Tur 5] materyalite ağırlığı öncesi HAM sigmoid skoru
        "risk_score":          0.36,    ← [Tur 5] materyalite ağırlıklı NİHAİ skor (0-1)
        "safety_buffer":       34.4,    ← (q90 - q50) × buffer_ratio
        "risk_class":          "MEDIUM",← LOW / MEDIUM / HIGH (nihai risk_score'dan türetilir)
        "recommended_qty":     232.1,   ← ALNS'e önerilen kapasite rezervasyonu
      },
      ...
    ]
  }

--- Tur 5 Değişikliği (Materyalite Ağırlığı) — NEDEN GEREKLİ ---
Üretim çalıştırmasında (623 kayıt, run_forecast.py) gözlemlenen sorun:
  U_rel ortalaması 2.538 (!) ve HIGH oranı 64/623 (%10.3) — ikisi de PDF'in
  Tur 3 kalibrasyonunun hedeflediği makul aralığın çok üzerinde.

  Kök neden: relative_uncertainty = uncertainty_range / max(q50, 1.0).
  Tahmin ufkunun son günlerinde (özellikle düşük hacimli / durgun rotalarda)
  q50 sıfıra çok yakın çıkabiliyor (örn. q50=0 → v_safe=1.0 payda). Bu durumda
  q90-q10 farkı sadece birkaç yüz desi bile olsa oran onlarca-yüzlerce kat
  şişiyor (gözlemlenen uç örnek: "Tekirdağ → Denizli" 16 Mayıs, q50=0,
  uncertainty_range=125.25 → relative_uncertainty=125.25 → risk_score≈1.0 → HIGH).
  Bu gerçek bir operasyonel risk DEĞİL — birkaç yüz desilik bir sapma, zaten
  neredeyse boş olan bir rotada spot araç çağırmayı gerektirmez; sadece
  payda küçüklüğünden kaynaklanan matematiksel bir artefakttır.

  Çözüm — Materyalite Ağırlığı (materiality weight):
    weight(V) = min(1.0, V / materiality_floor)
    risk_score_final = risk_score_raw × sqrt(weight(V))
  materiality_floor, bu filonun gözlemlenen p10 hacmine (~734 desi — bkz.
  modül üstü Tur 2/3 notları) yakın tutuldu (750.0). Böylece:
    - q50 ≥ floor olan rotalarda davranış DEĞİŞMEZ (weight=1.0, Tur 3 ile birebir aynı).
    - q50 << floor olan (yapısal olarak önemsiz hacimli) rotalarda risk_score
      orantılı şekilde bastırılır — sıfır hacimli bir rota ASLA HIGH çıkamaz.
  Bu, sert bir eşik/kesme (hard cutoff) DEĞİL, sürekli bir sönümleme
  fonksiyonudur — PDF'in "dinamik" felsefesiyle tutarlı, ani sınıf sıçramaları
  yaratmaz. risk_score_raw da payload'a eklendi (tanı/denetim amaçlı) —
  ALNS motoru sadece nihai `risk_score` alanını okumaya devam eder,
  şema geriye dönük uyumludur (sadece yeni bir alan eklendi).

Not (Tur 2 → Tur 5 tarihçesi): Eski sabit-eşik sınıflandırması (ratio =
(q90-q10)/q50 > 0.40 → HIGH), Tur 2'de hacim ağırlıklı dinamik eşik + sigmoid
risk skoruna, Tur 3'te bu filonun gerçek U_rel tabanına (~%55) kalibre edilmiş
τ_base/κ değerlerine, Tur 5'te ise düşük-hacim payda patlamasına karşı
materyalite ağırlığına evrilmiştir. Üretilen payload alanları geriye dönük
uyumludur (eski alanlar korunmuştur), yeni alan (`risk_score_raw`) eklenmiştir.
"""

import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sabitler — Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli (Tur 2-3)
# ---------------------------------------------------------------------------
#
# (Tur 2/Tur 3 kalibrasyon notları — DEĞİŞMEDİ, bkz. dosya geçmişi için git blame)
# Kalibrasyon notu (Desi_talep.xlsx + Arac__Kapasite_Maliyet.xlsx):
#   - q50 (Toplam Desi) dağılımı: min ≈ 5, p10 ≈ 734, medyan ≈ 9.450,
#     p90 ≈ 24.755, max ≈ 67.800 desi.
#   - Filo kapasiteleri: Kamyonet 5.600 / Hafif Kamyon 7.200 / Kamyon 12.000 /
#     Tır 22.400 desi.
#   - [Tur 3] Gerçek model çıktısı (623 tahmin): q50 ortalaması ≈ 11.101 desi,
#     ortalama U_rel ≈ %55. τ_base=0.50, κ=5.0, k_min=2.0, γ=1.0, β=0.30
#     kalibrasyonu bu tabana göre ayarlandı.
#   - [Tur 5] Aynı 623 kayıtlık gerçek üretim çalıştırmasında q50'nin sıfıra
#     yakın olduğu kuyruk günlerinde (özellikle ufkun son 2-3 günü, durgun
#     rotalar) relative_uncertainty'nin patladığı ve HIGH oranını yapay olarak
#     şişirdiği gözlendi (bkz. modül üstü "Tur 5 Değişikliği" notu).
#     τ_base/κ/β/k_min/γ SABİT bırakıldı (yapısal U_rel tabanı hâlâ geçerli);
#     bunun yerine ayrı bir materyalite ağırlığı katmanı eklendi.

# --- [Tur 6] MAAPE / Arktanjant Dönüşümü ------------------------------------
# NEDEN GEREKLİ (bkz. "Seyrek Talepte Standart Hata Metriklerinin Sıkıntısı" PDF):
#   relative_uncertainty = uncertainty_range / max(q50, 1.0) matematiksel olarak
#   SINIRSIZDIR. q50 sıfıra yakınsa (09:00 slotu, durgun rota) oran yüzlerce/
#   binlerce % çıkabilir (gözlemlenen örnek: %3300). Tur 5'teki materyalite
#   ağırlığı bunu SONRADAN (sigmoid skorunu çarparak) telafi ediyordu; ama
#   relative_uncertainty alanının kendisi hâlâ patlak ve dynamic_threshold
#   (τ_base=0.50 gibi normal-ölçek sabitlerle) karşılaştırıldığında tau_base'i
#   yapay şekilde şişirmeye devam ediyordu (log: "τ_base 4.90'a şişiyor").
#
#   Çözüm — PDF'in önerdiği MAAPE (Arctan APE) mantığı: ham oranı arctan'dan
#   geçirip [0, π/2) aralığını π/2'ye bölerek [0, 1) aralığına normalize
#   ediyoruz. arctan asimptotik olduğu için x→∞ iken sonuç 1'e YAKLAŞIR ama
#   ASLA AŞMAZ — payda küçüklüğü artık hiçbir şekilde skoru patlatamaz.
#   Kapasite bazlı U_smart'tan (raporun alternatif önerisi) farkı: yeni bir
#   dış parametre (araç/filo kapasitesi) gerektirmez, mevcut materiality_floor
#   katmanıyla birlikte çalışır (iki savunma katmanı: 1) burada oran sınırlanır,
#   2) Tur 5'teki sqrt(weight) düşük hacimde nihai skoru ayrıca bastırır).
#
#   ÖNEMLİ: derive_risk_params_from_data() içindeki tau_base türetimi de AYNI
#   dönüşümden geçirilir (aşağıda _maape_scale kullanılıyor) — aksi halde
#   runtime'daki bounded relative_uncertainty, unbounded ham veriden türetilmiş
#   bir eşikle kıyaslanır ve sorun başka bir biçimde geri döner.
#
# --- [Tur 7] Ölçek (scale) parametresi — GERÇEK ÜRETİM VERİSİNDE BULUNAN KUSUR ---
# run_forecast.py çalıştırıldığında 09:00 slotu için (seyrek/durgun, çarpıklık
# +4.76) auto-derive edilen tau_base=0.872 çıktı — relative_uncertainty'nin
# tavanı (1.0) ile arasında sadece 0.128 boşluk kalıyor, k(V)~7 civarı bir
# eğimle HIGH eşiğine (sigmoid=0.82) ulaşmak için gereken ~0.22'lik fark hiçbir
# satır için mümkün olamıyor → 2023 kayıtta HIGH=0 (bkz. gerçek log). Kök neden:
# düz arctan(x), "tipik" (medyan) ham oranı DOĞRUDAN [0,1) ölçeğine taşıyor;
# 09:00 gibi yapısal olarak seyrek bir slotta tipik ham oran zaten ~%491 gibi
# devasa olduğundan, arctan(4.91)≈0.872 zaten tavana yapışıyor — outlier'lara
# hiç yer kalmıyor.
#
# Çözüm: arctan(x) yerine arctan(x / scale) kullanılır; scale, o slotun/alt
# kümenin KENDİ tipik (medyan) ham oranına eşitlenir. Böylece TANIM GEREĞİ
# medyan satır tam olarak 0.5'e denk gelir (arctan(1)·2/π=0.5) — slotlar
# arası "tipik büyüklük" farkı artık scale'e taşınır, tau_base HER SLOTTA
# aynı, taşınabilir/sabit bir sayı (örn. 0.45) olarak kalabilir ve gerçekten
# aşırı (medyanın kat kat üstü) satırlar hâlâ 1.0'a doğru ilerleyip HIGH
# tetikleyebilir. scale <= 0 (örn. tüm gözlemler sıfırsa) ise 1.0'a düşülür
# (eski/scale'siz davranışla aynı — geriye dönük güvenli varsayılan).
def _maape_scale(raw_ratio: float, scale: float = 1.0) -> float:
    """
    MAAPE/Arktanjant dönüşümü: ham (sınırsız) oranı [0, 1) aralığına sıkıştırır.

        f(x) = (2/π) · arctan(x / scale)

    Özellikler:
      f(0)        = 0.0    → belirsizlik yok
      f(scale)    = 0.5    → oran, o alt kümenin TİPİK (medyan) değerine eşit → orta seviye
      f(inf)      = 1.0    → asimptot, ASLA aşılmaz (klasik APE/MAPE'nin aksine)

    Parameters
    ----------
    raw_ratio : Ham (sınırsız) oran, örn. uncertainty_range / max(q50, 1.0).
    scale     : [Tur 7] Referans/tipik ölçek. Varsayılan 1.0 (Tur 6'daki eski
                davranışla birebir aynı — geriye dönük uyumlu). Genelde o
                slotun/alt kümenin ham oranının MEDYANI verilir (bkz.
                derive_risk_params_from_data, "maape_scale" dönüş değeri) —
                böylece farklı büyüklükteki slotlar (09:00 vs 17:00) kendi
                doğal ölçeklerine göre normalize olur, tau_base sabit kalabilir.

    raw_ratio negatif olamaz (uncertainty_range ve v_safe her zaman ≥ 0),
    yine de savunma amaçlı 0'ın altına düşürülmez. scale <= 0 ise (degenerate
    durum) 1.0'a düşülür — sıfıra bölme koruması.
    """
    raw_ratio = max(float(raw_ratio), 0.0)
    scale = float(scale)
    if scale <= 0.0:
        scale = 1.0
    return float((2.0 / np.pi) * np.arctan(raw_ratio / scale))


# [Tur 7] Varsayılan ölçek — slot-bazlı türetim yapılmazsa (derive_risk_params_
# from_data çağrılmazsa) bu kullanılır; Tur 6'daki eski (scale'siz) davranışla
# birebir aynıdır, geriye dönük uyumluluk için.
DEFAULT_MAAPE_SCALE: float = 1.0


# τ(V) = τ_base + κ · V^(-β)  → hacme göre maks. kabul edilebilir U_rel eşiği
#
# [Tur 7 KALİBRASYON NOTU] scale artık slotlar arası büyüklük farkını
# kendi içinde emdiği için (bkz. yukarıdaki _maape_scale notu), tau_base
# BAŞKA BİR ŞEYE ihtiyaç duymadan TÜM slotlarda aynı, taşınabilir bir sayı
# olarak kalabilir — "medyan satır → 0.5" tanımı scale ile zaten sağlanıyor,
# tau_base sadece bu 0.5 tipik noktasının etrafında ince ayar (headroom) yapar.
# 0.45 seçildi: medyanın biraz altı → tipik/medyan civarındaki satırlar zaten
# MEDIUM sınıfına yaklaşmaya başlasın, ama HIGH'a (sigmoid 0.82) ulaşmak için
# hâlâ gerçek bir outlier (medyanın kat kat üstü ham oran) gerekiyor.
# κ=0.35: en düşük hacimde (V=1) τ(V)=0.45+0.35=0.80 — 1.0 tavanının hâlâ
# altında, düşük hacme gevşeme payı veriyor ama HIGH'ı imkânsız kılmıyor
# (eski Tur 6 izdüşümünde κ=0.45 ile τ(1)=0.74 idi; burada tau_base 0.29'dan
# 0.45'e çıktığı için κ hafifçe düşürüldü, aynı ~0.80 tavan mantığı korundu).
DYNAMIC_TAU_BASE: float = 0.45   # [Tur 7] scale-normalize edilmiş taban — TÜM slotlarda taşınabilir/sabit
DYNAMIC_KAPPA:    float = 0.35   # [Tur 7] düşük hacim gevşeme payı — τ(V=1)=0.80, 1.0 tavanının altında kalır
DYNAMIC_BETA:     float = 0.30   # Sönümleme oranı (V^-β kuvvet yasası) — hacim ekseni değişmedi, AYNEN kalır

# k(V) = k_min + γ · log(1 + V)  → hacme göre sigmoid eğri katılığı
DYNAMIC_K_MIN: float = 2.0   # En küçük hacimlerde min. eğim
DYNAMIC_GAMMA: float = 1.0   # Hacim arttıkça eğrinin katılaşma hızı

# --- [Tur 5] Materyalite Ağırlığı ------------------------------------------
# weight(V) = min(1.0, V / MATERIALITY_FLOOR)
# risk_score_final = risk_score_raw × sqrt(weight(V))
#
# Neden 750.0? Gözlemlenen desi dağılımının p10'una (~734) yakın tutuldu —
# yani filonun "yapısal olarak düşük ama hâlâ gerçek" hacimlerinin alt
# sınırına denk geliyor. Bunun altındaki hacimler (kuyruk günleri, neredeyse
# durgun rotalar) için mutlak etki zaten küçük olduğundan, göreceli
# belirsizlik ne kadar patlarsa patlasın nihai risk skoru orantılı olarak
# bastırılır. q50=0 olan bir satır DAİMA weight=0 → risk_score=0 → LOW alır.
MATERIALITY_FLOOR: float = 5000.0

# --- [Düzeltme: derive_risk_params_from_data()] Sıfır-hariç istatistik + dinamik alt taban ---
# compute_target_skewness'in zaten uyguladığı "gözlem, sıfır hariç" mantığıyla
# tutarlı olacak şekilde: percentile/tau hesaplarına giren q50 dağılımından
# sıfıra çok yakın (durgun/talep-yok) satırlar dışlanır — aksi halde bu
# satırlar özellikle p25 gibi düşük bir persentili yapay şekilde aşağı çeker.
MIN_NONZERO_THRESHOLD:    float = 1.0    # Bu eşiğin altındaki q50 "durgun" sayılır, istatistikten dışlanır
ABSOLUTE_MIN_FLOOR:       float = 50.0   # materiality_floor asla bu değerin altına inmez (eski 1.0 anlamsız düşüktü)
MIN_FLOOR_FRAC_OF_MEDIAN: float = 0.10   # min_floor elle verilmezse: nonzero q50 medyanının bu kesri

# --- [Düzeltme: tau_base için ek alt eşik] ---
# MIN_NONZERO_THRESHOLD (1.0) sadece tam sıfıra yakın satırları eler;
# q50=2-5 gibi "teknik olarak nonzero ama slotun kendi ölçeğine göre
# anlamsız derecede küçük" satırlar hâlâ tau hesabına dahil olur. Bu tür
# satırlarda q90 birkaç kat büyük olsa bile (örn. q50=3, q90=45) U_rel
# devasa çıkar ve tau_base'i yukarı şişirir. Bu yüzden tau_base hesabı,
# floor hesabından daha SIKI bir alt eşikle (nonzero q50 dağılımının
# kendi p10'u) filtrelenir — bkz. derive_risk_params_from_data(),
# tau_min_volume_percentile parametresi.
TAU_MIN_VOLUME_PERCENTILE: float = 10.0  # tau hesabına giren satırlar için nonzero q50'nin bu persentilinin üzerinde olmalı

# Sürekli risk skorunu (0-1) operasyonel etiketlere bölen sınırlar (PDF Bölüm 9)
RISK_SCORE_LOW_MAX:    float = 0.35   # Risk_Score < 0.35  → LOW
RISK_SCORE_MEDIUM_MAX: float = 0.82   # 0.35 ≤ Risk_Score ≤ 0.82 → MEDIUM
                                       # Risk_Score > 0.82  → HIGH

# Floating-point overflow koruması (sigmoid exponent clipping)
SIGMOID_EXP_CLIP: float = 100.0

# --- Geriye dönük uyumluluk (artık _classify_risk içinde kullanılmıyor) ---
# Eski sabit-eşik sabitleri sadece referans/log amaçlı tutulur.
RISK_THRESHOLD_LOW:    float = 0.40   # [DEPRECATED] eski sabit eşik (ratio < 0.40 → LOW)
RISK_THRESHOLD_MEDIUM: float = 1.00   # [DEPRECATED] eski sabit eşik (ratio < 1.00 → MEDIUM)

# Güvenlik tamponu: q90 ile q50 arasındaki farkın kaçı eklenir?
# ALNS bunu "kapasite rezervasyonu" olarak kullanır
DEFAULT_BUFFER_RATIO: float = 0.5


# ---------------------------------------------------------------------------
# derive_risk_params_from_data() — Slot-Bazlı / Veri-Türetimli Risk Parametreleri
# ---------------------------------------------------------------------------
#
# NEDEN GEREKLİ:
#   MATERIALITY_FLOOR (ve tau_base/kappa/beta) yukarıda TEK bir sabit sayı
#   olarak tanımlı. Ancak 09:00 ve 17:00 slotlarının hacim (q50) dağılımları
#   yapısal olarak çok farklı (bkz. to_ortools_dataframe altındaki slot-bazlı
#   log notu: 09:00 ort. ~517, 17:00 ort. ~3225 — ~6 kat fark). Sabit bir
#   materiality_floor kullanmak, düşük hacimli 09:00 slotunu ya gereğinden
#   fazla bastırır ya da yüksek hacimli 17:00 slotunu yeterince bastırmaz.
#
#   Çözüm: derive_gamma_from_costs()'a benzer bir yaklaşım — floor'u sabit
#   bir sayı yazmak yerine, o slotun KENDİ q50 dağılımından (örn. p25 ya da
#   p50) türetmek. Böylece her slot kendi "yapısal olarak düşük ama hâlâ
#   gerçek" hacim eşiğine göre kalibre olur, elle ayarlanan sihirli sayılara
#   ihtiyaç kalmaz.
def derive_risk_params_from_data(
    q50_values: List[float],
    floor_percentile: float = 25.0,
    min_nonzero_threshold: float = MIN_NONZERO_THRESHOLD,
    min_floor: Optional[float] = None,
    min_floor_frac_of_median: float = MIN_FLOOR_FRAC_OF_MEDIAN,
    absolute_min_floor: float = ABSOLUTE_MIN_FLOOR,
    fallback_floor: float = MATERIALITY_FLOOR,
    derive_tau: bool = False,
    q10_values: Optional[List[float]] = None,
    q90_values: Optional[List[float]] = None,
    tau_headroom: float = 0.0,
    tau_min_volume_percentile: float = TAU_MIN_VOLUME_PERCENTILE,
) -> Dict[str, float]:
    """
    Bir slotun (örn. sadece 09:00 ya da sadece 17:00) kendi q50 dağılımından
    materiality_floor'u — ve istenirse tau_base'i — türetir.

    [Düzeltme] Sıfır-hariç istatistik
    ---------------------------------
    Percentile (ve tau_base) hesabına giren q50 dağılımından, `min_nonzero_
    threshold`'un altındaki (durgun/talep-yok) satırlar DIŞLANIR —
    compute_target_skewness'in zaten uyguladığı "gözlem, sıfır hariç"
    mantığıyla tutarlı. Neden gerekli: durgun rotalarda q50 sık sık tam
    sıfıra ya da sıfıra çok yakın çıkıyor; bu satırlar ham dağılıma dahil
    edilirse, özellikle p25 gibi düşük bir persentil sıfıra doğru yapay
    şekilde çekilir ve floor, slotun GERÇEK/anlamlı hacimli rotalarını
    temsil etmeyen, yapay bir düşük sayıya düşer.

    materiality_floor
    ------------------
    Nonzero q50 alt kümesinin `floor_percentile`'ına (varsayılan: p25)
    eşittir. Sabit bir sayı yerine dağılımdan türetildiği için:
      - Yapısal olarak düşük hacimli bir slotta (örn. 09:00) floor da
        otomatik olarak düşük çıkar → o slotun rotaları gereksiz yere
        materyalite ağırlığıyla bastırılmaz.
      - Yapısal olarak yüksek hacimli bir slotta (örn. 17:00) floor da
        yükselir → gerçekten düşük kalan (görece durgun) rotalar hâlâ
        doğru şekilde bastırılır.
    Nonzero gözlem yoksa (tüm slot durgunsa) `fallback_floor`'a (varsayılan:
    modülün MATERIALITY_FLOOR sabiti) düşülür — asla None/0 dönmez,
    UncertaintyBand her zaman geçerli bir sayı alır.

    [Düzeltme] Dinamik/gerçekçi alt taban (min_floor)
    ---------------------------------------------------
    Eski davranışta min_floor=1.0 sabitti — 09:00 gibi düşük hacimli ama
    q50 ortalaması onlarca/yüzlerce olan bir slot için pratikte anlamsız
    bir tabandı (floor'un neredeyse hiç zemin oluşturmaması demekti).
    Artık:
      - `min_floor` elle (sabit bir sayı olarak) verilirse AYNEN kullanılır
        (örn. filo için "500 desiden az asla anlamlı değildir" gibi
        elle konmuş bir iş kuralınız varsa).
      - Verilmezse (None, varsayılan davranış) DİNAMİK hesaplanır: nonzero
        q50 medyanının `min_floor_frac_of_median` kesri (varsayılan: %10),
        ama her durumda `absolute_min_floor`'un (varsayılan: 50.0 — eski
        1.0'dan çok daha gerçekçi bir emniyet tabanı) altına inmez.

    tau_base (opsiyonel, derive_tau=True)
    --------------------------------------
    O slotun kendi gözlemlenen U_rel dağılımının MEDYANINA (+ isteğe bağlı
    bir `tau_headroom` payı) eşitlenir — Tur 3'teki "modelin gerçek U_rel
    tabanına kalibre et" mantığının slot-bazlı hâli.

    [Düzeltme] Neden mean değil median?
    Ortalama (mean), birkaç uç değerden (örn. q50≈1-5 ama q90≈40-60 gibi
    satırlardan — küçük mutlak farklar bile küçük paydada devasa U_rel
    üretir) kolayca şişer; tau_base tüm dağılımı temsil etmesi gereken bir
    "taban" olduğu için, birkaç uç satırın onu yukarı çekmesi istenmez.
    Medyan bu tür uç değerlere karşı çok daha dayanıklıdır.

    [Düzeltme] Ek hacim filtresi (tau_min_volume_percentile)
    Sıfır-hariç mantığı (min_nonzero_threshold) sadece tam sıfıra yakın
    satırları eler; q50=2-5 gibi "teknik olarak nonzero ama slotun kendi
    ölçeğine göre anlamsız derecede küçük" satırlar hâlâ dahil olurdu. Bu
    satırlarda küçük mutlak sapmalar bile U_rel'i devasa şişirebiliyor.
    Bu yüzden tau_base hesabına giren alt küme, floor hesabından daha SIKI
    bir eşikle filtrelenir: q50, nonzero q50 dağılımının kendi
    `tau_min_volume_percentile`'ının (varsayılan: p10) ÜZERİNDE olmalı.
    Bu, floor hesabının (`floor_percentile`, varsayılan p25) kullandığı
    kümeden farklı/daha dar bir alt kümedir — kasıtlı: tau_base'in aşırı
    düşük hacimli kuyruğa karşı floor'dan da hassas olması istenir.

    q10_values/q90_values verilmezse tau_base hiç hesaplanmaz (params
    sözlüğünde yer almaz) — bu, ikincil/kaba bir kaldıraçtır, önce
    materiality_floor düzeltmesi denenmeli (bkz. modül üstü PDF notları).

    Parameters
    ----------
    q50_values      : O slota ait ham tahminlerin q50 listesi.
    floor_percentile: materiality_floor için kullanılacak persentil (0-100),
                      nonzero alt küme üzerinden hesaplanır.
    min_nonzero_threshold: Bu eşiğin altındaki/eşit q50 değerleri "durgun"
                      sayılır ve percentile/tau istatistiklerinden dışlanır.
    min_floor       : Elle sabit bir alt taban. None ise dinamik hesaplanır
                      (bkz. yukarıdaki "Dinamik/gerçekçi alt taban" notu).
    min_floor_frac_of_median: min_floor=None olduğunda kullanılan oran.
    absolute_min_floor: Dinamik min_floor hesaplanırken asla altına
                      inilmeyecek mutlak taban.
    fallback_floor  : Nonzero q50 gözlemi yoksa kullanılacak varsayılan.
    derive_tau      : True ise tau_base de q10/q90'dan türetilir.
    q10_values, q90_values : derive_tau=True olduğunda gerekli, q50_values
                      ile AYNI uzunlukta ve aynı sırada olmalı (satır bazlı
                      hizalama için — indeks kayması olursa tau_base yanlış
                      hesaplanır).
    tau_headroom    : Türetilen tau_base'e eklenecek sabit pay (gevşetme).
    tau_min_volume_percentile: tau_base hesabına giren satırlar için ek
                      alt eşik — q50, nonzero q50 dağılımının bu
                      persentilinin (varsayılan: p10) üzerinde olmalı.
                      0 verilirse bu ek filtre devre dışı kalır (yalnızca
                      min_nonzero_threshold uygulanır).

    Returns
    -------
    Dict[str, float]
        En az {"materiality_floor": ...} içerir; derive_tau=True ve
        q10/q90 verilmişse ayrıca {"tau_base": ..., "maape_scale": ...} de
        içerir. [Tur 7] "maape_scale", bu alt kümenin tipik (medyan) ham
        U_rel oranıdır — UncertaintyBand(maape_scale=...) parametresine
        aktarılmalıdır (bkz. _maape_scale). Aktarılmazsa DEFAULT_MAAPE_SCALE
        (1.0) kullanılır ve tau_base yeniden tavana yapışabilir.
    """
    # None → NaN'a çevirip finite/nonzero maskelerini q50/q10/q90 arasında
    # HİZALI tutuyoruz (eski sürümde arr, q10_arr/q90_arr'dan bağımsız
    # filtrelendiği için None/NaN durumunda satır kayması riski vardı).
    q50_arr = np.asarray(
        [np.nan if v is None else v for v in q50_values], dtype=float
    )
    finite_mask = np.isfinite(q50_arr)

    # [Düzeltme] Sıfır/neredeyse-sıfır (durgun) gözlemleri dışla
    nonzero_mask = finite_mask & (q50_arr > min_nonzero_threshold)
    nonzero_q50 = q50_arr[nonzero_mask]

    if nonzero_q50.size == 0:
        floor = fallback_floor
        floor_min = absolute_min_floor if min_floor is None else min_floor
    else:
        floor = float(np.percentile(nonzero_q50, floor_percentile))
        if min_floor is not None:
            floor_min = min_floor
        else:
            floor_min = max(
                absolute_min_floor,
                min_floor_frac_of_median * float(np.median(nonzero_q50)),
            )

    floor = max(floor, floor_min)
    params: Dict[str, float] = {"materiality_floor": round(floor, 4)}

    if derive_tau and q10_values is not None and q90_values is not None:
        q10_arr = np.asarray(
            [np.nan if v is None else v for v in q10_values], dtype=float
        )
        q90_arr = np.asarray(
            [np.nan if v is None else v for v in q90_values], dtype=float
        )
        if q10_arr.size == q50_arr.size == q90_arr.size:
            # [Düzeltme] Ek hacim filtresi: floor hesabından daha SIKI bir
            # alt eşik — q50=2-5 gibi "teknik olarak nonzero ama slotun
            # kendi ölçeğine göre anlamsız derecede küçük" satırlar tau
            # hesabından da dışlansın. Eşik = nonzero q50 dağılımının
            # kendi p{tau_min_volume_percentile}'ı (varsayılan p10).
            if tau_min_volume_percentile and nonzero_q50.size:
                tau_volume_threshold = max(
                    min_nonzero_threshold,
                    float(np.percentile(nonzero_q50, tau_min_volume_percentile)),
                )
            else:
                tau_volume_threshold = min_nonzero_threshold

            tau_mask = (
                finite_mask
                & np.isfinite(q10_arr)
                & np.isfinite(q90_arr)
                & (q50_arr > tau_volume_threshold)
            )
            if tau_mask.any():
                v_safe = np.maximum(q50_arr[tau_mask], 1.0)
                urel_raw = (q90_arr[tau_mask] - q10_arr[tau_mask]) / v_safe
                urel_raw = urel_raw[np.isfinite(urel_raw)]
                if urel_raw.size:
                    # [Tur 7] KRİTİK DÜZELTME (gerçek üretim çalıştırmasında
                    # gözlemlendi — bkz. run_forecast.py log: 09:00 slotu için
                    # tau_base=0.872 çıktı, HIGH sayısı 0'a düştü): scale=1.0
                    # ile düz arctan kullanmak, seyrek/durgun bir slotta (09:00,
                    # çarpıklık +4.76) TİPİK ham oranı bile doğrudan tavana
                    # (1.0) yakın bir yere taşıyordu — outlier'lara hiç pay
                    # kalmıyordu. Çözüm: scale'i, bu alt kümenin KENDİ tipik
                    # (medyan) ham oranına eşitle. Böylece tanım gereği medyan
                    # satır 0.5'e denk gelir, tau_base tüm slotlarda taşınabilir/
                    # sabit kalabilir (bkz. DYNAMIC_TAU_BASE modül üstü notu),
                    # ve gerçek outlier'lar (medyanın kat kat üstü) hâlâ 1.0'a
                    # doğru ilerleyip HIGH tetikleyebilir.
                    scale_c = float(np.median(urel_raw))
                    if scale_c <= 0.0:
                        scale_c = 1.0  # degenerate durum (tüm gözlemler sıfır) — güvenli varsayılan
                    params["maape_scale"] = round(scale_c, 6)

                    # [Tur 6] KRİTİK: tau_base, runtime'da _compute_dynamic_risk'in
                    # ürettiği relative_uncertainty (MAAPE/arctan ile [0,1)'e
                    # sınırlanmış, AYNI scale ile) ile AYNI ölçekte olmalı. Burada
                    # hâlâ ham/sınırsız urel_raw üzerinden medyan alıp tau_base'e
                    # yazsaydık, runtime'daki bounded metrik ile unbounded'dan
                    # türetilmiş bir eşiği kıyaslamış olurduk. Bu yüzden medyan
                    # alınmadan ÖNCE her gözlem _maape_scale'den (scale_c ile)
                    # geçirilir — tanım gereği bu medyan ≈ 0.5 çıkar (+ tau_headroom).
                    urel_bounded = np.array([_maape_scale(x, scale_c) for x in urel_raw])
                    # [Düzeltme] mean → median: birkaç uç değerden (örn.
                    # q50≈1-5 ama q90≈40-60 gibi satırlardan) kolayca
                    # şişen ortalama yerine, aşırı değerlere karşı çok
                    # daha dayanıklı olan medyan kullanılır.
                    params["tau_base"] = round(float(np.median(urel_bounded)) + tau_headroom, 4)

    return params


# ---------------------------------------------------------------------------
# Dataclass: Tek Bir Satırın Belirsizlik Bandı
# ---------------------------------------------------------------------------

@dataclass
class DemandBand:
    """
    Tek bir (tarih, TM_ID) çifti için kantil bant verisi.

    Attributes
    ----------
    tarih            : Tahmin tarihi (YYYY-MM-DD)
    tm_id            : Transfer Merkezi kimliği
    slot             : Saat dilimi ("09:00" / "17:00") — DemandForecaster.predict()'ten
                       gelen "slot" alanını taşır. Sadece kimlik/etiketleme amaçlıdır,
                       risk hesaplama formüllerini (_compute_dynamic_risk) etkilemez —
                       q10/q50/q90 zaten slot-spesifik geldiği için matematik aynı kalır.
    q10              : Düşük senaryo (alt güven sınırı)
    q50              : Medyan tahmin (operasyonel plan)
    q90              : Yüksek senaryo (spot araç alarm seviyesi)
    uncertainty_range   : q90 - q10 (toplam belirsizlik genişliği)
    relative_uncertainty_raw: [Tur 6] Ham/sınırsız oran = uncertainty_range / max(q50, 1.0).
                       SADECE tanı/denetim amaçlı — sigmoid/τ karşılaştırmasında KULLANILMAZ,
                       çünkü q50→0 iken sınırsız büyüyebilir (bkz. relative_uncertainty).
    relative_uncertainty: [Tur 6/7] U_rel = MAAPE/Arktanjant ile [0,1) aralığına sınırlanmış nihai
                       oran = (2/π)·arctan(relative_uncertainty_raw / scale). q50 sıfıra yaklaştığında
                       artık PATLAMAZ, 1.0'a asimptotik olarak yaklaşır. [Tur 7] `scale`, o slotun/alt
                       kümenin TİPİK (medyan) ham oranıdır — bkz. derive_risk_params_from_data,
                       "maape_scale". Verilmezse (scale=1.0) Tur 6'daki eski davranışla aynıdır.
    dynamic_threshold   : τ(V) = τ_base + κ·V^(-β)  (hacme özel kabul edilebilir U_rel eşiği,
                       artık relative_uncertainty ile AYNI [0,1) ölçeğinde kalibre edilmelidir)
    dynamic_steepness   : k(V) = k_min + γ·log(1+V) (hacme özel sigmoid eğri katılığı)
    risk_score_raw      : [Tur 5] Materyalite ağırlığından ÖNCEKİ ham sigmoid skoru (tanı amaçlı)
    risk_score          : [Tur 5] Materyalite ağırlıklı NİHAİ skor, 0.0 (kesin LOW) - 1.0 (kesin HIGH)
    safety_buffer    : (q90 - q50) × buffer_ratio
    risk_class       : LOW / MEDIUM / HIGH (nihai risk_score'dan türetilir)
    recommended_qty  : ALNS'e önerilen kapasite rezervasyonu
    """
    tarih:             str
    tm_id:             Optional[str]
    q10:               float
    q50:               float
    q90:               float
    uncertainty_range: float   = field(init=False)
    safety_buffer:     float   = field(init=False)
    risk_class:        str     = field(init=False)
    recommended_qty:   float   = field(init=False)

    # --- Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli ---
    relative_uncertainty_raw: float = field(init=False)  # [Tur 6] ham/sınırsız oran, tanı amaçlı
    relative_uncertainty: float = field(init=False)  # [Tur 6] U_rel = (2/π)·arctan(ham oran), [0,1)
    dynamic_threshold:    float = field(init=False)  # τ(V)
    dynamic_steepness:    float = field(init=False)  # k(V)
    risk_score_raw:       float = field(init=False)  # [Tur 5] materyalite öncesi ham skor
    risk_score:           float = field(init=False)  # [Tur 5] materyalite ağırlıklı nihai skor

    # --- Saat dilimi (09:00 / 17:00) — sadece kimlik/etiketleme amaçlı ---
    slot: Optional[str] = None

    # --- [PDF: Gelişmiş Çözüm Aşaması] Talep ID — zorunlu çıktı formatı ---
    # PDF: "Kendi talep tahminlerinizi bize gönderirken de benzer şekilde her
    # talep için talep ID oluşturmanızı bekliyoruz. Talep ID formatı:
    # D00001, D00002, ..." Sıralı ID, UncertaintyBand.from_json() içinde
    # bands_ listesi oluşturulduktan sonra atanır (bkz. _assign_talep_ids()).
    # Burada sadece None varsayılanla alan tanımlanıyor; DemandBand tek
    # başına örneklendiğinde (örn. testlerde) talep_id boş kalabilir —
    # zorunluluk yalnızca UncertaintyBand üzerinden üretilen payload'larda.
    talep_id: Optional[str] = None

    # buffer_ratio dataclass'a init parametresi olarak almıyoruz
    # (asdict() serileştirmesini karmaşıklaştırır); __post_init__'e geçiyoruz
    _buffer_ratio: float = field(default=DEFAULT_BUFFER_RATIO, repr=False)

    # --- Dinamik Sigmoid Risk Modeli hiperparametreleri ---
    # UncertaintyBand seviyesinde set edilir, her DemandBand'e aktarılır.
    _tau_base: float = field(default=DYNAMIC_TAU_BASE, repr=False)
    _kappa:    float = field(default=DYNAMIC_KAPPA, repr=False)
    _beta:     float = field(default=DYNAMIC_BETA, repr=False)
    _k_min:    float = field(default=DYNAMIC_K_MIN, repr=False)
    _gamma:    float = field(default=DYNAMIC_GAMMA, repr=False)

    # --- [Tur 5] Materyalite ağırlığı tabanı ---
    _materiality_floor: float = field(default=MATERIALITY_FLOOR, repr=False)

    # --- [Tur 7] Slot-bazlı ölçek (scale) — MAAPE/arctan dönüşümünün referans noktası ---
    _maape_scale_c: float = field(default=DEFAULT_MAAPE_SCALE, repr=False)

    def __post_init__(self):
        # Negatif değer koruması
        self.q10 = max(self.q10, 0.0)
        self.q50 = max(self.q50, 0.0)
        self.q90 = max(self.q90, 0.0)

        # Monotonluk garantisi: q10 ≤ q50 ≤ q90
        self.q10 = min(self.q10, self.q50)
        self.q90 = max(self.q90, self.q50)

        # Türetilmiş alanlar
        self.uncertainty_range = round(self.q90 - self.q10, 4)

        # Güvenlik tamponu: (q90 - q50) × buffer_ratio
        # ALNS bunu "minimum rezerve kapasite" olarak kullanır
        self.safety_buffer = round((self.q90 - self.q50) * self._buffer_ratio, 4)

        # Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli + Materyalite Ağırlığı (Tur 5)
        self._compute_dynamic_risk()

        # ALNS'e önerilen rezervasyon = q50 + safety_buffer
        self.recommended_qty = round(self.q50 + self.safety_buffer, 4)

    def _compute_dynamic_risk(self) -> None:
        """
        PDF: "Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli" + [Tur 5] Materyalite Ağırlığı

        1. U_rel             = (q90 - q10) / V_safe
        2. τ(V)  = τ_base + κ · V_safe^(-β)        ← dinamik eşik
        3. k(V)  = k_min + γ · log(1 + V_safe)     ← dinamik katılık
        4. Risk_Score_raw = 1 / (1 + exp(-k(V) · (U_rel - τ(V))))
        5. [Tur 5] weight(V)     = min(1.0, q50 / materiality_floor)
           Risk_Score_final = Risk_Score_raw × sqrt(weight(V))
        6. Risk_Score_final → LOW / MEDIUM / HIGH

        V_safe = max(q50, 1.0)  → sıfır/çok küçük hacimlerde bölme hatası
        önler. Ancak V_safe küçükse U_rel matematiksel olarak patlayabilir
        (örn. q50=0, uncertainty_range=125 → U_rel=125). Bu artık adım 5'teki
        materyalite ağırlığıyla dengelenir: gerçek hacim (q50, floor'a göre
        DEĞİL 1.0'a göre ölçülür) küçükse, ham skor ne kadar yüksek olursa
        olsun nihai skor da orantılı şekilde küçültülür.
        """
        v_safe = max(self.q50, 1.0)

        # 1a. Ham/sınırsız oran — SADECE tanı amaçlı, payload'a diagnostic alan
        #     olarak eklenir. q50→0 iken sınırsız büyüyebilir (örn. %3300);
        #     bu yüzden aşağıdaki adım 1b'den ÖNCE, sigmoid/τ karşılaştırmasına
        #     hiç girmeden, sadece gözlemlenebilirlik için saklanır.
        raw_ratio = self.uncertainty_range / v_safe
        self.relative_uncertainty_raw = round(raw_ratio, 4)

        # 1b. [Tur 6] MAAPE/Arktanjant dönüşümü — nihai U_rel, [0,1) ile SINIRLI.
        #     f(x) = (2/π)·arctan(x): q50→0, uncertainty_range sabit kalsa bile
        #     artık patlamaz; 1.0'a asimptotik yaklaşır (bkz. _maape_scale).
        self.relative_uncertainty = round(_maape_scale(raw_ratio, self._maape_scale_c), 4)

        # 2. Dinamik eşik: τ(V) = τ_base + κ · V^(-β)
        self.dynamic_threshold = round(
            self._tau_base + self._kappa * (v_safe ** (-self._beta)), 4
        )

        # 3. Dinamik katılık: k(V) = k_min + γ · log(1 + V)
        self.dynamic_steepness = round(
            self._k_min + self._gamma * np.log1p(v_safe), 4
        )

        # 4. Sigmoid HAM risk skoru (overflow korumalı)
        exponent = -self.dynamic_steepness * (
            self.relative_uncertainty - self.dynamic_threshold
        )
        exponent = float(np.clip(exponent, -SIGMOID_EXP_CLIP, SIGMOID_EXP_CLIP))
        raw_score = 1.0 / (1.0 + np.exp(exponent))
        self.risk_score_raw = round(raw_score, 4)

        # 5. [Tur 5] Materyalite ağırlığı — düşük mutlak hacimde ham skoru bastır.
        #    Sert kesme değil, sürekli/orantılı sönümleme (0'dan 1'e yumuşak geçiş).
        floor = max(self._materiality_floor, 1e-6)
        materiality_weight = min(1.0, self.q50 / floor)
        # Tur 5:
        # Lineer bastırma yerine sqrt(weight) kullan.
        # Küçük hacimli rotalar tamamen LOW'a düşmesin,
        # fakat gereksiz HIGH üretimi azalsın.
        materiality_weight = np.sqrt(materiality_weight)
        self.risk_score = round(raw_score * materiality_weight, 4)

        # 6. Sürekli skoru operasyonel etikete çevir (nihai/ağırlıklı skor üzerinden)
        self.risk_class = self._classify_risk()

    def _classify_risk(self) -> str:
        """
        Sürekli, materyalite-ağırlıklı risk_score'u (0.0-1.0) operasyonel
        etikete çevirir (PDF Bölüm: "Sürekli Risk Skorlarının Operasyonel
        Etiketlere Çevrilmesi").

          LOW    : risk_score < 0.33  → Rutin planlama, spot araç gerekmez
          MEDIUM : 0.33 ≤ score ≤ 0.66 → İzleme listesi (watchlist), kontrol kulesi sarı uyarı
          HIGH   : risk_score > 0.66  → Otomatik müdahale sinyali, spot araç tedariği
        """
        if self.risk_score < RISK_SCORE_LOW_MAX:
            return "LOW"
        elif self.risk_score <= RISK_SCORE_MEDIUM_MAX:
            return "MEDIUM"
        else:
            return "HIGH"

    def to_dict(self) -> Dict[str, Any]:
        """ALNS payload formatına uygun sözlük döndürür."""
        return {
            "talep_id":             self.talep_id,   # [PDF] D00001, D00002, ...
            "tarih":                self.tarih,
            "TM_ID":                self.tm_id,
            "slot":                 self.slot,
            "demand_low":           self.q10,
            "demand_base":          self.q50,
            "demand_high":          self.q90,
            "uncertainty_range":    self.uncertainty_range,
            "relative_uncertainty_raw": self.relative_uncertainty_raw,  # [Tur 6] tanı amaçlı, sınırsız
            "relative_uncertainty": self.relative_uncertainty,  # [Tur 6] MAAPE-bounded, [0,1)
            "dynamic_threshold":    self.dynamic_threshold,
            "dynamic_steepness":    self.dynamic_steepness,
            "risk_score_raw":       self.risk_score_raw,   # [Tur 5] tanı amaçlı, ALNS okumak zorunda değil
            "risk_score":           self.risk_score,
            "safety_buffer":        self.safety_buffer,
            "risk_class":           self.risk_class,
            "recommended_qty":      self.recommended_qty,
        }


# ---------------------------------------------------------------------------
# UncertaintyBand: Toplu Dönüşüm ve Yönetim
# ---------------------------------------------------------------------------

class UncertaintyBand:
    """
    DemandForecaster.predict() çıktısını ALNS payload'ına dönüştürür.

    Parameters
    ----------
    buffer_ratio : Güvenlik tamponu katsayısı.
                   recommended_qty = q50 + (q90 - q50) × buffer_ratio
                   Varsayılan: 0.5 → q50 ile q90'ın tam ortası
    logging_enabled : Detaylı log. Varsayılan: True

    tau_base, kappa, beta, k_min, gamma :
        Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli hiperparametreleri.

    materiality_floor : [Tur 5] Bu hacmin altındaki q50 değerlerinde nihai
        risk_score, ham sigmoid skoruna oranla (q50/materiality_floor)
        bastırılır. Amaç: neredeyse durgun rotalarda küçük mutlak sapmaların
        payda küçüklüğü yüzünden yapay HIGH üretmesini engellemek.
        Varsayılan: MATERIALITY_FLOOR (750.0 desi, ≈ filo p10 hacmi).

    Examples
    --------
    >>> results = forecaster.predict(test_df)       # List[Dict]
    >>> band = UncertaintyBand(buffer_ratio=0.5)
    >>> payload = band.to_alns_payload(results)     # ALNS formatı
    >>> alns_engine.run(payload)
    """

    def __init__(
        self,
        buffer_ratio: float = DEFAULT_BUFFER_RATIO,
        logging_enabled: bool = True,
        tau_base: float = DYNAMIC_TAU_BASE,
        kappa: float = DYNAMIC_KAPPA,
        beta: float = DYNAMIC_BETA,
        k_min: float = DYNAMIC_K_MIN,
        gamma: float = DYNAMIC_GAMMA,
        materiality_floor: float = MATERIALITY_FLOOR,
        maape_scale: float = DEFAULT_MAAPE_SCALE,
    ):
        self.buffer_ratio    = buffer_ratio
        self.logging_enabled = logging_enabled

        # Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli hiperparametreleri
        self.tau_base = tau_base
        self.kappa    = kappa
        self.beta     = beta
        self.k_min    = k_min
        self.gamma    = gamma

        # [Tur 5] Materyalite ağırlığı
        self.materiality_floor = materiality_floor

        # [Tur 7] MAAPE/arctan ölçeği — bu slotun/alt kümenin tipik (medyan)
        # ham U_rel oranı. derive_risk_params_from_data(..., derive_tau=True)
        # çıktısındaki "maape_scale" buraya aktarılmalı; aksi halde
        # DEFAULT_MAAPE_SCALE (1.0) kullanılır ve tau_base seyrek/durgun
        # slotlarda tavana (1.0) yapışıp HIGH'ı imkânsız kılabilir (bkz.
        # modül üstü "Tur 7" notu).
        self.maape_scale = maape_scale

        self.bands_: List[DemandBand] = []

    def from_json(
        self,
        predictions: List[Dict[str, Any]],
        date_key:  str = "tarih",
        group_key: str = "TM_ID",
        slot_key:  str = "slot",
    ) -> "UncertaintyBand":
        """
        predict() çıktısını (List[Dict]) DemandBand listesine dönüştürür.

        Parameters
        ----------
        predictions : DemandForecaster.predict() çıktısı
        date_key    : Tarih sütunu anahtarı
        group_key   : TM_ID sütunu anahtarı
        slot_key    : Saat dilimi sütunu anahtarı (varsayılan: "slot") — date_key/
                      group_key ile aynı desen: ileride farklı bir slot alan adı
                      gelirse (örn. "talep_tamamlanma_saati") kod değişmeden çalışır.

        Returns
        -------
        self (method chaining için)
        """
        self.bands_ = []

        for rec in predictions:
            band = DemandBand(
                tarih=str(rec.get(date_key, "N/A")),
                tm_id=str(rec.get(group_key, "N/A")),
                slot=rec.get(slot_key, "N/A"),
                q10=float(rec.get("q10", 0.0)),
                q50=float(rec.get("q50", 0.0)),
                q90=float(rec.get("q90", 0.0)),
                _buffer_ratio=self.buffer_ratio,
                _tau_base=self.tau_base,
                _kappa=self.kappa,
                _beta=self.beta,
                _k_min=self.k_min,
                _gamma=self.gamma,
                _materiality_floor=self.materiality_floor,
                _maape_scale_c=self.maape_scale,
            )
            self.bands_.append(band)

        # [PDF: Gelişmiş Çözüm Aşaması] Talep ID ataması — D00001, D00002, ...
        # Sıralı, 1-index. Bir talebin sonradan birden fazla araca bölünmesi
        # durumunda (D00001-1, D00001-2) formatı PDF'te belirtiliyor, ancak
        # bölme kararı optimizasyon/rota planlama aşamasında verildiği için
        # burada YALNIZCA temel (bölünmemiş) talep ID'si üretilir — ALNS/
        # OR-Tools motoru gerekirse bu ID'nin sonuna "-1", "-2" ekleyerek
        # kendi bölme mantığını uygular.
        for _i, _band in enumerate(self.bands_, start=1):
            _band.talep_id = f"D{_i:05d}"

        if self.logging_enabled:
            self._log_summary()

        return self

    def to_alns_payload(
        self,
        predictions: Optional[List[Dict[str, Any]]] = None,
        date_key:  str = "tarih",
        group_key: str = "TM_ID",
        slot_key:  str = "slot",
    ) -> Dict[str, Any]:
        """
        ALNS motorunun tüketeceği nihai in-memory payload'ı üretir.

        Disk I/O YOK — direkt Dict olarak return edilir.

        Parameters
        ----------
        predictions : Opsiyonel. Verilirse from_json() otomatik çağrılır.
        date_key    : Tarih anahtarı
        group_key   : Grup anahtarı
        slot_key    : Saat dilimi anahtarı (varsayılan: "slot")

        Returns
        -------
        Dict[str, Any]
            {
              "metadata": { ..., "has_slot_dimension": True },
              "demands":  [ DemandBand.to_dict(), ... ],  ← her kayıt artık "slot" taşır
              "risk_summary": { "LOW": n, "MEDIUM": n, "HIGH": n },
              "risk_summary_by_slot": { "09:00": {...}, "17:00": {...} }
            }
        """
        if predictions is not None:
            self.from_json(predictions, date_key=date_key, group_key=group_key, slot_key=slot_key)

        if not self.bands_:
            raise ValueError(
                "❌ Bant verisi yok! Önce from_json() çağırın "
                "veya predictions parametresi geçin."
            )

        # Risk dağılımı özeti (tüm slotlar dahil, tek potada — genel özet için hâlâ anlamlı)
        risk_summary = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for b in self.bands_:
            risk_summary[b.risk_class] += 1

        # İsteğe bağlı: slot bazlı kırılım — OR-Tools/ALNS motorunun işini kolaylaştırır
        # (zorunlu değil, ama küçük bir ek — genel risk_summary'nin yerini almaz)
        risk_summary_by_slot: Dict[str, Dict[str, int]] = {}
        for b in self.bands_:
            slot_label = b.slot or "N/A"
            bucket = risk_summary_by_slot.setdefault(slot_label, {"LOW": 0, "MEDIUM": 0, "HIGH": 0})
            bucket[b.risk_class] += 1

        # Tarih aralığı
        dates = [b.tarih for b in self.bands_ if b.tarih != "N/A"]
        horizon_days = len(set(dates))

        payload: Dict[str, Any] = {
            "metadata": {
                "generated_at":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "n_records":     len(self.bands_),
                "horizon_days":  horizon_days,
                "buffer_ratio":  self.buffer_ratio,
                # ALNS bu bayrağı okuyarak high-risk satırlara öncelik verir
                "has_high_risk": risk_summary["HIGH"] > 0,
                # Şema notu: her "demands" kaydı artık bir "slot" alanı taşıyor
                # (09:00/17:00) — ALNS tarafındaki arkadaşlar payload'ı ilk
                # gördüğünde şemanın değiştiğini fark etsin diye.
                "has_slot_dimension": True,
                # Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli parametreleri
                # (her kaydın risk_score'u bu parametrelerle üretildi)
                "risk_model": {
                    "name":              "volume_weighted_dynamic_sigmoid",
                    "tau_base":          self.tau_base,
                    "kappa":             self.kappa,
                    "beta":              self.beta,
                    "k_min":             self.k_min,
                    "gamma":             self.gamma,
                    "materiality_floor": self.materiality_floor,  # [Tur 5]
                    "maape_scale": self.maape_scale,  # [Tur 7]
                    "materiality_function": "sqrt",  # [Tur 5]
                    "relative_uncertainty_scale": "maape_arctan_0_1",  # [Tur 6]
                },
            },
            "risk_summary": risk_summary,
            # İsteğe bağlı slot bazlı kırılım — zorunlu değil, genel risk_summary'nin
            # yerini almaz, sadece OR-Tools/ALNS motoruna kolaylık sağlar
            "risk_summary_by_slot": risk_summary_by_slot,
            # Ana veri: ALNS araç atama algoritmasına girdi
            "demands": [b.to_dict() for b in self.bands_],
        }

        if self.logging_enabled:
            logger.info(
                f"✅ ALNS payload hazır: {len(self.bands_)} talep kaydı\n"
                f"   Risk dağılımı → "
                f"LOW: {risk_summary['LOW']} | "
                f"MEDIUM: {risk_summary['MEDIUM']} | "
                f"HIGH: {risk_summary['HIGH']}"
            )

        return payload

    def to_ortools_dataframe(
        self,
        predictions: List[Dict[str, Any]],
        date_key: str = "tarih",
        group_key: str = "TM_ID",
        slot_key: str = "slot",
    ) -> "pd.DataFrame":
        """
        Tahminleri OR-Tools Optimizasyon motorunun kullanabileceği
        düz (flat) DataFrame formatına çevirir.

        TM_ID'yi "Kaynak" ve "Varış" düğümlerine böler; OR-Tools'un
        kenar tabanlı araç atama algoritması bu ayrımı zorunlu kılar.

        Çıktı sütunlar
        --------------
        talep_id            : [PDF] Sıralı talep kimliği (D00001, D00002, ...)
        date                : Tahmin tarihi
        slot                : Saat dilimi ("09:00" / "17:00") — ZORUNLU: bu olmadan
                               OR-Tools iki farklı saat dilimindeki talebi aynı
                               satırmış gibi işler (yanlış kapasite/SLA planlaması)
        source              : Kaynak Transfer Merkezi
        destination         : Varış Transfer Merkezi
        q10                 : Düşük senaryo
        q50                 : Medyan tahmin
        q90                 : Yüksek senaryo (spot araç alarm seviyesi)
        recommended_demand  : q50 + safety_buffer (OR-Tools kapasite girdisi)
        risk_class          : LOW / MEDIUM / HIGH
        risk_score          : Materyalite ağırlıklı nihai risk skoru
        risk_score_raw      : [Tur 5] tanı amaçlı ham risk skoru

        Parameters
        ----------
        predictions : DemandForecaster.predict() çıktısı (List[Dict])
        date_key    : Tarih anahtarı (varsayılan: "tarih")
        group_key   : Grup anahtarı (varsayılan: "TM_ID")
        slot_key    : Saat dilimi anahtarı (varsayılan: "slot")

        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd

        # 1. Tahminleri içeri al ve belirsizlik bantlarını/tamponları hesapla
        self.from_json(predictions, date_key=date_key, group_key=group_key, slot_key=slot_key)

        # 2. OR-Tools formatında listeyi hazırla
        records = []
        for b in self.bands_:
            # TM_ID'yi "Kaynak" ve "Varış" olarak ikiye böl (OR-Tools node'ları)
            # DemandBand'deki alan adları: b.tm_id ve b.tarih
            group_id = b.tm_id or "Bilinmiyor"
            source, dest = group_id, "Bilinmiyor"
            if " → " in group_id:
                source, dest = group_id.split(" → ", 1)
            elif " -> " in group_id:
                source, dest = group_id.split(" -> ", 1)
            elif "-" in group_id:
                source, dest = group_id.split("-", 1)

            # OR-Tools için net talep = Medyan Tahmin + Risk Tamponu
            recommended_demand = b.q50 + b.safety_buffer
            records.append({
                "talep_id":           b.talep_id,   # [PDF] D00001, D00002, ...
                "date":               b.tarih,
                "demand_start_time":               b.slot,       # ZORUNLU — bkz. docstring/madde 4 notu
                "source":             source.strip(),
                "destination":        dest.strip(),
                "q10":                round(b.q10, 2),
                "q50":                round(b.q50, 2),
                "q90":                round(b.q90, 2),
                "recommended_demand": round(recommended_demand, 2),
                "risk_class":         b.risk_class,
                "risk_score":         b.risk_score,
                "risk_score_raw":     b.risk_score_raw,  # [Tur 5] tanı amaçlı
            })

        df_ortools = pd.DataFrame(records)

        if self.logging_enabled:
            logger.info(
                f"⚙️  OR-Tools payload'u hazırlandı: {len(df_ortools)} satır, "
                f"{len(df_ortools.columns)} sütun (talep_id, date, slot, source, destination, q10, q50, q90, "
                f"recommended_demand, risk_class, risk_score, risk_score_raw)."
            )

        return df_ortools

    def get_high_risk_records(self) -> List[Dict[str, Any]]:
        """
        Yalnızca HIGH riskli kayıtları döndürür.

        ALNS motoru önce bu kayıtlara araç atayarak
        spot araç riskini minimize eder.
        """
        if not self.bands_:
            raise ValueError("❌ Önce from_json() veya to_alns_payload() çağırın!")

        return [b.to_dict() for b in self.bands_ if b.risk_class == "HIGH"]

    def _log_summary(self) -> None:
        """Bant istatistiklerini loglar."""
        if not self.bands_:
            return

        q50_vals  = np.array([b.q50 for b in self.bands_])
        weights = np.sqrt(
            np.minimum(
                1.0,
                q50_vals / max(self.materiality_floor, 1e-6)
            )
        )
        unc_vals  = np.array([b.uncertainty_range for b in self.bands_])
        urel_vals = np.array([b.relative_uncertainty for b in self.bands_])
        score_raw_vals = np.array([b.risk_score_raw for b in self.bands_])
        score_vals = np.array([b.risk_score for b in self.bands_])
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for b in self.bands_:
            risk_counts[b.risk_class] += 1

        # [Tur 5] Kaç kayıt materyalite ağırlığıyla bastırıldı (raw HIGH ama final değil)?
        dampened = sum(
            1 for b in self.bands_
            if b.risk_score_raw > RISK_SCORE_MEDIUM_MAX and b.risk_class != "HIGH"
        )

        logger.info(
            f"\n📐 UncertaintyBand Özeti ({len(self.bands_)} kayıt) "
            f"[Hacim Ağırlıklı Dinamik Sigmoid + Materyalite Ağırlığı — Tur 5]\n"
            f"   q50 ort/max        : {q50_vals.mean():.1f} / {q50_vals.max():.1f}\n"
            f"   Belirsizlik ort    : {unc_vals.mean():.1f}\n"
            f"   U_rel ort          : {urel_vals.mean():.3f}\n"
            f"   risk_score_raw ort : {score_raw_vals.mean():.3f}\n"
            f"   risk_score (final) : {score_vals.mean():.3f}\n"
            f"   Materiality weight : "
            f"ort={weights.mean():.3f} | "
            f"medyan={np.median(weights):.3f} | "
            f"min={weights.min():.3f}\n"
            f"   Materyalite ile bastırılan (ham HIGH → final≠HIGH): {dampened} kayıt\n"
            f"   Risk dağılımı      → "
            f"LOW: {risk_counts['LOW']} | "
            f"MEDIUM: {risk_counts['MEDIUM']} | "
            f"HIGH: {risk_counts['HIGH']}"
        )

        # --- İsteğe bağlı: slot bazlı ayrıştırılmış özet ---------------------
        # 09:00 ve 17:00'nin hacim dağılımları çok farklı olabiliyor (örn.
        # 09:00 ort. 517, 17:00 ort. 3225 — ~6 kat fark). Tek bir genel
        # ortalamada bu ikisi karışırsa yanıltıcı olur; birden fazla slot
        # varsa her biri için ayrı bir mini özet de basılır (teşhis amaçlı).
        distinct_slots = sorted({b.slot for b in self.bands_ if b.slot not in (None, "N/A")})
        if len(distinct_slots) > 1:
            for slot_label in distinct_slots:
                slot_bands = [b for b in self.bands_ if b.slot == slot_label]
                if not slot_bands:
                    continue
                s_q50  = np.array([b.q50 for b in slot_bands])
                s_urel = np.array([b.relative_uncertainty for b in slot_bands])
                s_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
                for b in slot_bands:
                    s_counts[b.risk_class] += 1
                logger.info(
                    f"   ── [{slot_label}] {len(slot_bands)} kayıt → "
                    f"q50 ort/max: {s_q50.mean():.1f}/{s_q50.max():.1f} | "
                    f"U_rel ort: {s_urel.mean():.3f} | "
                    f"Risk → LOW: {s_counts['LOW']} | MEDIUM: {s_counts['MEDIUM']} | HIGH: {s_counts['HIGH']}"
                )


# ---------------------------------------------------------------------------
# combine_slot_bands() — Ayrı Slot Parametreleriyle İşlenen Bantları Birleştir
# ---------------------------------------------------------------------------
#
# NEDEN GEREKLİ:
#   to_ortools_dataframe() / to_alns_payload() TEK bir UncertaintyBand
#   örneği üzerinden çalışır → tek bir parametre setine (materiality_floor,
#   tau_base, ...) mahkûmdur. 09:00 ve 17:00'yi kendi (derive_risk_params_
#   from_data() ile türetilmiş) parametreleriyle ayrı ayrı işlemek için iki
#   ayrı UncertaintyBand örneği gerekir — bu fonksiyon ikisinin çıktısını
#   TEK bir OR-Tools DataFrame'i ve TEK bir ALNS payload'ında birleştirir.
#
#   talep_id'ler: her bandın kendi from_json() çağrısı D00001'den başlar
#   (bkz. from_json altındaki "_assign_talep_ids" notu) — yani iki bandı
#   olduğu gibi yan yana koyarsak D00001 iki kez üretilir. Bu fonksiyon,
#   birleştirdikten SONRA (tarih, slot, TM_ID) sırasına göre talep_id'leri
#   YENİDEN ve sıralı atar; PDF'in zorunlu kıldığı "her talep için tekil
#   D0000N kimliği" garantisi böylece slot sayısından bağımsız korunur.
def combine_slot_bands(
    bands: List["UncertaintyBand"],
) -> Dict[str, Any]:
    """
    Her biri kendi (muhtemelen slot-bazlı türetilmiş) parametreleriyle
    from_json() çağrılmış birden fazla UncertaintyBand örneğini TEK bir
    OR-Tools DataFrame'i + TEK bir ALNS payload'ında birleştirir.

    Parameters
    ----------
    bands : List[UncertaintyBand]
        Her biri için önce from_json() (veya to_ortools_dataframe/
        to_alns_payload) çağrılmış olmalı — yani `band.bands_` dolu olmalı.
        Tipik kullanım: 09:00 için bir UncertaintyBand, 17:00 için bir
        diğeri, her biri kendi materiality_floor/tau_base'iyle.

    Returns
    -------
    Dict[str, Any]
        {
          "dataframe": pd.DataFrame,   ← to_ortools_dataframe ile aynı şema
          "payload":   Dict[str, Any], ← to_alns_payload ile aynı şema,
                                          ek olarak metadata.risk_model_by_slot
        }

    Examples
    --------
    >>> band_0900 = UncertaintyBand(materiality_floor=params_0900["materiality_floor"])
    >>> band_0900.from_json(preds_0900, slot_key="slot")
    >>> band_1700 = UncertaintyBand(materiality_floor=params_1700["materiality_floor"])
    >>> band_1700.from_json(preds_1700, slot_key="slot")
    >>> result = combine_slot_bands([band_0900, band_1700])
    >>> df_ortools = result["dataframe"]
    >>> alns_payload = result["payload"]
    """
    import pandas as pd

    if not bands:
        raise ValueError("❌ En az bir UncertaintyBand geçilmeli.")

    all_dbands: List[DemandBand] = []
    risk_model_by_slot: Dict[str, Dict[str, Any]] = {}

    for band in bands:
        if not band.bands_:
            raise ValueError(
                "❌ Her UncertaintyBand için önce from_json() (ya da "
                "to_ortools_dataframe/to_alns_payload) çağrılmalı."
            )
        all_dbands.extend(band.bands_)
        for b in band.bands_:
            slot_label = b.slot or "N/A"
            # Aynı slot birden fazla bandda geçerse ilkini koru (çakışma
            # olmaması gerekir ama teşhis kolaylığı için sessizce üzerine yazma)
            risk_model_by_slot.setdefault(slot_label, {
                "name":                "volume_weighted_dynamic_sigmoid",
                "tau_base":            band.tau_base,
                "kappa":               band.kappa,
                "beta":                band.beta,
                "k_min":               band.k_min,
                "gamma":               band.gamma,
                "materiality_floor":   band.materiality_floor,  # slot-bazlı türetilmiş
                "maape_scale":          band.maape_scale,  # [Tur 7] slot-bazlı türetilmiş
                "materiality_function": "sqrt",
                "relative_uncertainty_scale": "maape_arctan_0_1",  # [Tur 6]
            })

    # (tarih, slot, TM_ID) sırasına göre kararlı sıralama — SONRA talep_id ata
    all_dbands.sort(key=lambda b: (b.tarih, b.slot or "", b.tm_id or ""))
    for _i, _band in enumerate(all_dbands, start=1):
        _band.talep_id = f"D{_i:05d}"

    # --- OR-Tools DataFrame (to_ortools_dataframe ile birebir aynı şema) ---
    records = []
    for b in all_dbands:
        group_id = b.tm_id or "Bilinmiyor"
        source, dest = group_id, "Bilinmiyor"
        if " → " in group_id:
            source, dest = group_id.split(" → ", 1)
        elif " -> " in group_id:
            source, dest = group_id.split(" -> ", 1)
        elif "-" in group_id:
            source, dest = group_id.split("-", 1)

        recommended_demand = b.q50 + b.safety_buffer
        records.append({
            "talep_id":           b.talep_id,
            "date":               b.tarih,
            "demand_start_time":  b.slot,
            "source":             source.strip(),
            "destination":        dest.strip(),
            "q10":                round(b.q10, 2),
            "q50":                round(b.q50, 2),
            "q90":                round(b.q90, 2),
            "recommended_demand": round(recommended_demand, 2),
            "risk_class":         b.risk_class,
            "risk_score":         b.risk_score,
            "risk_score_raw":     b.risk_score_raw,
        })
    df_ortools = pd.DataFrame(records)

    # --- ALNS Payload (to_alns_payload ile birebir aynı şema + risk_model_by_slot) ---
    risk_summary = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    risk_summary_by_slot: Dict[str, Dict[str, int]] = {}
    for b in all_dbands:
        risk_summary[b.risk_class] += 1
        slot_label = b.slot or "N/A"
        bucket = risk_summary_by_slot.setdefault(slot_label, {"LOW": 0, "MEDIUM": 0, "HIGH": 0})
        bucket[b.risk_class] += 1

    dates = [b.tarih for b in all_dbands if b.tarih != "N/A"]
    horizon_days = len(set(dates))

    payload: Dict[str, Any] = {
        "metadata": {
            "generated_at":       datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_records":          len(all_dbands),
            "horizon_days":       horizon_days,
            "has_high_risk":      risk_summary["HIGH"] > 0,
            "has_slot_dimension": True,
            # [Slot-bazlı türetim] Artık TEK bir risk_model yerine, her
            # slotun kendi (derive_risk_params_from_data ile türetilmiş)
            # parametre setini ayrı ayrı raporluyoruz.
            "risk_model_by_slot": risk_model_by_slot,
        },
        "risk_summary":          risk_summary,
        "risk_summary_by_slot":  risk_summary_by_slot,
        "demands":               [b.to_dict() for b in all_dbands],
    }

    return {"dataframe": df_ortools, "payload": payload}