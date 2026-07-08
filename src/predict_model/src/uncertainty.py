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

# τ(V) = τ_base + κ · V^(-β)  → hacme göre maks. kabul edilebilir U_rel eşiği
DYNAMIC_TAU_BASE: float = 0.50   # Asimptotik taban eşik — modelin doğal U_rel tabanına (~%55) yakın
DYNAMIC_KAPPA:    float = 5.0    # Düşük/orta hacim gevşeme çarpanı
DYNAMIC_BETA:     float = 0.30   # Sönümleme oranı (Taylor güç yasası; bu filo ölçeğine göre kalibre)

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
    q10              : Düşük senaryo (alt güven sınırı)
    q50              : Medyan tahmin (operasyonel plan)
    q90              : Yüksek senaryo (spot araç alarm seviyesi)
    uncertainty_range   : q90 - q10 (toplam belirsizlik genişliği)
    relative_uncertainty: U_rel = uncertainty_range / max(q50, 1.0)
    dynamic_threshold   : τ(V) = τ_base + κ·V^(-β)  (hacme özel kabul edilebilir U_rel eşiği)
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
    relative_uncertainty: float = field(init=False)  # U_rel = (q90-q10)/q50
    dynamic_threshold:    float = field(init=False)  # τ(V)
    dynamic_steepness:    float = field(init=False)  # k(V)
    risk_score_raw:       float = field(init=False)  # [Tur 5] materyalite öncesi ham skor
    risk_score:           float = field(init=False)  # [Tur 5] materyalite ağırlıklı nihai skor

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

        # 1. Göreceli belirsizlik (U_rel) — ham/tanısal, floor'dan etkilenmez
        self.relative_uncertainty = round(self.uncertainty_range / v_safe, 4)

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
            "tarih":                self.tarih,
            "TM_ID":                self.tm_id,
            "demand_low":           self.q10,
            "demand_base":          self.q50,
            "demand_high":          self.q90,
            "uncertainty_range":    self.uncertainty_range,
            "relative_uncertainty": self.relative_uncertainty,
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

        self.bands_: List[DemandBand] = []

    def from_json(
        self,
        predictions: List[Dict[str, Any]],
        date_key:  str = "tarih",
        group_key: str = "TM_ID",
    ) -> "UncertaintyBand":
        """
        predict() çıktısını (List[Dict]) DemandBand listesine dönüştürür.

        Parameters
        ----------
        predictions : DemandForecaster.predict() çıktısı
        date_key    : Tarih sütunu anahtarı
        group_key   : TM_ID sütunu anahtarı

        Returns
        -------
        self (method chaining için)
        """
        self.bands_ = []

        for rec in predictions:
            band = DemandBand(
                tarih=str(rec.get(date_key, "N/A")),
                tm_id=str(rec.get(group_key, "N/A")),
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
            )
            self.bands_.append(band)

        if self.logging_enabled:
            self._log_summary()

        return self

    def to_alns_payload(
        self,
        predictions: Optional[List[Dict[str, Any]]] = None,
        date_key:  str = "tarih",
        group_key: str = "TM_ID",
    ) -> Dict[str, Any]:
        """
        ALNS motorunun tüketeceği nihai in-memory payload'ı üretir.

        Disk I/O YOK — direkt Dict olarak return edilir.

        Parameters
        ----------
        predictions : Opsiyonel. Verilirse from_json() otomatik çağrılır.
        date_key    : Tarih anahtarı
        group_key   : Grup anahtarı

        Returns
        -------
        Dict[str, Any]
            {
              "metadata": { ... },
              "demands":  [ DemandBand.to_dict(), ... ],
              "risk_summary": { "LOW": n, "MEDIUM": n, "HIGH": n }
            }
        """
        if predictions is not None:
            self.from_json(predictions, date_key=date_key, group_key=group_key)

        if not self.bands_:
            raise ValueError(
                "❌ Bant verisi yok! Önce from_json() çağırın "
                "veya predictions parametresi geçin."
            )

        # Risk dağılımı özeti
        risk_summary = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for b in self.bands_:
            risk_summary[b.risk_class] += 1

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
                    "materiality_function": "sqrt",  # [Tur 5]
                },
            },
            "risk_summary": risk_summary,
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
    ) -> "pd.DataFrame":
        """
        Tahminleri OR-Tools Optimizasyon motorunun kullanabileceği
        düz (flat) DataFrame formatına çevirir.

        TM_ID'yi "Kaynak" ve "Varış" düğümlerine böler; OR-Tools'un
        kenar tabanlı araç atama algoritması bu ayrımı zorunlu kılar.

        Çıktı sütunlar
        --------------
        date                : Tahmin tarihi
        source              : Kaynak Transfer Merkezi
        destination         : Varış Transfer Merkezi
        q10                 : Düşük senaryo
        q50                 : Medyan tahmin
        q90                 : Yüksek senaryo (spot araç alarm seviyesi)
        recommended_demand  : q50 + safety_buffer (OR-Tools kapasite girdisi)

        Parameters
        ----------
        predictions : DemandForecaster.predict() çıktısı (List[Dict])
        date_key    : Tarih anahtarı (varsayılan: "tarih")
        group_key   : Grup anahtarı (varsayılan: "TM_ID")

        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd

        # 1. Tahminleri içeri al ve belirsizlik bantlarını/tamponları hesapla
        self.from_json(predictions, date_key=date_key, group_key=group_key)

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
                "date":               b.tarih,
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
                f"{len(df_ortools.columns)} sütun (date, source, destination, q10, q50, q90, "
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