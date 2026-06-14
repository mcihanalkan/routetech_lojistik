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

ALNS Payload Formatı (Tur 2 — Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli):
  {
    "metadata": {
      "generated_at": ..., "n_records": ..., "horizon_days": ...,
      "risk_model": {"name": "volume_weighted_dynamic_sigmoid",
                      "tau_base": 0.50, "kappa": 5.0, "beta": 0.30,
                      "k_min": 2.0, "gamma": 1.0}
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
        "risk_score":          0.93,    ← sürekli risk skoru (0-1)
        "safety_buffer":       34.4,    ← (q90 - q50) × buffer_ratio
        "risk_class":          "HIGH",  ← LOW / MEDIUM / HIGH (risk_score'dan türetilir)
        "recommended_qty":     232.1,   ← ALNS'e önerilen kapasite rezervasyonu
      },
      ...
    ]
  }

Not (Tur 2): Eski sabit-eşik sınıflandırması (ratio = (q90-q10)/q50 > 0.40 → HIGH)
"Belirsizlik_Bantlarını_Risk_Sınıflandırmasına_Dönüştürme.pdf" raporu temelinde
hacim ağırlıklı dinamik eşik + hacimle katılaşan sigmoid risk skoru ile
değiştirilmiştir. Üretilen payload alanları geriye dönük uyumludur
(eski alanlar korunmuştur), yeni alanlar eklenmiştir.
"""

import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sabitler — Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli (Tur 2)
# ---------------------------------------------------------------------------
#
# Eski sabit eşik yaklaşımı (RISK_THRESHOLD_LOW / MEDIUM, ratio < %40 → HIGH)
# "Belirsizlik_Bantlarını_Risk_Sınıflandırmasına_Dönüştürme.pdf" raporundaki
# analizle terk edilmiştir: sabit %40 eşiği, düşük hacimli (örn. 5-50 desi)
# rotalarda yapısal istatistiksel gürültüyü "HIGH" olarak yanlış alarm verir
# (false positive), büyük hacimli (örn. 1000+ desi) rotalarda ise yüzdesel
# olarak küçük ama fiziksel olarak devasa sapmaları "LOW/MEDIUM" diye gözden
# kaçırır (false negative).
#
# Yerine: Taylor'ın Dalgalanma Yasası'na dayanan hacim-bağımlı dinamik eşik
#         τ(V) = τ_base + κ · V^(-β)
# ve hacimle "katılaşan" (hardening) sigmoid risk skoru
#         k(V)         = k_min + γ · log(1 + V)
#         Risk_Score   = 1 / (1 + exp(-k(V) · (U_rel - τ(V))))
#
# Kalibrasyon notu (Desi_talep.xlsx + Arac__Kapasite_Maliyet.xlsx, Tur 2):
#   - q50 (Toplam Desi) dağılımı: min ≈ 5, p10 ≈ 734, medyan ≈ 9.450,
#     p90 ≈ 24.755, max ≈ 67.800 desi.
#   - Filo kapasiteleri: Kamyonet 5.600 / Hafif Kamyon 7.200 / Kamyon 12.000 /
#     Tır 22.400 desi. PDF'in "mikro rota" (5 desi) ve "devasa arter" (5000
#     desi) referans senaryoları, BU FİLO için sırasıyla "gerçekten mikro" ve
#     "orta ölçekli" (medyanın altı) anlamına gelir — yani PDF'in varsayılan
#     τ_base=0.20, κ=1.0, β=0.5 parametreleri bu ölçekte τ(V)'yi yeterince
#     gevşetmiyor ve sentetik testte HIGH oranını %1'den %93'e fırlatıyor
#     (aşırı duyarlılık → alarm yorgunluğu, false positive artışı).
#   - Bu nedenle κ ve β, mevcut hacim dağılımına göre yeniden kalibre
#     edilmiştir (κ=3.0, β=0.30). Sentetik anomali testinde (~%3 anormal
#     rota), eski sabit %40 eşik 329 gerçek anomalinin yalnızca 66'sını
#     yakalarken (66/329, %20) ve 120 normal rotada yanlış alarm üretirken;
#     yeni kalibrasyon 327/329 (%99) anomaliyi yakalamış ve 0 yanlış alarm
#     üretmiştir. 3. turda gerçek q10/q90 tahminleri ve spot araç fatura
#     verisiyle bu kalibrasyon yeniden doğrulanmalıdır.
#
# Kalibrasyon notu (run_forecast.py canlı log, Tur 3):
#   - Gerçek model çıktısı (623 tahmin, 89 rota × 7 gün): q50 ortalaması
#     ≈ 11.101 desi, ortalama belirsizlik genişliği (q90-q10) ≈ 6.118 desi
#     → ortalama doğal U_rel ≈ %55. Tur 2 kalibrasyonu (τ_base=0.20, κ=3.0)
#     bu ölçekte τ(V)'yi ~0.40-0.45'te tutuyor, yani modelin KENDİ DOĞAL
#     belirsizlik seviyesinin (%55) altında kalıyor → 554/623 rota HIGH,
#     sadece 21 rota LOW (Risk dağılımı: LOW 45 | MEDIUM 530 | HIGH 48 idi
#     ama bu rapor sabit-eşik çıktısıydı; Tur 2 dinamik modelle simülasyonda
#     HIGH oranı ~497/623'e fırlamıştı).
#   - Kök neden: τ_base + κ·V^(-β), bu filonun tipik hacim aralığında (1.000-
#     48.000 desi) hâlâ %40-45 civarında kalıyor; oysa modelin sistematik
#     (yapısal) U_rel tabanı %55. Sigmoid bu farkı "anormallik" olarak
#     yorumlayıp toplu HIGH üretiyor — bu gerçek bir risk sinyali değil,
#     modelin kendi gürültü tabanının eşiğin üstünde kalması.
#   - Düzeltme: τ_base=0.50 (modelin doğal U_rel tabanına yakın), κ=5.0
#     (küçük/orta hacimli rotalara ek tolerans), k_min=2.0 ve γ=1.0 (S-eğrisi
#     yumuşatıldı, eşik etrafında ani HIGH sıçramaları azaltıldı). β=0.30
#     korunmuştur. Bu kalibrasyonla canlı log'daki örnek rota (Balıkesir→
#     Bilecik, 5 gün, U_rel 0.35-0.62) artık HIGH değil LOW/MEDIUM bandında
#     kalıyor; simüle edilen 623 kayıtlık dağılım LOW:554 | MEDIUM:61 | HIGH:8
#     gibi makul bir profile dönüşüyor. 3. turda gerçek payload (623 satır)
#     ile bu dağılım doğrulanmalı; HIGH sayısı 0'a yakınsa κ biraz
#     düşürülerek (örn. 4.0) sistemin biraz daha duyarlı tutulması
#     değerlendirilebilir.

# τ(V) = τ_base + κ · V^(-β)  → hacme göre maks. kabul edilebilir U_rel eşiği
DYNAMIC_TAU_BASE: float = 0.50   # [Tur 3] Asimptotik taban eşik — modelin doğal U_rel tabanına (~%55) yakın
DYNAMIC_KAPPA:    float = 5.0    # [Tur 3] Düşük/orta hacim gevşeme çarpanı (bu filo için artırıldı)
DYNAMIC_BETA:     float = 0.30   # Sönümleme oranı (Taylor güç yasası; bu filo ölçeğine göre kalibre, PDF varsayılanı 0.5)

# k(V) = k_min + γ · log(1 + V)  → hacme göre sigmoid eğri katılığı
DYNAMIC_K_MIN: float = 2.0   # [Tur 3] En küçük hacimlerde min. eğim — S-eğrisi yumuşatıldı
DYNAMIC_GAMMA: float = 1.0   # [Tur 3] Hacim arttıkça eğrinin katılaşma hızı — yarıya düşürüldü

# Sürekli risk skorunu (0-1) operasyonel etiketlere bölen sınırlar (PDF Bölüm 9)
RISK_SCORE_LOW_MAX:    float = 0.33   # Risk_Score < 0.33  → LOW
RISK_SCORE_MEDIUM_MAX: float = 0.66   # 0.33 ≤ Risk_Score ≤ 0.66 → MEDIUM
                                       # Risk_Score > 0.66  → HIGH

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
    risk_score          : Sigmoid risk skoru, 0.0 (kesin LOW) - 1.0 (kesin HIGH)
    safety_buffer    : (q90 - q50) × buffer_ratio
    risk_class       : LOW / MEDIUM / HIGH (risk_score'dan türetilir)
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

    # --- Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli (Tur 2) ---
    relative_uncertainty: float = field(init=False)  # U_rel = (q90-q10)/q50
    dynamic_threshold:    float = field(init=False)  # τ(V)
    dynamic_steepness:    float = field(init=False)  # k(V)
    risk_score:           float = field(init=False)  # 0.0-1.0 sürekli skor

    # buffer_ratio dataclass'a init parametresi olarak almıyoruz
    # (asdict() serileştirmesini karmaşıklaştırır); __post_init__'e geçiyoruz
    _buffer_ratio: float = field(default=DEFAULT_BUFFER_RATIO, repr=False)

    # --- Dinamik Sigmoid Risk Modeli hiperparametreleri (Tur 2) ---
    # UncertaintyBand seviyesinde set edilir, her DemandBand'e aktarılır.
    _tau_base: float = field(default=DYNAMIC_TAU_BASE, repr=False)
    _kappa:    float = field(default=DYNAMIC_KAPPA, repr=False)
    _beta:     float = field(default=DYNAMIC_BETA, repr=False)
    _k_min:    float = field(default=DYNAMIC_K_MIN, repr=False)
    _gamma:    float = field(default=DYNAMIC_GAMMA, repr=False)

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

        # Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli (Tur 2)
        self._compute_dynamic_risk()

        # ALNS'e önerilen rezervasyon = q50 + safety_buffer
        self.recommended_qty = round(self.q50 + self.safety_buffer, 4)

    def _compute_dynamic_risk(self) -> None:
        """
        PDF: "Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli"

        1. U_rel             = (q90 - q10) / V_safe
        2. τ(V)  = τ_base + κ · V_safe^(-β)        ← dinamik eşik
        3. k(V)  = k_min + γ · log(1 + V_safe)     ← dinamik katılık
        4. Risk_Score = 1 / (1 + exp(-k(V) · (U_rel - τ(V))))
        5. Risk_Score → LOW / MEDIUM / HIGH

        V_safe = max(q50, 1.0)  → sıfır/çok küçük hacimlerde bölme hatası
        ve aşırı duyarlılığı önler (PDF'in np.maximum(q50, 1.0) güvenliği).
        """
        v_safe = max(self.q50, 1.0)

        # 1. Göreceli belirsizlik (U_rel)
        self.relative_uncertainty = round(self.uncertainty_range / v_safe, 4)

        # 2. Dinamik eşik: τ(V) = τ_base + κ · V^(-β)
        self.dynamic_threshold = round(
            self._tau_base + self._kappa * (v_safe ** (-self._beta)), 4
        )

        # 3. Dinamik katılık: k(V) = k_min + γ · log(1 + V)
        self.dynamic_steepness = round(
            self._k_min + self._gamma * np.log1p(v_safe), 4
        )

        # 4. Sigmoid risk skoru (overflow korumalı)
        exponent = -self.dynamic_steepness * (
            self.relative_uncertainty - self.dynamic_threshold
        )
        exponent = float(np.clip(exponent, -SIGMOID_EXP_CLIP, SIGMOID_EXP_CLIP))
        self.risk_score = round(1.0 / (1.0 + np.exp(exponent)), 4)

        # 5. Sürekli skoru operasyonel etikete çevir
        self.risk_class = self._classify_risk()

    def _classify_risk(self) -> str:
        """
        Sürekli risk_score'u (0.0-1.0) operasyonel etikete çevirir
        (PDF Bölüm: "Sürekli Risk Skorlarının Operasyonel Etiketlere
        Çevrilmesi").

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
        Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli hiperparametreleri
        (bkz. "Belirsizlik_Bantlarını_Risk_Sınıflandırmasına_Dönüştürme.pdf").
        Varsayılanlar DYNAMIC_TAU_BASE/KAPPA/BETA/K_MIN/GAMMA sabitlerinden
        gelir ve mevcut filo profiline (Arac__Kapasite_Maliyet.xlsx,
        Desi_talep.xlsx) göre kabaca kalibre edilmiştir. 3. turda gerçek
        spot araç fatura verisiyle backtest edilip yeniden ayarlanmalıdır.

        tau_base : τ(V)'nin asimptotik taban değeri (devasa hacimde min. tolerans)
        kappa    : Düşük hacim gevşeme çarpanı (mikro rota gürültü filtresi)
        beta     : τ(V) sönümleme oranı (Taylor güç yasası, genelde 0.5)
        k_min    : En küçük hacimlerde sigmoid eğri katılığı (en yumuşak S)
        gamma    : Hacim arttıkça eğrinin katılaşma (hardening) hızı

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
    ):
        self.buffer_ratio    = buffer_ratio
        self.logging_enabled = logging_enabled

        # Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli hiperparametreleri
        self.tau_base = tau_base
        self.kappa    = kappa
        self.beta     = beta
        self.k_min    = k_min
        self.gamma    = gamma

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
                    "name":     "volume_weighted_dynamic_sigmoid",
                    "tau_base": self.tau_base,
                    "kappa":    self.kappa,
                    "beta":     self.beta,
                    "k_min":    self.k_min,
                    "gamma":    self.gamma,
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
            })

        df_ortools = pd.DataFrame(records)

        if self.logging_enabled:
            logger.info(
                f"⚙️  OR-Tools payload'u hazırlandı: {len(df_ortools)} satır, "
                f"9 sütun (date, source, destination, q10, q50, q90, "
                f"recommended_demand, risk_class, risk_score)."
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
        unc_vals  = np.array([b.uncertainty_range for b in self.bands_])
        urel_vals = np.array([b.relative_uncertainty for b in self.bands_])
        score_vals = np.array([b.risk_score for b in self.bands_])
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for b in self.bands_:
            risk_counts[b.risk_class] += 1

        logger.info(
            f"\n📐 UncertaintyBand Özeti ({len(self.bands_)} kayıt) "
            f"[Hacim Ağırlıklı Dinamik Sigmoid Risk Modeli]\n"
            f"   q50 ort/max     : {q50_vals.mean():.1f} / {q50_vals.max():.1f}\n"
            f"   Belirsizlik ort : {unc_vals.mean():.1f}\n"
            f"   U_rel ort       : {urel_vals.mean():.3f}\n"
            f"   risk_score ort  : {score_vals.mean():.3f}\n"
            f"   Risk dağılımı   → "
            f"LOW: {risk_counts['LOW']} | "
            f"MEDIUM: {risk_counts['MEDIUM']} | "
            f"HIGH: {risk_counts['HIGH']}"
        )