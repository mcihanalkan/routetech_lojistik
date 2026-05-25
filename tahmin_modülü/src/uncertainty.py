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

ALNS Payload Formatı:
  {
    "metadata": { "generated_at": ..., "n_records": ..., "horizon_days": ... },
    "demands":  [
      {
        "tarih":           "2026-01-08",
        "TM_ID":           "IST-01",
        "demand_low":      142.3,   ← q10 (kötümser alt sınır)
        "demand_base":     198.7,   ← q50 (operasyonel plan tahmini)
        "demand_high":     267.4,   ← q90 (spot araç alarm seviyesi)
        "safety_buffer":   34.4,    ← (q90 - q50) × buffer_ratio
        "risk_class":      "HIGH",  ← LOW / MEDIUM / HIGH
        "recommended_qty": 232.1,   ← ALNS'e önerilen kapasite rezervasyonu
      },
      ...
    ]
  }
"""

import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sabitler — Risk Sınıflandırma Eşikleri
# ---------------------------------------------------------------------------

# Belirsizlik oranı = uncertainty_range / q50
# Bu oran yüksekse tahmin güvenilmez → daha fazla güvenlik tamponu gerekir
RISK_THRESHOLD_LOW:    float = 0.20   # < %20 belirsizlik → LOW
RISK_THRESHOLD_MEDIUM: float = 0.50   # %20–%50          → MEDIUM
                                       # > %50            → HIGH

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
    uncertainty_range: q90 - q10 (toplam belirsizlik genişliği)
    safety_buffer    : (q90 - q50) × buffer_ratio
    risk_class       : LOW / MEDIUM / HIGH
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

    # buffer_ratio dataclass'a init parametresi olarak almıyoruz
    # (asdict() serileştirmesini karmaşıklaştırır); __post_init__'e geçiyoruz
    _buffer_ratio: float = field(default=DEFAULT_BUFFER_RATIO, repr=False)

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

        # Risk sınıfı
        self.risk_class = self._classify_risk()

        # ALNS'e önerilen rezervasyon = q50 + safety_buffer
        self.recommended_qty = round(self.q50 + self.safety_buffer, 4)

    def _classify_risk(self) -> str:
        """
        Belirsizlik oranına göre risk sınıfı belirler.

        Oran = uncertainty_range / q50
          LOW    : < %20  → Tahmin güvenilir, standart araç planlaması yeterli
          MEDIUM : %20–50 → Orta belirsizlik, küçük tampon öner
          HIGH   : > %50  → Yüksek belirsizlik, q90 alarmı → spot araç riski var
        """
        if self.q50 == 0:
            return "HIGH"   # Sıfır tahmin → veri sorunu, ihtiyatlı ol

        ratio = self.uncertainty_range / self.q50

        if ratio < RISK_THRESHOLD_LOW:
            return "LOW"
        elif ratio < RISK_THRESHOLD_MEDIUM:
            return "MEDIUM"
        else:
            return "HIGH"

    def to_dict(self) -> Dict[str, Any]:
        """ALNS payload formatına uygun sözlük döndürür."""
        return {
            "tarih":             self.tarih,
            "TM_ID":             self.tm_id,
            "demand_low":        self.q10,
            "demand_base":       self.q50,
            "demand_high":       self.q90,
            "uncertainty_range": self.uncertainty_range,
            "safety_buffer":     self.safety_buffer,
            "risk_class":        self.risk_class,
            "recommended_qty":   self.recommended_qty,
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
    ):
        self.buffer_ratio    = buffer_ratio
        self.logging_enabled = logging_enabled
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

        q50_vals = np.array([b.q50 for b in self.bands_])
        unc_vals = np.array([b.uncertainty_range for b in self.bands_])
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for b in self.bands_:
            risk_counts[b.risk_class] += 1

        logger.info(
            f"\n📐 UncertaintyBand Özeti ({len(self.bands_)} kayıt)\n"
            f"   q50 ort/max   : {q50_vals.mean():.1f} / {q50_vals.max():.1f}\n"
            f"   Belirsizlik ort: {unc_vals.mean():.1f}\n"
            f"   Risk dağılımı → "
            f"LOW: {risk_counts['LOW']} | "
            f"MEDIUM: {risk_counts['MEDIUM']} | "
            f"HIGH: {risk_counts['HIGH']}"
        )