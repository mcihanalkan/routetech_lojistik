# src/metrics.py
"""
Forecast Evaluation Modülü — İş Odaklı Metrikler

Metrik Felsefesi (Teknofest kısıtı):
  ❌ RMSE kullanılmaz  → Yön bilgisi taşımaz; "9 fazla" ile "9 eksik" aynı.
  ✅ WAPE              → Ana hat rotalarındaki hatayı hacimle ağırlıklandırır.
  ✅ Decision Regret   → Eksik tahmin = spot araç maliyeti simülasyonu.
  ✅ Quantile Coverage → Kantil bantlarının istatistiksel kalibrasyonu.
  ✅ Spot Cost Sim     → q90 alarmının gerçek TL maliyetini tahmin eder.

Hem bağımsız fonksiyon (geriye uyumluluk) hem ForecastEvaluator sınıfı
(pipeline entegrasyonu) olarak sunulur.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. TEMEL METRİK FONKSİYONLARI (geriye uyumlu, bağımsız kullanım)
# ===========================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    ⚠️  Lojistik değerlendirmede KULLANILMAZ (yön bilgisi taşımaz).
    Yalnızca referans karşılaştırma için bırakıldı.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: Ortalama desi hacmi sapması."""
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE) — Ana Operasyonel Metrik.

    Neden MAPE değil?
      MAPE: her satırı eşit ağırlıklandırır → küçük hacimli rotalar
            büyük yüzde hata üretir, ana hat rotaları geri planda kalır.
      WAPE: hatayı toplam hacme böler → yüksek hacimli rotaların
            hatası otomatik olarak daha ağır basar.

    Neden sıfır bölme riski yok?
      Payda `sum(y_true)`; tek bir satır değil, toplam hacim.
      Toplam hacim sıfırsa zaten anlamlı değerlendirme yapılamaz.

    Formül:
      WAPE = Σ|y - ŷ| / Σy

    Parameters
    ----------
    y_true : Gerçek değerler
    y_pred : Tahmin değerleri

    Returns
    -------
    float : 0.0 = mükemmel, 1.0 = %100 hata
    """
    sum_true = np.sum(y_true)
    if sum_true == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / sum_true)


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    """
    Pinball Loss (Kantil Kaybı) — Kantil Kalibrasyon Metriği.

    q10, q50, q90 bantlarının istatistiksel olarak ne kadar
    doğru kalibre edildiğini ölçer.

    İyi kalibre edilmiş q90: gerçek değerlerin %90'ı q90 altında kalır.

    Parameters
    ----------
    y_true : Gerçek değerler
    y_pred : Belirli bir kantil için tahmin
    alpha  : Kantil seviyesi (0.1, 0.5, 0.9)

    Returns
    -------
    float : Düşük = iyi kalibre
    """
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def decision_regret(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spot_multiplier: float = 9.0,
    idle_multiplier: float = 1.0,
) -> float:
    """
    Karar Pişmanlığı (Decision Regret / SPO Loss Simülasyonu).

    Lojistik ağındaki asimetrik maliyet yapısını sayısal olarak ölçer:

      Eksik tahmin (gerçek > tahmin):
        → Araç kapasitesi yetersiz → spot araç kiralanır
        → Maliyet: fark × spot_multiplier (varsayılan 9x)

      Fazla tahmin (gerçek < tahmin):
        → Kiralık araçta boş yer kalır (idle kapasite)
        → Maliyet: fark × idle_multiplier (varsayılan 1x)

    Bu metrik RMSE/MAE'den üstündür çünkü:
      "9 birim eksik tahmin" ile "9 birim fazla tahmin" aynı değildir.
      Decision Regret bu asimetriyi doğrudan kodlar.

    Parameters
    ----------
    y_true           : Gerçek talep değerleri
    y_pred           : Model tahminleri (genellikle q50)
    spot_multiplier  : Spot araç ceza katsayısı. Varsayılan: 9.0
    idle_multiplier  : Boş kapasite ceza katsayısı. Varsayılan: 1.0

    Returns
    -------
    float : Ortalama pişmanlık skoru (↓ düşük = daha az maliyet)
    """
    diff = y_true - y_pred

    regret = np.where(
        diff > 0,
        diff * spot_multiplier,        # Eksik tahmin → spot araç
        np.abs(diff) * idle_multiplier  # Fazla tahmin → boş kapasite
    )

    return float(np.mean(regret))


def quantile_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    Kantil Kapsama Oranı (Coverage Rate).

    "q10–q90 bandı gerçek değerlerin kaçını kapsıyor?"

    İdeal değer: ~%80 (q10 ile q90 arasında %80 veri beklenir).
    Çok düşük → bantlar çok dar, model özgüvenli ama yanıltıcı.
    Çok yüksek → bantlar çok geniş, ALNS için işe yaramaz belirsizlik.

    Parameters
    ----------
    y_true   : Gerçek değerler
    y_lower  : Alt bant (q10 tahminleri)
    y_upper  : Üst bant (q90 tahminleri)

    Returns
    -------
    float : 0.0–1.0 arası oran
    """
    covered = np.sum((y_true >= y_lower) & (y_true <= y_upper))
    return float(covered / len(y_true))


def spot_cost_simulation(
    y_true: np.ndarray,
    y_pred_q50: np.ndarray,
    y_pred_q90: np.ndarray,
    cost_per_unit_spot: float = 1.0,
    cost_per_unit_idle: float = 0.2,
) -> Dict[str, float]:
    """
    Spot Araç Maliyet Simülasyonu.

    İki strateji karşılaştırır:
      A) q50 ile karar ver  → medyan tahmini kullan
      B) q90 ile karar ver  → yüksek senaryo, daha az spot risk

    Her iki stratejinin beklenen toplam maliyetini hesaplar.
    ALNS motoru hangi kantili kullanacağına bu simülasyona bakarak karar verir.

    Parameters
    ----------
    y_true              : Gerçek talep
    y_pred_q50          : Medyan tahmin
    y_pred_q90          : Yüksek senaryo tahmin
    cost_per_unit_spot  : Spot araç birim maliyeti. Varsayılan: 1.0
    cost_per_unit_idle  : Boş kapasite birim maliyeti. Varsayılan: 0.2

    Returns
    -------
    Dict[str, float]
        q50_total_cost, q90_total_cost, savings_with_q90
    """
    def _total_cost(y_true, y_pred):
        diff = y_true - y_pred
        spot_cost = np.sum(np.maximum(diff, 0)) * cost_per_unit_spot
        idle_cost = np.sum(np.maximum(-diff, 0)) * cost_per_unit_idle
        return float(spot_cost + idle_cost)

    q50_cost = _total_cost(y_true, y_pred_q50)
    q90_cost = _total_cost(y_true, y_pred_q90)

    return {
        "q50_total_cost":   round(q50_cost, 4),
        "q90_total_cost":   round(q90_cost, 4),
        # Pozitif → q90 daha ucuz (daha az spot araç)
        # Negatif → q90 fazla ihtiyatlı (fazla idle kapasite)
        "savings_with_q90": round(q50_cost - q90_cost, 4),
    }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Geriye uyumlu tek seferlik değerlendirme fonksiyonu.

    ⚠️  RMSE referans için bırakıldı, operasyonel karar için kullanılmaz.
    Asıl metrikler: WAPE ve Decision_Regret.

    Parameters
    ----------
    y_true : Gerçek değerler
    y_pred : q50 tahminleri

    Returns
    -------
    Dict[str, float]
    """
    return {
        "WAPE":              wape(y_true, y_pred),
        "Decision_Regret":   decision_regret(y_true, y_pred),
        "MAE":               mae(y_true, y_pred),
        "Quantile_Loss_q50": quantile_loss(y_true, y_pred, alpha=0.5),
        "RMSE":              rmse(y_true, y_pred),  # yalnızca referans
    }


# ===========================================================================
# 2. FORECAST EVALUATOR SINIFI (pipeline entegrasyonu)
# ===========================================================================

class ForecastEvaluator:
    """
    DemandForecaster çıktılarını değerlendiren sınıf.

    Hem `pd.DataFrame` hem de `List[Dict]` (in-memory JSON) formatını kabul eder.
    Tek bir `evaluate()` çağrısıyla tüm metrikleri hesaplar.

    Parameters
    ----------
    spot_multiplier     : Decision Regret spot araç ceza katsayısı. Varsayılan: 9.0
    idle_multiplier     : Decision Regret boş kapasite katsayısı. Varsayılan: 1.0
    cost_per_unit_spot  : Spot maliyet simülasyonu birim fiyatı. Varsayılan: 1.0
    cost_per_unit_idle  : Boş kapasite simülasyonu birim fiyatı. Varsayılan: 0.2
    logging_enabled     : Detaylı log. Varsayılan: True

    Examples
    --------
    >>> evaluator = ForecastEvaluator(spot_multiplier=9.0)

    >>> # predict() çıktısından (List[Dict]) direkt değerlendirme
    >>> results = forecaster.predict(test_df)
    >>> scores = evaluator.evaluate_from_json(results, y_true=test_df["desi_hacmi"])

    >>> # DataFrame ile değerlendirme
    >>> scores = evaluator.evaluate(y_true, q50_preds, q10_preds, q90_preds)
    """

    def __init__(
        self,
        spot_multiplier: float = 9.0,
        idle_multiplier: float = 1.0,
        cost_per_unit_spot: float = 1.0,
        cost_per_unit_idle: float = 0.2,
        logging_enabled: bool = True,
    ):
        self.spot_multiplier    = spot_multiplier
        self.idle_multiplier    = idle_multiplier
        self.cost_per_unit_spot = cost_per_unit_spot
        self.cost_per_unit_idle = cost_per_unit_idle
        self.logging_enabled    = logging_enabled

        self.scores_: Dict[str, Any] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_q50: np.ndarray,
        y_pred_q10: Optional[np.ndarray] = None,
        y_pred_q90: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Tüm metrikleri tek seferde hesaplar.

        Parameters
        ----------
        y_true      : Gerçek talep değerleri
        y_pred_q50  : Medyan tahmin (operasyonel karar tahmini)
        y_pred_q10  : Alt güven bandı (opsiyonel)
        y_pred_q90  : Üst güven bandı / spot araç alarm (opsiyonel)

        Returns
        -------
        Dict[str, Any] : Tüm metrik skorları
        """
        y_true     = np.asarray(y_true,     dtype=float)
        y_pred_q50 = np.asarray(y_pred_q50, dtype=float)

        scores: Dict[str, Any] = {
            # ── Ana Operasyonel Metrikler ──────────────────────────────────
            "WAPE": wape(y_true, y_pred_q50),
            "Decision_Regret": decision_regret(
                y_true, y_pred_q50,
                spot_multiplier=self.spot_multiplier,
                idle_multiplier=self.idle_multiplier,
            ),
            # ── Kantil Kalibrasyonu ────────────────────────────────────────
            "Pinball_q50": quantile_loss(y_true, y_pred_q50, alpha=0.5),
            # ── Referans (karar için kullanılmaz) ─────────────────────────
            "MAE":  mae(y_true, y_pred_q50),
            "RMSE": rmse(y_true, y_pred_q50),
            "n_samples": len(y_true),
        }

        # Kantil bantları varsa ek metrikler
        if y_pred_q10 is not None and y_pred_q90 is not None:
            y_pred_q10 = np.asarray(y_pred_q10, dtype=float)
            y_pred_q90 = np.asarray(y_pred_q90, dtype=float)

            scores["Pinball_q10"] = quantile_loss(y_true, y_pred_q10, alpha=0.1)
            scores["Pinball_q90"] = quantile_loss(y_true, y_pred_q90, alpha=0.9)
            scores["Coverage_q10_q90"] = quantile_coverage(
                y_true, y_pred_q10, y_pred_q90
            )
            scores["Spot_Cost_Sim"] = spot_cost_simulation(
                y_true, y_pred_q50, y_pred_q90,
                cost_per_unit_spot=self.cost_per_unit_spot,
                cost_per_unit_idle=self.cost_per_unit_idle,
            )

        self.scores_ = scores

        if self.logging_enabled:
            self._log_scores(scores)

        return scores

    def evaluate_from_json(
        self,
        predictions: List[Dict[str, Any]],
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """
        DemandForecaster.predict() çıktısından (List[Dict]) direkt değerlendirme.

        predict() in-memory JSON döndürür; bu metod o formatı doğrudan kabul eder,
        ayrıca DataFrame dönüşümüne gerek kalmaz.

        Parameters
        ----------
        predictions : forecaster.predict() çıktısı (List[Dict])
        y_true      : Gerçek talep değerleri (aynı sırada)

        Returns
        -------
        Dict[str, Any]
        """
        q10 = np.array([r["q10"] for r in predictions], dtype=float)
        q50 = np.array([r["q50"] for r in predictions], dtype=float)
        q90 = np.array([r["q90"] for r in predictions], dtype=float)

        return self.evaluate(
            y_true=y_true,
            y_pred_q50=q50,
            y_pred_q10=q10,
            y_pred_q90=q90,
        )

    def _log_scores(self, scores: Dict[str, Any]) -> None:
        """Metrikleri okunabilir formatta loglar."""
        lines = [
            "\n📊 ForecastEvaluator — Değerlendirme Sonuçları",
            "=" * 50,
            f"  WAPE             : {scores['WAPE']:.4%}   (↓ düşük = iyi)",
            f"  Decision Regret  : {scores['Decision_Regret']:.4f}  (↓ az spot maliyet)",
            f"  Pinball q50      : {scores['Pinball_q50']:.4f}",
            f"  MAE              : {scores['MAE']:.4f}",
            f"  RMSE [ref]       : {scores['RMSE']:.4f}   (karar için kullanılmaz)",
        ]
        if "Coverage_q10_q90" in scores:
            lines += [
                f"  Coverage (q10–q90): {scores['Coverage_q10_q90']:.2%}  (ideal: ~%80)",
                f"  Pinball q10      : {scores['Pinball_q10']:.4f}",
                f"  Pinball q90      : {scores['Pinball_q90']:.4f}",
            ]
        if "Spot_Cost_Sim" in scores:
            sim = scores["Spot_Cost_Sim"]
            lines += [
                f"  Spot Cost (q50)  : {sim['q50_total_cost']:.2f}",
                f"  Spot Cost (q90)  : {sim['q90_total_cost']:.2f}",
                f"  q90 Tasarrufu    : {sim['savings_with_q90']:.2f}",
            ]
        lines.append("=" * 50)
        logger.info("\n".join(lines))

    def to_dataframe(self) -> pd.DataFrame:
        """
        Son değerlendirme sonuçlarını düz (flat) bir DataFrame'e çevirir.

        Spot_Cost_Sim iç içe sözlüğü düzleştirilir.
        MLflow veya raporlama için kullanışlı.
        """
        if not self.scores_:
            raise ValueError("❌ Önce evaluate() çağırın!")

        flat: Dict[str, float] = {}
        for k, v in self.scores_.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[f"{k}__{sub_k}"] = sub_v
            else:
                flat[k] = v

        return pd.DataFrame([flat])