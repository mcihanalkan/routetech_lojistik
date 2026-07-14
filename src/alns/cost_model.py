"""Paylaşılan maliyet/kapasite hesap yardımcıları.

CP-SAT (`src/optimization.py`) ve ALNS (`src/alns_engine.py`) motorlarının aynı
maliyet formüllerini kullanmasını garanti eder — iki motor farklı sonuç verirse
neden farklı verdiğini anlamak neredeyse imkansız hale gelir, bu yüzden formüller
tek bir yerde tutuluyor.
"""

from __future__ import annotations

import math

from src.alns.time_model import RouteLookup
from src.config import (
    ELLECLEME_DAKIKA_PER_DESI
)
def vehicle_leg_cost(
    route_lookup: RouteLookup,
    hat: tuple[str, str],
    arac_turu: str,
    hourly_rate: float,
    km_rate: float
    # tasinan_desi
) -> int:
    """Tek bir araç sevkiyatının maliyeti: saatlik_kira × seyir_süresi + mesafe × km_maliyeti.
    NOT: SADECE SEYİR SÜRESİ ÜZERİNDEN SAATLİK KİRA HESABI YAPAR. ELLEÇLEME SÜRESİNİ DAHİL ETMEZ!

    PDF formülü: Toplam Araç Maliyeti = (Saatlik Kiralama Maliyeti × Kullanım Süresi)
    + (Kat Edilen Mesafe × Kilometre Başı Maliyet). CP-SAT tamsayı katsayı gerektirdiği
    için en yakın TL'ye yuvarlanır (ALNS için de tutarlılık amacıyla aynı yuvarlama).
    """
    toplam_sure = 0
    # ellecleme_suresi = tasinan_desi * 0.01 # dakika cinsinden
    # ellecleme_suresi_saat = ellecleme_suresi / 60
    entry = route_lookup.get(hat)
    dist = entry["distance_km"] if entry else 0.0
    seyir_saat = entry[arac_turu] if entry else 0.0
    # toplam_sure = seyir_saat + ellecleme_suresi_saat
    return int(round(hourly_rate * seyir_saat + dist * km_rate))


def ellecleme_maliyet_hesapla(desi: float, kiralık_saat_maliyet: float) -> float:
    """Elleçleme süresi = desi * 0.01 dk. Konsolidasyonda (indir + tekrar yükle) 2x sayılır."""
    sure = desi * ELLECLEME_DAKIKA_PER_DESI # 1000 desi elleçlenirse 10dk
    sure_saat = sure / 60 # 1/6 saat
    sure_saat = math.ceil(sure_saat) # 1 saat
    maliyet = sure_saat * kiralık_saat_maliyet 
    return int(round(maliyet))

def spot_vehicle_count(desi: float, capacity: float, max_spot: int) -> int:
    """Bir bacakta taşınacak desi miktarını, kapasiteye göre gereken spot araç sayısına çevirir."""
    if desi <= 0 or capacity <= 0:
        return 0
    return min(max_spot, math.ceil(desi / capacity))
