"""Paylaşılan maliyet/kapasite hesap yardımcıları.

CP-SAT (`src/optimization.py`) ve ALNS (`src/alns_engine.py`) motorlarının aynı
maliyet formüllerini kullanmasını garanti eder — iki motor farklı sonuç verirse
neden farklı verdiğini anlamak neredeyse imkansız hale gelir, bu yüzden formüller
tek bir yerde tutuluyor.
"""

from __future__ import annotations

import math

from src.alns.time_model import RouteLookup

# 1. Elleçleme Süresi
# Elleçleme süresi elleçleme işlemleri için gereken süreyi ifade etmektedir. Araçlar bir transfer
# merkezine geldiğinde barkod okutma gibi birçok işlem yapılır ve bunlara elleçleme denir. Bu
# işlemler süre alan işlemlerdir.
# Bu aşama için elleçleme süresi desi başına 0.01 dakika şeklinde belirlenmiştir.
# Örneğin 07.07 tarihinde 15.00’da bir transfer merkezine 5000 desilik bir yük ulaşmışsa;

# Elleçleme süresi:
# 5000*0.01= 50 dakikadır.
# Bu gönderiler 15.50’de transfer merkezine ulaşmış ve elleçlenmiş kabul edilir. SLA için süre
# hesaplarken araç varış anını değil elleçlenme işleminin tamamlanma anını esas almanız
# gerekmektedir.
def vehicle_leg_cost(
    route_lookup: RouteLookup,
    hat: tuple[str, str],
    arac_turu: str,
    hourly_rate: float,
    km_rate: float,
    tasinan_desi,
    is_consolidation: bool, 
) -> int:
    """Tek bir araç sevkiyatının maliyeti: saatlik_kira × seyir_süresi + mesafe × km_maliyeti.

    PDF formülü: Toplam Araç Maliyeti = (Saatlik Kiralama Maliyeti × Kullanım Süresi)
    + (Kat Edilen Mesafe × Kilometre Başı Maliyet). CP-SAT tamsayı katsayı gerektirdiği
    için en yakın TL'ye yuvarlanır (ALNS için de tutarlılık amacıyla aynı yuvarlama).
    """
    toplam_sure = 0
    ellecleme_suresi = tasinan_desi * 0.01 # dakika cinsinden
    if is_consolidation:
        ellecleme_suresi *= 2
    entry = route_lookup.get(hat)
    dist = entry["distance_km"] if entry else 0.0
    seyir_saat = entry[arac_turu] if entry else 0.0
    ellecleme_suresi_saat = ellecleme_suresi / 60
    toplam_sure = seyir_saat + ellecleme_suresi_saat
    return int(round(hourly_rate * toplam_sure + dist * km_rate))


def spot_vehicle_count(desi: float, capacity: float, max_spot: int) -> int:
    """Bir bacakta taşınacak desi miktarını, kapasiteye göre gereken spot araç sayısına çevirir."""
    if desi <= 0 or capacity <= 0:
        return 0
    return min(max_spot, math.ceil(desi / capacity))
