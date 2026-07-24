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
_vehicle_leg_cost_cache: dict = {}


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

    PERFORMANS: bu fonksiyonun sonucu SADECE (route_lookup, hat, arac_turu, hourly_rate,
    km_rate)'e bağlı - taşınan desiye ya da state'e HİÇ bağlı değil, yani ALNS'in tüm
    çalışması boyunca AYNI girdiler için hep AYNI sonucu üretir. Bu fonksiyon saniyede
    binlerce kez (arama sırasında her aday değerlendirmesinde) çağrıldığı için,
    sonuçları önbelleğe alıp gereksiz yeniden hesaplamayı önlüyoruz (profil ile
    doğrulandı - bkz. sohbet geçmişi)."""
    cache_key = (id(route_lookup), hat, arac_turu, hourly_rate, km_rate)
    if cache_key in _vehicle_leg_cost_cache:
        return _vehicle_leg_cost_cache[cache_key]

    entry = route_lookup.get(hat)
    dist = entry["distance_km"] if entry else 0.0
    seyir_saat = entry[arac_turu] if entry else 0.0
    sonuc = int(round(hourly_rate * seyir_saat + dist * km_rate))
    _vehicle_leg_cost_cache[cache_key] = sonuc
    return sonuc


def ellecleme_maliyet_hesapla(desi: float, kiralık_saat_maliyet: float) -> float:
    """Elleçleme süresi = desi * 0.01 dk. Konsolidasyonda (indir + tekrar yükle) 2x sayılır."""
    sure = desi * ELLECLEME_DAKIKA_PER_DESI # 1000 desi elleçlenirse 10dk
    sure = math.ceil(sure) # dakika biriminde yukarı yuvarla
    sure_saat = sure / 60 # 1/6 saat
    maliyet = sure_saat * kiralık_saat_maliyet 
    return int(round(maliyet))

def spot_vehicle_count(desi: float, capacity: float, max_spot: int) -> int:
    """Bir bacakta taşınacak desi miktarını, kapasiteye göre gereken spot araç sayısına çevirir.

    DUZELTME: math.ceil()'e kucuk bir epsilon (1e-9) toleransi eklendi.
    ALNS, leg_spot_desi[key]'i binlerce iterasyon boyunca += / -= ile
    ARTIMLI guncelliyor; desi artik hep tam sayi olmadigindan (ondalikli
    talepler de girebiliyor) bu birikimli float toplama/cikarma, kapasite
    sinirinda (orn. tam 5600.0) minik bir kayma (5600.00000001 gibi)
    yaratabiliyor - epsilonsuz ceil() bu kaymayi FAZLADAN BIR ARAC olarak
    sayiyordu, bu da State.objective()'in (ALNS'in arama sirasinda gordugu
    sinyal) gercek maliyetten (rapor aninda bucket_toplam_desi'den SIFIRDAN
    hesaplanan) sapmasina yol aciyordu (bkz. sohbet gecmisi)."""
    if desi <= 0 or capacity <= 0:
        return 0
    return min(max_spot, math.ceil(desi / capacity - 1e-9))
