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


def ellecleme_maliyet_hesapla(desi: float, kiralık_saat_maliyet: float, dokunus_sayisi: int = 2) -> float:
    """Bir BACAĞIN (tek fiziksel sevkiyatın) elleçleme maliyeti.

    PDF (OptiVision örneği): tek bacaklı, konsolidasyonsuz bir sevkiyatta bile
    Kullanım Süresi hem ÇIKIŞ hem VARIŞ elleçlemesini içerir (100dk+300dk(yol)+100dk=500dk).
    Yani her bacak - konsolidasyonlu olsun olmasın - kendi çıkışını VE kendi varışını
    faturalandırmalı: N bacaklı bir yolda toplam elleçleme "dokunuşu" varsayılan olarak
    2N'dir (ilk durak sadece yükleme, son durak sadece indirme, aradaki her durak hem
    indirme hem yeniden yükleme = 2 dokunuş).

    dokunus_sayisi: bu ÇAĞRIDA kaç elleçleme ucu (çıkış/varış) faturalandırılacağı
    (0, 1 ya da 2). Varsayılan 2 (mevcut/normal konsolidasyon davranışı). Uğrama
    (milk-run) senaryosunda ARA duraklarda araç hiç indirilmediği için o ucun payı
    0 (araya giren durak) ya da 1 (zincirin ilk/son bacağı, sadece kendi ucu gerçek)
    olabilir - bkz. alns_engine._commit_leg skip_src_handling/skip_dst_handling.

    time_model.ellecleme_suresi_dakika() ile AYNI yuvarlama sırasını kullanır:
    önce ham süre dokunus_sayisi ile çarpılır, SONRA tek seferde yukarı yuvarlanır."""
    if dokunus_sayisi <= 0 or desi <= 0:
        return 0
    sure = desi * ELLECLEME_DAKIKA_PER_DESI * dokunus_sayisi
    sure = math.ceil(sure) # dakika biriminde yukarı yuvarla
    sure_saat = sure / 60 # saat
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
