"""ALNS (Adaptive Large Neighborhood Search) motoru — Faz 2 optimizasyonu.

CP-SAT'ın (src/optimization.py) aggregate/direkt-hat modelinin aksine burada
talepler (hat, gün, slot, desi) birden fazla PARÇAYA (Assignment) bölünüp farklı
yollara (direkt, 1-aktarmalı ya da 2-aktarmalı — serbest konsolidasyon)
atanabiliyor. Karar uzayı "hangi parça hangi yola/araca/slota gidiyor" şeklinde,
destroy/repair hamleleriyle iteratif iyileştiriliyor — tam çözüm garantisi yok
ama CP-SAT'ın aggregate/direkt modelinin yapamadığı esnekliği (konsolidasyon,
büyük ölçekte makul sürede iyi sonuç) sağlıyor.

Maliyet/zaman formülleri src/cost_model.py ve src/time_model.py'den paylaşılıyor
— iki motorun (CP-SAT / ALNS) farklı sonuç vermesinin nedeni sadece karar uzayı
temsili olsun, formül farkı olmasın diye.

Kapsam sınırlamaları (v1 — bkz. plan dosyası):
  - En fazla 2 aktarma (3 bacak); daha uzun zincirler desteklenmiyor (arama
    uzayını sınırlı tutmak için — bkz. MAX_2HOP_CANDIDATES).
  - Hat başına en fazla MAX_RELAY_CANDIDATES (1-aktarma) + MAX_2HOP_CANDIDATES
    (2-aktarma) aday yol (tüm TM kombinasyonları değil).
  - Kiralık filo her zaman günün İLK slotunda (DEMAND_ARRIVAL_TIMES[0]) kalkar kabul
    edilir — kiralık zorunlu/sabit kalkış olduğu için tır kapasitesine katkısı
    sabit bir sayıdır (hangi kargoyu taşıdığı hâlâ serbestçe optimize edilir,
    sadece kalkış SLOTU sabitlenmiştir; maliyeti zaten slot'tan bağımsızdır).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy.random as rnd
import math
from ortools.sat.python import cp_model

from src.alns.cost_model import spot_vehicle_count, vehicle_leg_cost, ellecleme_maliyet_hesapla
from src.alns.time_model import (
    DISPATCH_SLOTS,
    DEMAND_ARRIVAL_TIMES,
    RouteLookup,
    arrival_day,
    ellecleme_tamamlanma_zamani,
    gecikme_saat,
    next_dispatch_slot,
    sla_cezasi_tl,
    sla_deadline,
    slot_datetime,
    slot_to_hour,
    varis_zamani,
)

MAX_SPOT = 500
MAX_RELAY_CANDIDATES = 4       # hat basina en fazla 1-aktarmali (2 bacak) aday sayisi
MAX_2HOP_CANDIDATES = 3        # hat basina en fazla 2-aktarmali (3 bacak) aday sayisi
KIRALIK_DISPATCH_SLOT = DEMAND_ARRIVAL_TIMES[0]


# ============================================================================
# Problem verisi (salt-okunur, State'ler arasında paylaşılır)
# ============================================================================
@dataclass
class ProblemData:
    route_lookup: RouteLookup
    arac_turleri: list
    arac_parametreleri: dict
    kiralik_stok_gunluk: dict
    handling_capacity: dict
    tir_capacity: dict
    tir_arac_turu: Optional[str]
    gunler: list
    merkezler: list
    demands: list  # (hat, gun, slot, desi, talep_id)
    zaman_sirali: list = field(default_factory=list) # Python'ın nesneler arası hafıza çakışmasını engeller, her nesneye özel ve bağımsız bir boş liste üretir.
    relay_candidates: dict = field(default_factory=dict)
    fixed_kiralik_tir_usage: dict = field(default_factory=dict)  # (tm, gun) -> adet

    def __post_init__(self):
        if not self.zaman_sirali:
            self.zaman_sirali = [(g, s) for g in self.gunler for s in DEMAND_ARRIVAL_TIMES]
        if not self.relay_candidates:
            hatlar = sorted({h for (h, g, s, _, _) in self.demands})
            self.relay_candidates = _build_relay_candidates(self.route_lookup, hatlar, self.merkezler)
        if not self.fixed_kiralik_tir_usage and self.tir_arac_turu is not None:
            self.fixed_kiralik_tir_usage = _build_fixed_kiralik_tir_usage(self)


def _build_relay_candidates(route_lookup: RouteLookup, hatlar, merkezler) -> dict:
    """Hat başına aday konsolidasyon yollarını üretir: 1-aktarmalı (src->r->dst)
    ve 2-aktarmalı (src->r1->r2->dst) — ikisi de mesafeye göre en ucuz birkaç
    adayla sınırlı (arama uzayının patlamasını önlemek için). Her aday, path
    tuple'ı olarak saklanır: (r,) ya da (r1, r2)."""
    candidates: dict = {}
    for hat in hatlar:
        src, dst = hat
        tek_aktarma = []
        for relay in merkezler:
            if relay in (src, dst):
                continue
            leg1 = route_lookup.get((src, relay))
            leg2 = route_lookup.get((relay, dst))
            if leg1 is None or leg2 is None: # İki bacağın da olması gerekir.
                continue
            tek_aktarma.append((leg1["distance_km"] + leg2["distance_km"], relay))

        #   def isimsiz_fonksiyon(t):
        #       return t[0]
        #   lambda t: t[0]
        # ikisi aynı işi yapar.

        tek_aktarma.sort(key=lambda t: t[0])
        en_iyi_tek = [r for _, r in tek_aktarma[:MAX_RELAY_CANDIDATES]]

        # 2-aktarma adayları yalnızca en iyi tek-aktarma relay'lerinin komşuları
        # arasından aranır (tüm TM çiftlerini taramak yerine) - arama uzayını
        # makul tutmak için (18 TM'de bile tam tarama hat başına yüzlerce
        # kombinasyon demek, MAX_2HOP_CANDIDATES ile zaten en ucuzlar seçiliyor).
        iki_aktarma = []
        for r1 in en_iyi_tek:
            leg1 = route_lookup.get((src, r1))
            if leg1 is None:
                continue
            for r2 in merkezler:
                if r2 in (src, dst, r1):
                    continue
                leg2 = route_lookup.get((r1, r2))
                leg3 = route_lookup.get((r2, dst))
                if leg2 is None or leg3 is None:
                    continue
                toplam = leg1["distance_km"] + leg2["distance_km"] + leg3["distance_km"]
                iki_aktarma.append((toplam, (r1, r2)))
        iki_aktarma.sort(key=lambda t: t[0])

        candidates[hat] = [(r,) for r in en_iyi_tek] + [p for _, p in iki_aktarma[:MAX_2HOP_CANDIDATES]]
    return candidates


def _build_fixed_kiralik_tir_usage(data: "ProblemData") -> dict:
    """Kiralık filo her gün DEMAND_ARRIVAL_TIMES[0]'da kalkar (bkz. modül docstring) —
    tır kapasitesine sabit katkısı, karar değişkenlerinden bağımsız olarak
    baştan hesaplanabilir.

    Kenar durum: bir TM'nin tır kapasitesi 0 (ya da kiralık stoktan az) olabilir.
    CP-SAT modelinde (src/optimization.py) kiralik_x tır kapasitesine karşı da
    kısıtlanır — sabit maliyet her hâlükârda ödenir ama fiilen o kadar araç
    DIŞARI ÇIKARILMAZ. Burada da aynı sonucu taklit etmek için, TM başına toplam
    sabit kullanım kapasiteyle sınırlanır (fazlası "kâğıt üzerinde kalır")."""
    usage: dict = {}
    if data.tir_arac_turu is None:
        return usage
    for (hat, arac_turu), stok in data.kiralik_stok_gunluk.items():
        if arac_turu != data.tir_arac_turu or stok <= 0:
            continue
        src, dst = hat
        for gun in data.gunler:
            usage[(src, gun)] = usage.get((src, gun), 0) + stok
            varis_g = arrival_day(data.route_lookup, data.gunler, hat, gun, KIRALIK_DISPATCH_SLOT, arac_turu)
            if varis_g:
                usage[(dst, varis_g)] = usage.get((dst, varis_g), 0) + stok
    for (tm, gun), miktar in list(usage.items()):
        cap = data.tir_capacity.get(tm)
        if cap is not None and miktar > cap:
            usage[(tm, gun)] = int(cap)
    return usage


# ============================================================================
# Assignment — bir talep parçasının somut yerleşimi
# ============================================================================
@dataclass(frozen=True)
class Leg:
    src: str
    dst: str
    gun: str
    slot: str
    arac_turu: str
    is_kiralik: bool


@dataclass(frozen=True)
class Assignment:
    demand_hat: tuple
    demand_gun: str
    demand_slot: str
    desi: float
    legs: tuple  # tuple[Leg, ...] — 1 (direkt) ya da 2 (relay)
    sla_cost: float
    vehicle_cost: float  # sadece bu parcaya ait MARJINAL spot maliyeti (kiralik = 0)
    talep_id: str = ""


# ============================================================================
# State
# ============================================================================
class State:
    def __init__(self, data: ProblemData):
        self.data = data
        self.assignments: list = [] # (demand_hat, demand_gun, demand_slot, desi, legs, sla_cost, vehicle_cost)
        self.unassigned: list = list(data.demands)
        self.leg_spot_desi: dict = {}      # (src,dst,gun,slot,arac_turu) -> desi
        self.leg_kiralik_desi: dict = {}   # (src,dst,gun,slot,arac_turu) -> desi
        self.handling_usage: dict = {}     # (tm,gun) -> desi
        self.tir_usage: dict = {}          # (tm,gun) -> adet (spot kaynakli, kiralik ayrica sabit)
        # self._fixed_kiralik_cost = self._compute_fixed_kiralik_cost()
        self._fixed_kiralik_cost = self._kiralik_bos_seyir_maliyeti()

    # def _compute_fixed_kiralik_cost(self) -> float:
    #     total = 0.0
    #     for (hat, arac_turu), stok in self.data.kiralik_stok_gunluk.items():
    #         if stok <= 0:
    #             continue
    #         p = self.data.arac_parametreleri[arac_turu]
    #         birim = vehicle_leg_cost(
    #             self.data.route_lookup,
    #             hat,
    #             arac_turu,
    #             p["rental_hourly"],
    #             p["rental_km"],
    #             tasinan_desi=0.0, # kiralık aracın sabit maliyeti hesaplanıyor
    #         )
    #         total += len(self.data.gunler) * stok * birim
    #     return total

    def _kiralik_bos_seyir_maliyeti(self) -> float:
        total = 0.0
        for (hat, arac_turu), stok in self.data.kiralik_stok_gunluk.items():
            if stok <= 0:
                continue
            p = self.data.arac_parametreleri[arac_turu]
            # SADECE yolda geçen süre ve mesafe (elleçleme yok, desi = 0)
            entry = self.data.route_lookup.get(hat)
            seyir_saat = entry[arac_turu] if entry else 0.0
            dist = entry["distance_km"] if entry else 0.0
            birim_bos_maliyet = (seyir_saat * p["rental_hourly"]) + (dist * p["rental_km"])
            
            total += len(self.data.gunler) * stok * birim_bos_maliyet
        return total
        
    def copy(self) -> "State":
        new = State.__new__(State)
        new.data = self.data
        new.assignments = list(self.assignments)
        new.unassigned = list(self.unassigned)
        new.leg_spot_desi = dict(self.leg_spot_desi)
        new.leg_kiralik_desi = dict(self.leg_kiralik_desi)
        new.handling_usage = dict(self.handling_usage)
        new.tir_usage = dict(self.tir_usage)
        new._fixed_kiralik_cost = self._fixed_kiralik_cost
        return new

    def objective(self) -> float:
        total = self._fixed_kiralik_cost
        for a in self.assignments:
            total += a.vehicle_cost + a.sla_cost
        if self.unassigned:
            # Guvenlik agi: "tum desiler teslim edilmeli" sert bir gereksinim (bkz.
            # plan/PDF). Repair operatorleri her zaman unassigned'i tam bosaltmali;
            # ama bir operator bunu (bug ya da kenar durum yuzunden) yapamazsa,
            # objective() bunu ASLA odullendirmemeli - aksi halde ALNS, teslim
            # etmeyi "unutmayi" ucuz bir cozum sanip tercih edebilir. Cok agir bir
            # ceza (asiri gecikme varsayimiyla) ekleniyor.
            eksik_desi = sum(x[3] for x in self.unassigned)
            total += sla_cezasi_tl(eksik_desi, 24 * 30)  # 30 gunluk gecikme varsayimi

        # Guvenlik agi #2: force_insert() (son care, tum desilerin teslimini
        # garanti etmek icin) kapasite kontrolunu KASITLI olarak atlar - bu yuzden
        # elleceleme/tir kapasitesini asan durumlar teorik olarak olusabilir.
        # objective() bunu da agir cezalandirmali ki ALNS (daha fazla iterasyonla)
        # bu asimlari azaltmaya/ortadan kaldirmaya CALISSIN - aksi halde arama hic
        # bu yonde bir baski hissetmez. (CP-SAT'ta bu durum hic olusamaz - burada
        # heuristik oldugumuz icin acikca cezalandiriyoruz.)
        for tm, cap in self.data.handling_capacity.items():
            for gun in self.data.gunler:
                asim = self.handling_usage.get((tm, gun), 0.0) - cap
                if asim > 0:
                    total += asim * 1000.0
        if self.data.tir_arac_turu is not None:
            for tm, cap in self.data.tir_capacity.items():
                for gun in self.data.gunler:
                    kullanim = self.tir_usage.get((tm, gun), 0) + self.data.fixed_kiralik_tir_usage.get((tm, gun), 0)
                    asim = kullanim - cap
                    if asim > 0:
                        total += asim * 50000.0
        return total

    # ---- kapasite sorguları ----
    def kiralik_available_desi(self, hat, gun, arac_turu) -> float:
        stok = self.data.kiralik_stok_gunluk.get((hat, arac_turu), 0)
        if stok <= 0:
            return 0.0
        kap = self.data.arac_parametreleri[arac_turu]["kapasite_desi"]
        used = self.leg_kiralik_desi.get((hat[0], hat[1], gun, KIRALIK_DISPATCH_SLOT, arac_turu), 0.0)
        return max(0.0, stok * kap - used)

    def handling_available(self, tm, gun) -> float:
        cap = self.data.handling_capacity.get(tm)
        if cap is None:
            return float("inf")
        return max(0.0, cap - self.handling_usage.get((tm, gun), 0.0))

    def tir_available(self, tm, gun) -> float:
        cap = self.data.tir_capacity.get(tm)
        if cap is None:
            return float("inf")
        fixed = self.data.fixed_kiralik_tir_usage.get((tm, gun), 0)
        return max(0.0, cap - fixed - self.tir_usage.get((tm, gun), 0))

    def spot_capacity_left_on_leg(self, src, dst, gun, slot, arac_turu) -> float:
        kap = self.data.arac_parametreleri[arac_turu]["kapasite_desi"]
        mevcut = self.leg_spot_desi.get((src, dst, gun, slot, arac_turu), 0.0)
        return max(0.0, MAX_SPOT * kap - mevcut)

    # ---- bir bacağa desi eklemenin (varsa) izin verilen en fazla miktarını hesapla ----
    def max_addable_on_leg(self, src, dst, gun, slot, arac_turu, is_kiralik) -> float:
        if is_kiralik:
            limit = self.kiralik_available_desi((src, dst), gun, arac_turu)
        else:
            limit = self.spot_capacity_left_on_leg(src, dst, gun, slot, arac_turu)
        limit = min(limit, self.handling_available(src, gun))
        varis_g = arrival_day(self.data.route_lookup, self.data.gunler, (src, dst), gun, slot, arac_turu)
        if varis_g is None:
            return 0.0
        limit = min(limit, self.handling_available(dst, varis_g))
        if arac_turu == self.data.tir_arac_turu and not is_kiralik:
            # Tir kapasitesi arac SAYISI bazli - mevcut sayidan ne kadar artis olabilecegini
            # (kapasitenin izin verdigi) desi'ye kabaca cevirip sinirla.
            kap = self.data.arac_parametreleri[arac_turu]["kapasite_desi"]
            mevcut_desi = self.leg_spot_desi.get((src, dst, gun, slot, arac_turu), 0.0)
            mevcut_adet = spot_vehicle_count(mevcut_desi, kap, MAX_SPOT)
            departure_room = self.tir_available(src, gun)
            arrival_room = self.tir_available(dst, varis_g)
            adet_limiti = min(departure_room, arrival_room)
            # mevcut_adet'i asmadan kac desi daha eklenebilir (adet_limiti kadar YENI arac acilabilir)
            max_adet = mevcut_adet + adet_limiti
            limit = min(limit, max(0.0, max_adet * kap - mevcut_desi))
        return max(0.0, limit)

    # ---- bir bacağa desi commit et (trackerlari günceller, delta maliyet döndürür) ----ü
    # Örnek olarak, bir bacakta gidecek 15000 desi olsun. 2000 desi daha eklenecek.
    def _commit_leg(self, src, dst, gun, slot, arac_turu, desi, is_kiralik) -> float:
        key = (src, dst, gun, slot, arac_turu) 
        p = self.data.arac_parametreleri[arac_turu]
        
        # 1. Kiralık ve Spot saatlik/km ücretlerini belirle
        hourly_rate = p["rental_hourly"] if is_kiralik else p["spot_hourly"]
        km_rate = p["rental_km"] if is_kiralik else p["spot_km"]

        # 2. O bacağın o anki TOPLAM faturasını hesaplayan yerel formül
        def bacak_toplam_maliyeti(mevcut_desi, mevcut_arac_sayisi):

            birim = vehicle_leg_cost(self.data.route_lookup, (src,dst) , arac_turu, hourly_rate, km_rate)
            seyir_faturasi = mevcut_arac_sayisi * birim
            ellecleme_faturasi = ellecleme_maliyet_hesapla(mevcut_desi,hourly_rate)

            return seyir_faturasi + ellecleme_faturasi

        if is_kiralik:
            eski_desi = self.leg_kiralik_desi.get(key, 0.0)
            yeni_desi = eski_desi + desi
            self.leg_kiralik_desi[key] = yeni_desi
            
            # Kiralık aracın seyir maliyeti baştan ödendi! 
            # Bize sadece bu desiyi yüklemek/indirmek için harcanan zamanın maliyeti yansır.
            
            eski_ellecleme = ellecleme_maliyet_hesapla(eski_desi, p["rental_hourly"])
            yeni_ellecleme = ellecleme_maliyet_hesapla(yeni_desi, p["rental_hourly"])
            marjinal_maliyet = yeni_ellecleme - eski_ellecleme
            
        else:
            eski_desi = self.leg_spot_desi.get(key, 0.0)
            kap = p["kapasite_desi"]
            eski_adet = spot_vehicle_count(eski_desi, kap, MAX_SPOT)
            
            yeni_desi = eski_desi + desi
            yeni_adet = spot_vehicle_count(yeni_desi, kap, MAX_SPOT)
            self.leg_spot_desi[key] = yeni_desi
            
            # Spot aracta hem araç sayısı (seyir) hem de desi (elleçleme) artabilir. 
            # İkisinin de yarattığı fiyat artışı kusursuzca yakalanır.
            eski_maliyet = bacak_toplam_maliyeti(eski_desi, eski_adet)
            yeni_maliyet = bacak_toplam_maliyeti(yeni_desi, yeni_adet)
            marjinal_maliyet = yeni_maliyet - eski_maliyet

            if arac_turu == self.data.tir_arac_turu:
                delta_adet = yeni_adet - eski_adet
                self.tir_usage[(src, gun)] = self.tir_usage.get((src, gun), 0) + delta_adet
                varis_g = arrival_day(self.data.route_lookup, self.data.gunler, (src, dst), gun, slot, arac_turu)
                if varis_g:
                    self.tir_usage[(dst, varis_g)] = self.tir_usage.get((dst, varis_g), 0) + delta_adet

        ellecleme_suresi_saat = (desi * 0.01) / 60

        self.handling_usage[(src, gun)] = self.handling_usage.get((src, gun), 0.0) + desi
        varis_g = arrival_day(self.data.route_lookup, self.data.gunler, (src, dst), gun, slot, arac_turu)
        if varis_g:
            self.handling_usage[(dst, varis_g)] = self.handling_usage.get((dst, varis_g), 0.0) + desi
            
        return marjinal_maliyet


# ============================================================================
# Yol (path) seçenekleri ve maliyet/SLA hesabı
# ============================================================================
def _pick_vehicle_type(data: ProblemData, desi: float) -> list:
    """Araç türlerini büyükten küçüğe kapasiteye göre sırala (yalnızca zaman/hız
    tahmini gibi maliyetin önemsiz olduğu yerlerde kullanılır — bkz. _rank_spot_types_by_cost)."""
    return sorted(data.arac_turleri, key=lambda a: -data.arac_parametreleri[a]["kapasite_desi"])


def _rank_spot_types_by_cost(data: ProblemData, hat: tuple, desi: float) -> list:
    """Spot araç türlerini, bu MİKTARI taşımak için gereken tahmini toplam maliyete
    (ceil(desi/kapasite) × birim_maliyet) göre ARTAN sırada döndürür. "Büyük araç
    her zaman daha iyi" varsayımı YANLIŞ — küçük bir yük için gereksiz büyük/pahalı
    bir araç (Tır) seçmek maliyeti ciddi şekilde şişirir; bu yüzden her çağrıda
    gerçek maliyet karşılaştırılır."""
    def tahmini_maliyet(arac_turu: str) -> float:
        p = data.arac_parametreleri[arac_turu] 
        kap = p["kapasite_desi"] # Araç türünün kapasitesi
        adet = spot_vehicle_count(desi, kap, 10 ** 9) if desi > 0 else 0 # Bu kadar araç
        birim_maliyet = vehicle_leg_cost(data.route_lookup, hat, arac_turu, p["spot_hourly"], p["spot_km"])
        maliyet = adet * birim_maliyet
        ellecleme_maliyet = ellecleme_maliyet_hesapla(desi, p["spot_hourly"])
        return maliyet + ellecleme_maliyet

    return sorted(data.arac_turleri, key=tahmini_maliyet)


def leg_zaman_cizelgesi(data: ProblemData, legs: list, desi: float) -> list:
    """Her bacağın GERÇEK (elleçleme dahil) kalkış ve varış anını sırayla döndürür:
    [(kalkis_0, varis_0), (kalkis_1, varis_1), ...]. Hem SLA/tamamlanma hesabı
    (_completion_datetime) hem rapor (alns_optimize.py) bu ORTAK hesabı kullanır -
    iki yerde aynı mantığın ayrı ayrı yazılıp birbirinden sapmasını (bkz. Sorun 2) önlemek için."""
    cizelge = []
    zaman = None
    for i, leg in enumerate(legs):
        slot_zamani = slot_datetime(leg.gun, leg.slot)

        if i == 0:
            kalkis = ellecleme_tamamlanma_zamani(slot_zamani, desi, consolidation=False)
        else:
            kalkis = max(zaman, slot_zamani)

        seyir = data.route_lookup[(leg.src, leg.dst)][leg.arac_turu]
        varis = varis_zamani(kalkis, seyir)
        cizelge.append((kalkis, varis))

        if i < len(legs) - 1:
            zaman = ellecleme_tamamlanma_zamani(varis, desi, consolidation=True)
        else:
            zaman = varis

    return cizelge


def _completion_datetime(data: ProblemData, legs: list, desi: float):
    son_varis = leg_zaman_cizelgesi(data, legs, desi)[-1][1]
    return ellecleme_tamamlanma_zamani(son_varis, desi, consolidation=False)


# def _completion_datetime(data: ProblemData, legs: list, desi: float):
#     """SLA için esas alınan an: PDF'e göre araç VARIŞI değil, o TM'deki
#     elleçlemenin TAMAMLANMA anı. Son bacağın kalkışı+seyir süresiyle varışa
#     ulaşılır, sonra final destinasyondaki elleçleme süresi (desi × 0.01 dk)
#     eklenir — bu adım daha önce hiç yapılmıyordu ve büyük yüklerde (binlerce
#     desi → onlarca dakika/birkaç saat) SLA'yı yapay şekilde "zamanında"
#     gösteriyordu (bkz. sohbet geçmişi).

#     Ara (relay) bacaklarının kendi elleçlemesi burada AYRICA eklenmiyor -
#     `next_dispatch_slot` zaten bir sonraki bacağın kalkışını, elleçleme+bekleme
#     payı bırakarak hesaplıyor (bkz. o fonksiyonun docstring'i); yalnızca NİHAİ
#     varış noktasındaki elleçleme SLA tamamlanma anını geciktirir."""
#     zaman = None
#     for leg in legs:
#         kalkis = slot_datetime(leg.gun, leg.slot)
#         seyir = data.route_lookup[(leg.src, leg.dst)][leg.arac_turu]
#         zaman = varis_zamani(kalkis, seyir)
#     return ellecleme_tamamlanma_zamani(zaman, desi, consolidation=False)

def try_insert_path(
    state: State,
    hat: tuple,
    gun: str,
    slot: str,
    desi: float,
    talep_id: str = "",
    path: tuple = (),
    demand_gun: Optional[str] = None,
    demand_slot: Optional[str] = None,
) -> Optional[Assignment]:
    """Belirli bir yol (direkt ya da 1-2 aktarmalı) için mümkün olan en fazla
    deseyi yerleştirmeyi dener. Hiç yer yoksa None döner.

    `path`: ara aktarma noktalarının sıralı tuple'ı - boş tuple = direkt,
    `(r,)` = 1 aktarma, `(r1, r2)` = 2 aktarma. Kaç bacak olursa olsun aynı
    genel mantıkla işlenir (bkz. altta stops listesi).

    `gun`/`slot`: bu denemedeki FİİLİ kalkış (gün, slot) — ertelenmiş bir
    denemede bu, orijinal talep zamanından SONRAKİ bir slot olabilir.
    `demand_gun`/`demand_slot`: talebin GERÇEK oluşum (gün, slot)'u — SLA
    deadline'ı HER ZAMAN buradan hesaplanır (verilmezse gun/slot ile aynı
    kabul edilir, yani ertelenmemiş normal çağrı). Bu ayrım olmadan, ertelenmiş
    bir sevkiyatın SLA deadline'ı yanlışlıkla ertelenmiş kalkış anından
    hesaplanır — bu da gecikmeyi her zaman "0 saat" gibi gösterir (gerçek bug,
    bkz. sohbet geçmişi: "SLA cezası çok az" bulgusu)."""

    demand_gun = gun if demand_gun is None else demand_gun
    demand_slot = slot if demand_slot is None else demand_slot
    
    src, dst = hat
    data = state.data

    stops = [src, *path, dst] # yıldızlı kullanım, path tuple'ı içindeki elemanları tek tek açar.
    leg_departures = [(gun, slot)]
    for i in range(len(stops) - 1):
        if i == len(stops) - 2:
            break  # son bacagin kalkisi zaten bir onceki adimda belirlendi
        leg_src, leg_dst = stops[i], stops[i + 1]
        entry = data.route_lookup.get((leg_src, leg_dst))
        if entry is None:
            return None
        # Zamanlama tahmini icin en ucuz turun seyir suresi kullanilir (gercek
        # arac turu asagidaki leg_plans dongusunde ayrica/bagimsiz secilir).
        est_arac_turu = _rank_spot_types_by_cost(data, (leg_src, leg_dst), desi)[0]
        cur_gun, cur_slot = leg_departures[-1]
        sonraki = next_dispatch_slot(data.gunler, cur_gun, cur_slot, entry[est_arac_turu])
        if sonraki is None:
            return None
        leg_departures.append(sonraki)

    leg_pairs = [
        (stops[i], stops[i + 1], leg_departures[i][0], leg_departures[i][1])
        for i in range(len(stops) - 1)
    ] # Araç bu TM'den bu TM'ye şu günde şu saatte (09:00 veya 17:00) kalkacak.

    # Her bacak icin once KIRALIK (marjinal maliyet 0, ucretsiz kapasite) denenir;
    # yoksa SPOT icin en UCUZ (en buyuk kapasiteli degil!) arac turu secilir - bkz.
    # _rank_spot_types_by_cost docstring: "buyuk arac her zaman daha iyi" varsayimi
    # kucuk yukler icin maliyeti ciddi sise sisiriyordu (asil bug buradaydi).
    leg_plans = []
    for (leg_src, leg_dst, leg_gun, leg_slot) in leg_pairs:
        best = None  # (miktar, arac_turu, is_kiralik)
        if leg_slot == KIRALIK_DISPATCH_SLOT:
            for arac_turu in data.arac_turleri:
                miktar = state.max_addable_on_leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, True)
                if miktar > 0:
                    best = (miktar, arac_turu, True)
                    break
        is_final_slot = (leg_gun, leg_slot) == data.zaman_sirali[-1]
        if best is None:
            for arac_turu in _rank_spot_types_by_cost(data, (leg_src, leg_dst), desi):
                miktar = state.max_addable_on_leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, False)
                if miktar <= 0:
                    continue
                kap = data.arac_parametreleri[arac_turu]["kapasite_desi"]
                mevcut = state.leg_spot_desi.get((leg_src, leg_dst, leg_gun, leg_slot, arac_turu), 0.0)
                onerilen = min(desi, miktar)
                # Faz-1'deki %10 minimum doluluk kuralinin ALNS'teki esdegeri: bu
                # bacakta HENUZ spot arac yoksa (yeni acilacak), en az %10 dolulugu
                # saglamayan minik/verimsiz tek seferlik sevkiyati reddet - chunk
                # relay/ertelenmis slota yonlenir. Bu kural olmadan CP-SAT'in zamana
                # yayarak sagladigi konsolidasyon verimliligi hic yakalanamiyordu
                # (bkz. plan/sohbet gecmisi - asil maliyet farkinin nedeni buydu).
                
                # Spot araçlar için %10 kısıt kuralı kaldırıldı
                # if mevcut <= 0 and not is_final_slot and onerilen < 0.10 * kap:
                #     continue
                best = (onerilen, arac_turu, False)
                break
        if best is None:
            return None
        leg_plans.append((leg_src, leg_dst, leg_gun, leg_slot, *best))

    tasinabilir = min(desi, min(p[4] for p in leg_plans))
    if tasinabilir <= 0:
        return None

    legs = []
    vehicle_cost = 0.0
    for (leg_src, leg_dst, leg_gun, leg_slot, _miktar, arac_turu, is_kiralik) in leg_plans:
        vehicle_cost += state._commit_leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, tasinabilir, is_kiralik)
        legs.append(Leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, is_kiralik))

    tamamlanma = _completion_datetime(data, legs, tasinabilir)
    talep_tamamlanma = slot_datetime(demand_gun, demand_slot)  # GERCEK olusum ani (bkz. docstring)
    hedef_gun = data.route_lookup[(src, dst)]["target_delivery_days"]
    deadline = sla_deadline(talep_tamamlanma, hedef_gun)
    saat_gecikme = gecikme_saat(tamamlanma, deadline)
    sla_cost = sla_cezasi_tl(tasinabilir, saat_gecikme)

    assignment = Assignment(
        demand_hat=hat, demand_gun=demand_gun, demand_slot=demand_slot, desi=tasinabilir,
        legs=tuple(legs), sla_cost=sla_cost, vehicle_cost=vehicle_cost, talep_id=talep_id,
    )
    state.assignments.append(assignment)
    return assignment


def insertion_options(data: ProblemData, hat: tuple, gun: str, slot: str):
    """Aynı slotta denenecek yol seçeneklerini üretir: direkt + tüm 1/2-aktarmalı
    adaylar. Sıra önemli değil — `_insert_chunk` bunların HEPSİNİ deneyip
    maliyete göre en ucuzunu seçiyor (bkz. o fonksiyonun docstring'i)."""
    yield ()  # direkt
    yield from data.relay_candidates.get(hat, [])  # (r,) ya da (r1, r2) tuple'lari


def _insert_chunk(state: State, hat, gun, slot, desi, rng, talep_id) -> float:
    """Bir talep parçasını (gerekirse bölerek, gerekirse erteleyerek) tamamen
    yerleştirir. Geriye yerleştirilemeyen (garanti: normalde 0) miktarı döner.

    Aynı slotta direkt/1-aktarma/2-aktarma arasında SABİT bir öncelik sırası
    yok — hepsi (state kopyaları üzerinde) denenip, birim desi başına en
    ucuza mal olan seçilip GERÇEK state'e uygulanıyor. Hiçbiri yer bulamazsa
    (kapasite/tam), sıradaki (gün,slot)'a ertelenir (bkz. plan/sohbet geçmişi:
    "direkt/1-aktarma/2-aktarma fark etmez, en maliyetsizi bulunsun" talebi)."""
    kalan = desi
    secenekler = list(insertion_options(state.data, hat, gun, slot))

    while kalan > 1e-6:
        adaylar = []  # (-yerlesen_desi, birim_maliyet, deneme_state, assignment)
        for secenek in secenekler:
            deneme = state.copy()
            a = try_insert_path(deneme, hat, gun, slot, kalan, talep_id=talep_id, path=secenek)
            if a is not None and a.desi > 1e-9:
                birim_maliyet = (a.vehicle_cost + a.sla_cost) / a.desi
                adaylar.append((-a.desi, birim_maliyet, deneme, a))
        if not adaylar:
            break
        # ONCELIK: once en COK deseyi yerlestiren secenek (parcalanmayi onlemek
        # icin - "cok ucuz ama minik bir kirinti" bedava kiralik kapasitesini
        # her seferinde secip talebi asiri parcalayan bir bug'a yol aciyordu,
        # bkz. sohbet gecmisi). Ayni miktari yerlestiren secenekler arasinda ise
        # en ucuz (birim maliyet en dusuk) kazanir.
        adaylar.sort(key=lambda t: (t[0], t[1]))
        _, _, kazanan_state, kazanan = adaylar[0]
        state.assignments = kazanan_state.assignments
        state.unassigned = kazanan_state.unassigned
        state.leg_spot_desi = kazanan_state.leg_spot_desi
        state.leg_kiralik_desi = kazanan_state.leg_kiralik_desi
        state.handling_usage = kazanan_state.handling_usage
        state.tir_usage = kazanan_state.tir_usage
        kalan -= kazanan.desi

    if kalan > 1e-6:
        idx = state.data.zaman_sirali.index((gun, slot))
        for g2, s2 in state.data.zaman_sirali[idx + 1:]:
            if kalan <= 1e-6:
                break
            # demand_gun/demand_slot = GERCEK orijinal talep zamani (gun,slot) -
            # kalkis (g2,s2) ertelenmis olsa da SLA deadline'i buna gore hesaplanir.
            a = try_insert_path(state, hat, g2, s2, kalan, talep_id=talep_id, demand_gun=gun, demand_slot=slot)
            if a is not None:
                kalan -= a.desi
    return kalan


def force_insert(state: State, hat, gun, slot, desi, talep_id) -> None:
    """Son çare: kapasite kısıtlarını yok sayarak direkt yola zorla ekler — TÜM
    desilerin teslim edilmesini garanti eder (bkz. plan/PDF: erteleme yasağı).
    Normal koşullarda spot kapasitesi (MAX_SPOT çok yüksek) bu fonksiyona hiç
    gelinmeden yeterli olur; bu yalnızca aşırı uç senaryolar için bir emniyet ağıdır.
    """
    src, dst = hat
    data = state.data
    arac_turu = _rank_spot_types_by_cost(data, hat, desi)[0]  # bu miktar icin en ucuz tur
    key = (src, dst, gun, slot, arac_turu)
    eski = state.leg_spot_desi.get(key, 0.0)
    kap = data.arac_parametreleri[arac_turu]["kapasite_desi"]
    eski_adet = spot_vehicle_count(eski, kap, 10 ** 9)
    yeni = eski + desi
    yeni_adet = spot_vehicle_count(yeni, kap, 10 ** 9)
    delta_adet = (yeni_adet - eski_adet) 
    state.leg_spot_desi[key] = yeni
    p = data.arac_parametreleri[arac_turu]

    ellecleme_maliyet = ellecleme_maliyet_hesapla(desi, p["spot_hourly"])
    
    birim = vehicle_leg_cost(data.route_lookup, hat, arac_turu, p["spot_hourly"], p["spot_km"])
    vehicle_cost = delta_adet * birim + ellecleme_maliyet
    state.handling_usage[(src, gun)] = state.handling_usage.get((src, gun), 0.0) + desi
    varis_g = arrival_day(data.route_lookup, data.gunler, hat, gun, slot, arac_turu) or gun
    state.handling_usage[(dst, varis_g)] = state.handling_usage.get((dst, varis_g), 0.0) + desi

    leg = Leg(src, dst, gun, slot, arac_turu, False)

    #eski
    # varis = varis_zamani(slot_datetime(gun, slot), data.route_lookup[hat][arac_turu])
    # tamamlanma = ellecleme_tamamlanma_zamani(varis, desi, consolidation=False)
    
    #yeni
    tamamlanma = _completion_datetime(data, [leg], desi)
    
    deadline = sla_deadline(slot_datetime(gun, slot), data.route_lookup[hat]["target_delivery_days"])
    sla_cost = sla_cezasi_tl(desi, gecikme_saat(tamamlanma, deadline))
    state.assignments.append(
        Assignment(hat, gun, slot, desi, (leg,), sla_cost, vehicle_cost, talep_id)
    )


def dummy_initial_builder(state, rng, **kwargs):
    """
    ALNS döngüsü başlamadan ÖNCE, sistemi sıfır kapasite ihlali ile dolduran 
    özel 'Başlangıç İnşa' operatörüdür.
    
    Konsolidasyonu umursamaz. Eğer bir kargo o gün/slota sığmıyorsa, force_insert 
    yapmak yerine kargoyu zaman çizelgesinde (timeline) ileriye doğru kaydırarak 
    yasal boşluk arar.
    """
    state = state.copy()
    items = list(state.unassigned)
    state.unassigned = []
    
    # 1. Kargoları büyükten küçüğe sırala (Greedy/Açgözlü yerleştirme mantığı)
    # Büyük desileri (Örn: 5000 desi) önce yerleştirmek her zaman daha güvenlidir, 
    # küçük desiler aralara rahatça sızabilir.
    items.sort(key=lambda x: x[3], reverse=True)
    
    zamanlar = state.data.zaman_sirali  # Bütün (gun, slot) ikililerinin kronolojik listesi
    
    for item in items:
        hat, orj_gun, orj_slot, orj_desi, talep_id = item
        kalan_desi = orj_desi
        
        # Bu kargonun zaman çizelgesinde geldiği (başladığı) indeksi bul
        baslangic_idx = 0
        for idx, (g, s) in enumerate(zamanlar):
            if g == orj_gun and s == orj_slot:
                baslangic_idx = idx
                break
                
        # 2. Kargoyu yerleştirene kadar zaman çizelgesinde GELECEĞE doğru ilerle
        for idx in range(baslangic_idx, len(zamanlar)):
            aktif_gun, aktif_slot = zamanlar[idx]
            
            # _insert_chunk fonksiyonu, kargoyu 'aktif' zamana yasal sınırlar içinde yerleştirmeyi dener.
            # Yerleşen kısımlar araçlara/merkezlere atanır, FİZİKSEL OLARAK SIĞMAYAN miktar geri döner.
            kalan_desi = _insert_chunk(
                state, 
                hat, 
                aktif_gun, 
                aktif_slot, 
                kalan_desi, 
                rng, 
                talep_id
            )
            
            # Eğer tüm desi başarılı bir şekilde yerleştiyse (kalan < 0.000001)
            # Bu kargo için döngüyü kır ve sıradaki kargoya geç.
            if kalan_desi <= 1e-6:
                break
                
        # 3. KORUMA AĞI: Haftanın Sonuna Geldik ve Hâlâ Sığmadıysa
        if kalan_desi > 1e-6:
            # force_insert KULLANMIYORUZ! 
            # Sistemin toplam donanımı (haftalık kapasitesi) bile bu kargoyu kaldırmaya yetmedi demektir.
            # Bu durumda kargoyu "Hiç Teslim Edilemedi" olarak unassigned havuzuna geri atıyoruz.
            # Objective fonksiyonun unassigned kargolara zaten devasa bir SLA cezası kesiyor.
            state.unassigned.append((hat, orj_gun, orj_slot, kalan_desi, talep_id))
            
    return state


# ============================================================================
# Destroy operatörleri
# ============================================================================
def _remove_assignment(state: State, a: Assignment) -> None:
    for leg in a.legs:
        key = (leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu)
        if leg.is_kiralik:
            state.leg_kiralik_desi[key] = state.leg_kiralik_desi.get(key, 0.0) - a.desi
        else:
            eski = state.leg_spot_desi.get(key, 0.0)
            kap = state.data.arac_parametreleri[leg.arac_turu]["kapasite_desi"]
            eski_adet = spot_vehicle_count(eski, kap, MAX_SPOT)
            yeni = max(0.0, eski - a.desi)
            yeni_adet = spot_vehicle_count(yeni, kap, MAX_SPOT)
            state.leg_spot_desi[key] = yeni
            if leg.arac_turu == state.data.tir_arac_turu:
                delta = yeni_adet - eski_adet
                state.tir_usage[(leg.src, leg.gun)] = state.tir_usage.get((leg.src, leg.gun), 0) + delta
                varis_g = arrival_day(state.data.route_lookup, state.data.gunler, (leg.src, leg.dst), leg.gun, leg.slot, leg.arac_turu)
                if varis_g:
                    state.tir_usage[(leg.dst, varis_g)] = state.tir_usage.get((leg.dst, varis_g), 0) + delta
        state.handling_usage[(leg.src, leg.gun)] = state.handling_usage.get((leg.src, leg.gun), 0.0) - a.desi
        varis_g = arrival_day(state.data.route_lookup, state.data.gunler, (leg.src, leg.dst), leg.gun, leg.slot, leg.arac_turu)
        if varis_g:
            state.handling_usage[(leg.dst, varis_g)] = state.handling_usage.get((leg.dst, varis_g), 0.0) - a.desi
    state.unassigned.append((a.demand_hat, a.demand_gun, a.demand_slot, a.desi, a.talep_id))


def random_removal(state: State, rng: rnd.Generator, **kwargs) -> State:
    state = state.copy()
    if not state.assignments:
        return state
    n = max(1, int(0.1 * len(state.assignments)))
    idx = rng.choice(len(state.assignments), size=min(n, len(state.assignments)), replace=False)
    for i in sorted(idx, reverse=True):
        _remove_assignment(state, state.assignments.pop(i))
    return state

def low_occupancy_removal(state, rng, **kwargs):
    """
    Spot araçlardaki düşük doluluklu (%40'ın altı) bacakları hedef alıp,
    bu bacakları kullanan kargoları (atamaları) sistemden söken yıkıcı operatör.
    """
    state = state.copy()
    if not state.assignments:
        return state

    # 1. Her bir fiziksel bacaktaki (Leg) toplam desiyi hesapla
    # (Tıpkı çıktı dosyasındaki bucket_toplam_desi mantığı gibi)
    leg_desi_toplam = {}
    for a in state.assignments:
        for leg in a.legs:
            key = (leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu, leg.is_kiralik)
            leg_desi_toplam[key] = leg_desi_toplam.get(key, 0.0) + a.desi

    # 2. Düşük doluluklu (Örn: %40 altı) spot araç bacaklarını tespit et
    low_occ_legs = set()
    for key, toplam_desi in leg_desi_toplam.items():
        src, dst, gun, slot, arac_turu, is_kiralik = key
        
        # Sadece Spot araçları hedef alıyoruz (Kiralıklar zaten yola çıkmak zorunda)
        if not is_kiralik:
            kap = state.data.arac_parametreleri[arac_turu]["kapasite_desi"]
            
            # Bu bacak için kaç spot araç açıldığını bul
            arac_sayisi = math.ceil(toplam_desi / kap) if toplam_desi > 0 else 1
            
            # Toplam kapasitenin yüzde kaçı kullanılıyor?
            doluluk_yuzdesi = toplam_desi / (arac_sayisi * kap)
            
            # Eşik değer: %40'ın altındaysa "İsraf" olarak işaretle
            if doluluk_yuzdesi < 0.40:
                low_occ_legs.add(key)

    # 3. Kötü bacakları kullanan atamaları (assignments) bul
    candidates_to_remove = []
    for a in state.assignments:
        kullanilan_bacaklar = [(leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu, leg.is_kiralik) for leg in a.legs]
        # Eğer bu atama, israf yapan bacaklardan HERHANGİ BİRİNDEN geçiyorsa sökülecek listesine girer
        if any(bacak in low_occ_legs for bacak in kullanilan_bacaklar):
            candidates_to_remove.append(a)

    # Eğer hiç israf yapan araç yoksa (Harika durum!), state'i hiç bozmadan geri dön
    if not candidates_to_remove:
        return state

    # 4. ALNS Kuralı: Her şeyi aynı anda sökme! "Blast Radius" (Etki Alanı) belirle.
    # Tüm atamaların maksimum %15-20'sini sökmeliyiz ki motor tamamen sıfırlanıp başa sarmasın.
    max_removal_count = int(len(state.assignments) * 0.20) + 1
    num_to_remove = min(len(candidates_to_remove), max_removal_count)

    # Adaylar arasından rastgele bir kısmını seç (Böylece model her seferinde farklı kombinasyonlar dener)
    to_remove = rng.choice(candidates_to_remove, num_to_remove, replace=False)

    # 5. Seçilen atamaları State'ten çıkar ve 'unassigned' havuzuna geri at
    for a in to_remove:
        # DİKKAT: Burada senin state sınıfının içindeki sökme fonksiyonunu kullanmalısın.
        # worst_removal veya random_removal içinde hangi metot kullanılıyorsa onu çağır.
        # Genelde şu şekildedir:
        state.assignments.remove(a)
        _remove_assignment(state, a)
        
        # Talebi, Onarıcı (Repair) operatörlerin yeniden alabilmesi için unassigned listesine ekle
        state.unassigned.append((a.demand_hat, a.demand_gun, a.demand_slot, a.desi, a.talep_id))

    return state

def shaw_related_removal(state, rng, **kwargs):
    """
    Shaw (Related) Removal Yıkıcı Operatörü:
    Birbirine benzeyen (Aynı hat, aynı gün, aynı zaman dilimi) kargoları 
    hedef alarak aynı anda söker. Bu sayede repair (onarım) operatörünün 
    bunları tek bir araçta konsolide etmesini zorlar.
    """
    state = state.copy()
    if not state.assignments:
        return state

    # ALNS'nin her iterasyonda çok fazla veya çok az bozmasını engellemek için
    # toplam atamaların %10'u ile %20'si arasında bir kısmını sökeceğiz.
    min_remove = max(1, int(len(state.assignments) * 0.10))
    max_remove = max(2, int(len(state.assignments) * 0.20))
    num_to_remove = rng.integers(min_remove, max_remove + 1)
    
    # Güvenlik kontrolü
    num_to_remove = min(num_to_remove, len(state.assignments))

    # 1. TOHUM (Seed) SEÇİMİ: Rastgele bir atama seçiyoruz
    seed = rng.choice(state.assignments)
    
    # 2. BENZERLİK SKORLAMASI: Diğer tüm kargoların tohuma ne kadar benzediğini hesapla
    similarities = []
    for a in state.assignments:
        if a == seed:
            continue
            
        score = 0
        # Rota Benzerliği
        if a.demand_hat == seed.demand_hat:
            score += 10  # Birebir aynı rota ise devasa skor
        else:
            if a.demand_hat[0] == seed.demand_hat[0]: 
                score += 3 # Sadece çıkış noktası aynı
            if a.demand_hat[1] == seed.demand_hat[1]: 
                score += 3 # Sadece varış noktası aynı
                
        # Zaman Benzerliği
        if a.demand_gun == seed.demand_gun:
            score += 5 # Aynı gün
            if a.demand_slot == seed.demand_slot:
                score += 2 # Aynı gün ve aynı slot (Mükemmel konsolidasyon adayı)
                
        similarities.append((score, a))
        
    # 3. SIRALAMA: En çok benzeyenler (skoru en yüksek olanlar) en başa gelsin
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Sadece kargo objelerini bir listeye alalım
    candidates = [x[1] for x in similarities]
    to_remove = [seed]
    
    # 4. DETERMINIZM KIRICI (Randomization):
    # Eğer her seferinde en yüksek skorluyu kesin olarak alırsak algoritma kısır döngüye girer.
    # Klasik ALNS literatüründeki (y^p) kuralını uygulayarak, yüksek skorluları daha YÜKSEK İHTİMALLE,
    # düşük skorluları daha DÜŞÜK İHTİMALLE seçecek bir yapı kuruyoruz. (p=3 veya p=4 idealdir)
    p = 3 
    
    while len(to_remove) < num_to_remove and candidates:
        # rng.random() 0 ile 1 arası üretir. p. kuvvetini alınca 0'a çok yakınsar.
        # Bu da listenin başındaki (en benzer) elemanların seçilme ihtimalini aşırı artırır.
        idx = int(len(candidates) * (rng.random() ** p))
        
        # Güvenlik amaçlı indeks taşmasını engelle
        if idx >= len(candidates): 
            idx = len(candidates) - 1
            
        # Seçilen adayı listeden kopar ve silinecekler listesine ekle
        to_remove.append(candidates.pop(idx))

    # 5. SÖKME (Removal) İŞLEMİ: Senin verdiğin yapıya tam uygun olarak
    for a in to_remove:
        state.assignments.remove(a)
        _remove_assignment(state, a)
        
        # Sökülen kargoyu yeniden atanmak (repair) üzere unassigned havuzuna gönder
        state.unassigned.append((a.demand_hat, a.demand_gun, a.demand_slot, a.desi, a.talep_id))

    return state

def worst_removal(state: State, rng: rnd.Generator, **kwargs) -> State:
    state = state.copy()
    if not state.assignments:
        return state
    n = max(1, int(0.1 * len(state.assignments)))
    ranked = sorted(state.assignments, key=lambda a: -(a.sla_cost + a.vehicle_cost))
    for a in ranked[:n]:
        state.assignments.remove(a)
        _remove_assignment(state, a)
    return state


def tm_overload_removal(state: State, rng: rnd.Generator, **kwargs) -> State:
    state = state.copy()
    aday_tmler = [
        tm
        for tm in state.data.merkezler
        for gun in state.data.gunler
        if state.data.handling_capacity.get(tm) is not None
        and state.handling_usage.get((tm, gun), 0.0) > 0.85 * state.data.handling_capacity[tm]
    ]
    if not aday_tmler:
        return random_removal(state, rng, **kwargs)
    tm = aday_tmler[int(rng.integers(0, len(aday_tmler)))]
    to_remove = [a for a in state.assignments if any(leg.src == tm or leg.dst == tm for leg in a.legs)]
    rng.shuffle(to_remove)
    for a in to_remove[: max(1, len(to_remove) // 3)]:
        state.assignments.remove(a)
        _remove_assignment(state, a)
    return state


# ============================================================================
# Repair operatörleri
# ============================================================================
# def greedy_repair(state: State, rng: rnd.Generator, **kwargs) -> State:
#     state = state.copy()
#     items = list(state.unassigned)
#     state.unassigned = []
#     order = rng.permutation(len(items)) if items else [] # Rastgele karıştırıyor kendi içinde.
#     for i in order:
#         hat, gun, slot, desi, talep_id = items[i]
#         kalan = _insert_chunk(state, hat, gun, slot, desi, rng, talep_id)
#         if kalan > 1e-6: # Eğer kalan(yerleştirilemeyen) desi miktarı 10^-6'dan büyükse
#             force_insert(state, hat, gun, slot, kalan, talep_id) # Zorla yerleştir (TM Elleçleme, tır, kapasite kısıtlarını hiçe say.)
#     return state


def greedy_repair(state: State, rng: rnd.Generator, **kwargs) -> State:
    """
    Kapasite ihlali KESİNLİKLE yapmayan, sığmayan kargoları zamanda ileri 
    kaydıran ve hâlâ sığmıyorsa havuzda (unassigned) bırakan yeni onarıcı.
    """
    state = state.copy()
    items = list(state.unassigned)
    state.unassigned = []
    
    # Kargo sırasını rastgele karıştır ki her iterasyonda farklı bir rota ağacı oluşsun
    order = rng.permutation(len(items)) if items else [] 
    zamanlar = state.data.zaman_sirali
    
    for i in order:
        hat, orj_gun, orj_slot, orj_desi, talep_id = items[i]
        kalan_desi = orj_desi
        
        # Orijinal kalkış anının indeksini bul
        baslangic_idx = 0
        for idx, (g, s) in enumerate(zamanlar):
            if g == orj_gun and s == orj_slot:
                baslangic_idx = idx
                break
                
        # Zaman çizelgesinde geleceğe doğru boşluk ara
        for idx in range(baslangic_idx, len(zamanlar)):
            aktif_gun, aktif_slot = zamanlar[idx]
            
            kalan_desi = _insert_chunk(
                state, hat, aktif_gun, aktif_slot, kalan_desi, rng, talep_id
            )
            
            # Kargo tamamen yerleştiyse aramayı bırak
            if kalan_desi <= 1e-6:
                break
                
        # Tüm haftayı taradık ama hâlâ sığmadıysa
        if kalan_desi > 1e-6:
            # DİKKAT: force_insert İPTAL EDİLDİ! 
            # Bunun yerine kargo "atanamadı" olarak havuzda kalır.
            state.unassigned.append((hat, orj_gun, orj_slot, kalan_desi, talep_id))
            
    return state



def cpsat_hat_repair(state: State, rng: rnd.Generator, **kwargs) -> State:
    """Tek bir hat için (unassigned içindeki en çok parçaya sahip hat), o hattın
    tüm (gün,slot) atamasını küçük bir CP-SAT modeliyle YENİDEN VE TAM OPTİMAL
    çözer — diğer hatların o an kullandığı elleçleme/tır kapasitesini sabit kabul
    ederek geri kalan (paylaşılan) kapasiteyi kısıt olarak kullanır. Diğer
    hatlara ait unassigned parçalar varsa greedy_repair'e bırakılır.
    """
    state = state.copy()
    # Atanmamış kargo yoksa direkt fonksiyonu bitir.
    if not state.unassigned:
        return state

    hat_counts: dict = {}
    for (hat, *_rest) in state.unassigned: # _rest listesinin unassigned tuple'ındaki kullanmayacağımız şeyleri attık. 
        hat_counts[hat] = hat_counts.get(hat, 0) + 1
    target_hat = max(hat_counts, key=hat_counts.get)

    hat_items = [it for it in state.unassigned if it[0] == target_hat]
    other_items = [it for it in state.unassigned if it[0] != target_hat]
    state.unassigned = other_items

    data = state.data
    src, dst = target_hat
    talep = {}
    talep_queue_by_gs = {}

    for (_, gun, slot, desi, tid) in hat_items:
        key = (gun, slot)
        talep[key] = talep.get(key, 0.0) + desi
        talep_queue_by_gs.setdefault(key, []).append([tid, float(desi), gun, slot])

    active_talep_queue = []

    def take_from_active_queue(miktar):
        pieces = []
        remaining = float(miktar)

        while remaining > 1e-6 and active_talep_queue:
            tid, available, demand_gun, demand_slot = active_talep_queue[0]
            take = min(float(available), remaining)

            if take > 1e-6:
                pieces.append((tid, take, demand_gun, demand_slot))

            available = float(available) - take
            remaining -= take

            if available <= 1e-6:
                active_talep_queue.pop(0)
            else:
                active_talep_queue[0][1] = available

        return pieces
    model = cp_model.CpModel()
    zaman_sirali = data.zaman_sirali
    max_talep = max(1, int(round(sum(talep.values()))))

    kiralik_x, spot_y, ert, bir = {}, {}, {}, {}
    for (g, s) in zaman_sirali:
        for a in data.arac_turleri:
            stok = data.kiralik_stok_gunluk.get((target_hat, a), 0) if s == KIRALIK_DISPATCH_SLOT else 0
            kiralik_x[(g, s, a)] = model.NewIntVar(0, stok, f"kx_{g}_{s}_{a}")
            spot_y[(g, s, a)] = model.NewIntVar(0, MAX_SPOT, f"sy_{g}_{s}_{a}")
        ert[(g, s)] = model.NewIntVar(0, max_talep, f"ert_{g}_{s}")
        bir[(g, s)] = model.NewIntVar(0, max_talep, f"bir_{g}_{s}")

    # Paylasimli elleceleme kapasitesi: (TM, gun) basina TUM slotlarin toplam
    # katkisi, o an diger hatlarin kullandigi miktar dusuldukten sonra kalan
    # paya sigmali. Once tum slotlarin yuk terimlerini (TM,gun) bazinda topluyoruz,
    # sonra TEK bir kisit ekliyoruz (slot slot ayri kisitlamak yanlis olurdu -
    # ayni gunun iki slotu ayni kapasiteyi paylasir).

    yuk_terimleri_by_slot = {}
    handling_terimleri_by_tm_gun: dict = {}
    yuk_dict = {} # YENİ EKLENEN SÖZLÜK
    for idx, (g, s) in enumerate(zaman_sirali):
        bugun = int(round(talep.get((g, s), 0.0)))
        if idx == 0:
            model.Add(bir[(g, s)] == bugun)
        else:
            g0, s0 = zaman_sirali[idx - 1]
            model.Add(bir[(g, s)] == ert[(g0, s0)] + bugun)

        yuk_terimleri = []
        for a in data.arac_turleri:
            kap = int(data.arac_parametreleri[a]["kapasite_desi"])
            yuk = model.NewIntVar(0, (MAX_SPOT + 50) * kap, f"yuk_{g}_{s}_{a}")
            yuk_dict[(g, s, a)] = yuk # YENİ EKLENEN SATIR: Değişkeni aşağısı için hafızaya alıyoruz
            model.Add(yuk <= (kiralik_x[(g, s, a)] + spot_y[(g, s, a)]) * kap)
            yuk_terimleri.append(yuk)
            varis_g = arrival_day(data.route_lookup, data.gunler, target_hat, g, s, a) or g
            handling_terimleri_by_tm_gun.setdefault((src, g), []).append(yuk)
            handling_terimleri_by_tm_gun.setdefault((dst, varis_g), []).append(yuk)
        yuk_terimleri_by_slot[(g, s)] = yuk_terimleri
        model.Add(bir[(g, s)] == cp_model.LinearExpr.Sum(yuk_terimleri) + ert[(g, s)])

    # Diger hatlarin MEVCUT kullanimi sabit kabul edilip kalan pay bu hatta
    # (tum slotlarin TOPLAMI uzerinden, gun bazinda) kisitlaniyor.
    for (tm, gun_for_cap), terimler in handling_terimleri_by_tm_gun.items():
        if data.handling_capacity.get(tm) is None:
            continue
        kalan_kapasite = state.handling_available(tm, gun_for_cap)
        model.Add(cp_model.LinearExpr.Sum(terimler) <= int(kalan_kapasite))

    idx_son = len(zaman_sirali) - 1
    model.Add(ert[zaman_sirali[idx_son]] == 0)

    maliyet = []
    for a in data.arac_turleri:
        p = data.arac_parametreleri[a]
        # Seyir maliyetini senin güncellediğin vehicle_leg_cost fonksiyonundan alıyoruz
        spot_birim_maliyet = vehicle_leg_cost(data.route_lookup, target_hat, a, p["spot_hourly"], p["spot_km"])
        
        # CP-SAT sadece tamsayı kabul ettiği için katsayıyı yuvarlıyoruz
        ellecleme_katsayisi = int(round((0.01 / 60) * p["spot_hourly"])) # desi başına elleçleme maliyeti 
        
        for (g, s) in zaman_sirali:
            # 1. Spot aracın yola çıkma (seyir) maliyeti
            maliyet.append(spot_y[(g, s, a)] * spot_birim_maliyet)
            
            # 2. O araca binen yükün (bilinmeyen değişkenin) elleçleme maliyeti
            if ellecleme_katsayisi > 0:
                maliyet.append(yuk_dict[(g, s, a)] * ellecleme_katsayisi)
    for idx, (g, s) in enumerate(zaman_sirali):
        if idx + 1 >= len(zaman_sirali):
            continue
        g2, s2 = zaman_sirali[idx + 1]
        saat_farki = (24 if g2 != g else 0) + slot_to_hour(s2) - slot_to_hour(s)
        katsayi = int(round(sla_cezasi_tl(1.0, float(saat_farki))))
        if katsayi > 0:
            maliyet.append(ert[(g, s)] * katsayi)
    model.Minimize(cp_model.LinearExpr.Sum(maliyet))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # CP-SAT basarisiz olursa greedy'e devret (nadir - kucuk model, olmasi beklenmez).
        state.unassigned = other_items + hat_items
        return greedy_repair(state, rng, **kwargs)

    # CP-SAT'in onerdigi (g,s,a) dagilimini, State'in GUVENLI (kapasiteyi kontrol
    # eden) commit yolundan gecirerek uygula. Boylece bu operator hicbir zaman
    # elleceleme/tir kapasitesini asan bir durum uretemez - CP-SAT'in kisiti
    # yaklasik/gevsek kalsa bile nihai commit her zaman clamp'lidir. Karsilanamayan
    # artik miktar, diger repair operatorlerinin (greedy/force) isleyecegi sekilde
    # unassigned'a geri konur.
    for (g, s) in zaman_sirali:
        active_talep_queue.extend(talep_queue_by_gs.get((g, s), []))
        slotta_tasinan_yuk = max(0.0, float(solver.Value(bir[(g, s)]) - solver.Value(ert[(g, s)])))
        remaining_slot_load = slotta_tasinan_yuk

        for a in data.arac_turleri:
            if remaining_slot_load <= 1e-6:
                break

            k_adet = solver.Value(kiralik_x[(g, s, a)])
            s_adet = solver.Value(spot_y[(g, s, a)])
            if k_adet <= 0 and s_adet <= 0:
                continue

            kap = data.arac_parametreleri[a]["kapasite_desi"]
            toplam_yuk = min(
                remaining_slot_load,
                float(solver.Value(yuk_dict[(g, s, a)])),
                (k_adet + s_adet) * kap,
            )

            if k_adet > 0 and toplam_yuk > 0:
                istenen = min(toplam_yuk, k_adet * kap)
                miktar = min(istenen, state.max_addable_on_leg(src, dst, g, s, a, True))
                if miktar > 0:
                    state._commit_leg(src, dst, g, s, a, miktar, True)
                    leg = Leg(src, dst, g, s, a, True)
                    for tid, piece_desi, demand_gun, demand_slot in take_from_active_queue(miktar):
                        piece_tamamlanma = _completion_datetime(data, [leg], piece_desi)
                        piece_deadline = sla_deadline(slot_datetime(demand_gun, demand_slot), data.route_lookup[target_hat]["target_delivery_days"])
                        piece_sla_cost = sla_cezasi_tl(piece_desi, gecikme_saat(piece_tamamlanma, piece_deadline))
                        state.assignments.append(
                            Assignment(target_hat, demand_gun, demand_slot, piece_desi, (leg,), piece_sla_cost, 0.0, tid)
                        )
                    toplam_yuk -= miktar
                    remaining_slot_load -= miktar

            if s_adet > 0 and toplam_yuk > 0:
                istenen = min(toplam_yuk, s_adet * kap)
                miktar = min(istenen, state.max_addable_on_leg(src, dst, g, s, a, False))
                if miktar > 0:
                    vehicle_cost = state._commit_leg(src, dst, g, s, a, miktar, False)
                    leg = Leg(src, dst, g, s, a, False)
                    for tid, piece_desi, demand_gun, demand_slot in take_from_active_queue(miktar):
                        oran = piece_desi / miktar if miktar else 0.0
                        piece_vehicle_cost = vehicle_cost * oran
                        piece_tamamlanma = _completion_datetime(data, [leg], piece_desi)
                        piece_deadline = sla_deadline(slot_datetime(demand_gun, demand_slot), data.route_lookup[target_hat]["target_delivery_days"])
                        piece_sla_cost = sla_cezasi_tl(piece_desi, gecikme_saat(piece_tamamlanma, piece_deadline))
                        state.assignments.append(
                            Assignment(target_hat, demand_gun, demand_slot, piece_desi, (leg,), piece_sla_cost, piece_vehicle_cost, tid)
                        )
                    toplam_yuk -= miktar
                    remaining_slot_load -= miktar

            if toplam_yuk > 1e-6:
                for tid, piece_desi, demand_gun, demand_slot in take_from_active_queue(toplam_yuk):
                    kalan2 = _insert_chunk(state, target_hat, demand_gun, demand_slot, piece_desi, rng, tid)
                    if kalan2 > 1e-6:
                        force_insert(state, target_hat, demand_gun, demand_slot, kalan2, tid)
                remaining_slot_load -= toplam_yuk

    # Bu operator SADECE target_hat'i CP-SAT ile cozdu; destroy birden fazla
    # hattan parca kaldirmis olabilir - digerleri (other_items) hala
    # state.unassigned'da bekliyor olabilir. Bu fonksiyon da (greedy_repair gibi)
    # HER ZAMAN tam teslim garanti etmeli - kalanlari greedy sekilde yerlestiriyoruz.
    kalan_items = list(state.unassigned)
    state.unassigned = []
    for (hat2, gun2, slot2, desi2, talep_id2) in kalan_items:
        kalan2 = _insert_chunk(state, hat2, gun2, slot2, desi2, rng, talep_id2)
        if kalan2 > 1e-6:
            force_insert(state, hat2, gun2, slot2, kalan2, talep_id2)

    return state
