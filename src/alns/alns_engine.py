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
    DEMAND_ARRIVAL_TIMES,
    RouteLookup,
    arrival_day,
    ellecleme_tamamlanma_zamani,
    gecikme_saat,
    ellecleme_suresi_dakika,
    _ellecleme_dagilimi,
    seyir_suresi_saat,
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
MILK_RUN_ENABLED = True        # milk_run, ayni HUB bacagini paylasip farkli yonlere
                                # ayrisan taleplerin her birini BAGIMSIZ degerlendirdigi
                                # icin, fiziksel olarak imkansiz sekilde HEPSINDE HUB
                                # elleclemesini sifirlayabiliyordu (capraz-talep yon
                                # tutarliligi kontrolu yoktu). Artik State.milk_fwd/
                                # milk_bwd kilidi (bkz. o alanlarin docstring'i,
                                # evaluate_path._milk_junction_ok, commit_path,
                                # _remove_assignment) bir HUB bacaginin AYNI ANDA
                                # sadece TEK bir yone elleclenmeden devam etmesine izin
                                # veriyor - yon celismesi olan durumlar milk_run adayligindan
                                # otomatik elenir (gercek elleclemeye geri duser).

# Spot arac icin minimum kalkis doluluk orani (CP-SAT'taki (src/optimization.py)
# "%10 doluluk kurali" ile ayni mantik). Burada ALNS'te KESIN (hard, parasal
# olmayan) bir kisit olarak uygulanir - bkz. enforce_min_spot_occupancy().
MIN_SPOT_DOLULUK_ORANI = 0.10

# CP-SAT sadece EN SON slotu tamamen muaf tutuyor (tek "ucurum"). ALNS'te bunu
# AYNEN kopyalamak, tum ay biriken talebi TEK bir son slota sikistiriyor -
# gercek elleçleme/tir kapasitesini asip demandin "yerlestirilemeyen talep"
# olarak KAYBOLMASINA yol aciyordu (bkz. sohbet gecmisi: 265 satir / 12.366
# desi / 3.56M TL sanal ceza bulgusu). Bunun yerine son
# MIN_SPOT_DOLULUK_TAPER_GUN_SAYISI gun boyunca esik, tam orandan (taper
# penceresine girerken) 0'a (en son slotta) DOGRUSAL olarak azalir - boylece
# arama, birikmis talebi son birkac gune YAYARAK bosaltabilir, tek bir slota
# tikanmak zorunda kalmaz.
MIN_SPOT_DOLULUK_TAPER_GUN_SAYISI = 2


_hat_toplam_talep_cache: dict = {}


def _hat_toplam_talep(demands) -> dict:
    """Her hat (src,dst) icin, TUM ufuktaki (data.demands - statik, ALNS
    calismasi boyunca degismez) toplam talep desisini dondurur. Bu, bir
    hat'in konsolidasyonla ULASABILECEGI TEORIK TAVANI temsil eder -
    dinamik alt limit (bkz. _min_doluluk_esigi'nin ust_sinir parametresi)
    bunu kullanarak dusuk hacimli hatlarin imkansiz bir esigi kovalamasini
    engeller. id(demands) ile cache'leniyor (diger benzer cache'lerle ayni
    desen - bkz. arrival_day)."""
    cache_key = id(demands)
    if cache_key in _hat_toplam_talep_cache:
        return _hat_toplam_talep_cache[cache_key]
    toplam: dict = {}
    for (hat, _gun, _slot, desi, _talep_id) in demands:
        toplam[hat] = toplam.get(hat, 0.0) + desi
    _hat_toplam_talep_cache[cache_key] = toplam
    return toplam


def _min_doluluk_esigi(idx: int, toplam_slot: int, ust_sinir: float = 1.0) -> float:
    """zaman_sirali[idx] icin uygulanacak minimum doluluk esigini dondurur.
    Son MIN_SPOT_DOLULUK_TAPER_GUN_SAYISI gun disinda MIN_SPOT_DOLULUK_ORANI
    (sabit); taper penceresi icinde, en son slota (esik=0) dogru DOGRUSAL
    olarak azalir - bkz. MIN_SPOT_DOLULUK_TAPER_GUN_SAYISI tanimindaki not.

    DINAMIK ALT LIMIT: `ust_sinir`, bu hat icin FIILEN ulasilabilir maksimum
    doluluk oranidir (bkz. _hat_toplam_talep). Dusuk hacimli bir hatta TUM
    ufuktaki talep toplansa bile MIN_SPOT_DOLULUK_ORANI'na (%10) ulasilamiyorsa,
    esik bu ust_sinir'a KISILIR - aksi halde kural, hicbir zaman
    saglanamayacak bir esigi kovalayip mikro talebi ufkun sonuna kadar
    erteler ve gereksiz SLA cezasi biriktirir (bkz. sohbet gecmisi)."""
    taper_slot_sayisi = MIN_SPOT_DOLULUK_TAPER_GUN_SAYISI * len(DEMAND_ARRIVAL_TIMES)
    if taper_slot_sayisi <= 0:
        return min(MIN_SPOT_DOLULUK_ORANI, ust_sinir)
    kalan_slot = toplam_slot - 1 - idx  # en son slotta 0
    if kalan_slot >= taper_slot_sayisi:
        return min(MIN_SPOT_DOLULUK_ORANI, ust_sinir)
    return min(MIN_SPOT_DOLULUK_ORANI, ust_sinir) * (kalan_slot / taper_slot_sayisi)


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
    def is_valid_relay(d_src_dst,d_src_x,d_x_dst):
        # Optimizasyon amaçlı bir TM'nin ara TM olarak relay candidate seçilmesi için şu formül kullanılacak:
        result = d_src_x + d_x_dst <= d_src_dst * 1.35 # 1.35 degeri çok tahmini. Bu değer değiştirilebilir.
        return result
    candidates: dict = {}
    for hat in hatlar:
        src, dst = hat
        tek_aktarma = []
        leg_src_dst = route_lookup.get((src,dst))
        for relay in merkezler:
            if relay in (src, dst):
                continue
            leg_src_r = route_lookup.get((src, relay))
            leg_r_dst = route_lookup.get((relay, dst))
            
            d_src_r = leg_src_r["distance_km"]
            d_r_dst = leg_r_dst["distance_km"]
            d_src_dst = leg_src_dst["distance_km"]
            if leg_src_r is not None and leg_r_dst is not None and is_valid_relay(d_src_dst=d_src_dst, d_src_x=d_src_r, d_x_dst=d_r_dst):
                tek_aktarma.append((d_src_r + d_r_dst, relay))
            

        #   def isimsiz_fonksiyon(t):
        #       return t[0]
        #   lambda t: t[0]
        # ikisi aynı işi yapar.

        tek_aktarma.sort(key=lambda t: t[0])
        en_iyi_tek = [r for _, r in tek_aktarma[:MAX_RELAY_CANDIDATES]] # En iyi 4 ugrama adayi

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
    milk_run: bool = False  # True ise: legs zinciri ugrama (ayni arac, ara duraklarda
    # elleçleme YOK) - False (varsayilan) ise mevcut konsolidasyon davranisi (her ara
    # durakta indir+yeniden yukle). Sadece coklu bacakli (len(legs)>=2) atamalarda
    # anlamli - bkz. evaluate_path/commit_path/_remove_assignment.


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
        # (src,dst,gun,slot,arac_turu,dokunus_sayisi) -> desi. leg_spot_desi/leg_kiralik_desi'den
        # AYRI tutuluyor cunku onlar TOPLAM fiziksel desiyi (arac sayisi/kapasite icin, dokunustan
        # BAGIMSIZ) izlerken, bu sozluk ayni bacagi (ayni TM cifti/gun/slot/arac turu) FARKLI
        # dokunus sayisiyla kullanan (mesela normal bir sevkiyat 2x, ugramadan gecen bir sevkiyat
        # 0x/1x) atamalarin elleçleme maliyetini KARISTIRMADAN ayri ayri hesaplamasini saglar -
        # bkz. _commit_leg. Normal (ugramasiz) her atama HER ZAMAN dokunus_sayisi=2 kullandigi
        # icin, ugrama hic kullanilmadigi surece bu, leg_spot_desi/leg_kiralik_desi ile birebir
        # ayni degerleri tutar (davranis degismez).
        self.leg_ellecleme_desi: dict = {}
        self.handling_usage: dict = {}     # (tm,gun) -> desi
        self.tir_usage: dict = {}          # (tm,gun) -> adet (spot kaynakli, kiralik ayrica sabit)
        # UGRAMA (milk_run) YON-TUTARLILIGI KILIDI: bir spot_key = (src,dst,gun,slot,arac_turu)
        # bacagi HUB'da elleclenmeden devam ediyorsa (skip_dst/skip_src), bu bacagin TEK bir
        # sonraki/onceki bacakla eslenmesini zorunlu kilar. Boylece ayni HUB bacagini paylasan
        # FARKLI taleplerin her biri BAGIMSIZ olarak "ayni arac benim yonume devam ediyor"
        # diyip HUB elleclemesini sifirlayamaz - fiziksel olarak TEK bir arac ayni anda iki
        # farkli yone devam edemeyeceginden, bir bacak zaten BASKA bir yone kilitliyse yeni bir
        # yone milk_run ile devam etmek REDDEDILIR (bkz. evaluate_path/commit_path/_remove_assignment).
        self.milk_fwd: dict = {}           # spot_key -> spot_key (bu bacak, HANGI sonraki bacaga elleclenmeden devam ediyor)
        self.milk_bwd: dict = {}           # spot_key -> spot_key (bu bacak, HANGI onceki bacaktan elleclenmeden geldi)
        self.milk_junction_desi: dict = {} # (spot_key, spot_key) -> bu J->K kilidine dayanan toplam desi (0'a inince kilit acilir)
        self.tir_usage_in: dict = {}    # (tm,gun) -> varış yapan spot tır adet
        self.tir_usage_out: dict = {}   # (tm,gun) -> çıkış yapan spot tır adet
        # Gercek-zaman kapasite dogrulamasi icin: bu ucta GERCEKTEN elleclenen
        # (skip_src/skip_dst olmayan) desi - leg_ellecleme_desi'nin dokunus_sayisi=1
        # durumunda cikis/varis ayrimini kaybetmesinden farkli olarak ayri tutulur.
        self.cikis_ellecleme_desi: dict = {}  # (src,dst,gun,slot,arac_turu,is_kiralik) -> desi
        self.varis_ellecleme_desi: dict = {}  # (src,dst,gun,slot,arac_turu,is_kiralik) -> desi
        self._fixed_kiralik_cost = self._kiralik_bos_seyir_maliyeti()
        # Her bacagin GUNCEL arac maliyetinin (seyir+ellecleme) ARTIMLI takibi -
        # _commit_leg/force_insert eklerken, _remove_assignment cikarirken bunu
        # gunceller. objective() bu sayede O(1) okuyor, her cagrida TUM bacaklari
        # yeniden taramak zorunda kalmiyor (performans), ama HALA guncel/dogru
        # kaliyor (silme sirasinda da dogru dusuruldugu icin bayatlamiyor).
        self._arac_maliyeti_toplam: float = 0.0

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
        new.leg_ellecleme_desi = dict(self.leg_ellecleme_desi)
        new.handling_usage = dict(self.handling_usage)
        new.tir_usage = dict(self.tir_usage)
        new.milk_fwd = dict(self.milk_fwd)
        new.milk_bwd = dict(self.milk_bwd)
        new.milk_junction_desi = dict(self.milk_junction_desi)
        new.tir_usage_in = dict(self.tir_usage_in)
        new.tir_usage_out = dict(self.tir_usage_out)
        new.cikis_ellecleme_desi = dict(self.cikis_ellecleme_desi)
        new.varis_ellecleme_desi = dict(self.varis_ellecleme_desi)
        new._fixed_kiralik_cost = self._fixed_kiralik_cost
        new._arac_maliyeti_toplam = self._arac_maliyeti_toplam
        return new

    def objective(self) -> float:
        # DUZELTME: arac maliyeti artik assignment'larin kendi (potansiyel
        # olarak BAYAT/stale) vehicle_cost degerlerinden degil, _arac_maliyeti_
        # toplam adli ARTIMLI takip edilen bir alandan okunuyor - bu alan
        # _commit_leg/force_insert eklerken, _remove_assignment cikarirken
        # dogru sekilde guncelleniyor (bkz. o fonksiyonlar). Eski yontemde bir
        # talep silindiginde kapasite takibi guncelleniyordu ama KALAN
        # taleplerin kayitli vehicle_cost'u guncellenmiyordu - bu da ALNS'in
        # arama sirasinda gercek olmayan (bayat) bir maliyet sinyaline gore
        # karar vermesine yol aciyordu (bkz. sohbet gecmisi). Tum bacaklari
        # her cagrida yeniden taramak (ilk denenen duzeltme) DOGRU ama COK
        # YAVASTI (ALNS'in iterasyon sayisini ciddi dusurdu) - bu yuzden
        # artimli takibe gecildi: hem dogru hem O(1).
        total = self._fixed_kiralik_cost + self._arac_maliyeti_toplam

        # DUZELTME: a.sla_cost (donmus) yerine _fresh_sla_cost - bkz. o
        # fonksiyonun docstring'i. Boylece arama, ayni bacaga sonradan
        # binen yukun gecikmeyi buyuttugu durumlari da gorebiliyor.
        for a in self.assignments:
            total += _fresh_sla_cost(self, a)
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
                    spot_in = self.tir_usage_in.get((tm, gun), 0)
                    spot_out = self.tir_usage_out.get((tm, gun), 0)
                    # DUZELTME: genel havuzda gelen+giden TOPLANIR (KISITLAR.md madde 21).
                    # Ayni aracin devam ettigi (milk_run) noktalarda 1 birim indirimi
                    # zaten _commit_leg/_remove_assignment'ta kaynakta uygulaniyor -
                    # bu yuzden burada ayrica max() almaya gerek yok/yanlis olur.
                    spot_kullanim = spot_in + spot_out
                    kullanim = spot_kullanim + self.data.fixed_kiralik_tir_usage.get((tm, gun), 0)
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
        spot_in = self.tir_usage_in.get((tm, gun), 0)
        spot_out = self.tir_usage_out.get((tm, gun), 0)
        return max(0.0, cap - fixed - (spot_in + spot_out))
    
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
    def _commit_leg(self, src, dst, gun, slot, arac_turu, desi, is_kiralik,
                     skip_src_handling: bool = False, skip_dst_handling: bool = False) -> float:
        """skip_src_handling/skip_dst_handling: SADECE uğrama (milk-run) zincirlerinde
        True olur - bu bacağın çıkışı/varışı, aslında zincirin bir ÖNCEKİ/SONRAKİ
        bacağıyla aynı fiziksel aracın devam ettiği bir ara durak olduğu için o uçta
        hiç elleçleme yapılmaz (bkz. evaluate_path/commit_path). Normal (uğramasız)
        her çağrıda ikisi de False - davranış tamamen değişmeden kalır."""
        key = (src, dst, gun, slot, arac_turu)
        p = self.data.arac_parametreleri[arac_turu]

        # 1. Kiralık ve Spot saatlik/km ücretlerini belirle
        hourly_rate = p["rental_hourly"] if is_kiralik else p["spot_hourly"]
        km_rate = p["rental_km"] if is_kiralik else p["spot_km"]

        # dokunus_sayisi: bu SPESIFIK atamanin bu bacakta faturalandiracagi elleçleme
        # ucu sayisi (0,1,2). leg_ellecleme_desi'de AYRI (dokunus_sayisi'ne gore) bir
        # alt-kovada tutuluyor ki ayni fiziksel bacagi (ayni TM cifti/gun/slot/arac
        # turu) FARKLI dokunus sayisiyla kullanan atamalar (normal 2x, ugramadan gecen
        # 0x/1x) birbirinin elleçleme maliyetini KARISTIRMASIN - araç sayisi/kapasite
        # icin kullanilan leg_spot_desi/leg_kiralik_desi ise HER ZAMAN TOPLAM (dokunustan
        # bagimsiz) fiziksel desiyi tutar, çünkü kapasite/araç sayısı dokunuşa bakmaz.
        dokunus_sayisi = (0 if skip_src_handling else 1) + (0 if skip_dst_handling else 1)
        ellec_key = key + (dokunus_sayisi,)

        # 2. O bacağın o anki TOPLAM faturasını hesaplayan yerel formül
        def bacak_toplam_maliyeti(mevcut_arac_sayisi, mevcut_ellec_desi):
            birim = vehicle_leg_cost(self.data.route_lookup, (src,dst) , arac_turu, hourly_rate, km_rate)
            seyir_faturasi = mevcut_arac_sayisi * birim
            ellecleme_faturasi = ellecleme_maliyet_hesapla(mevcut_ellec_desi, hourly_rate, dokunus_sayisi)
            return seyir_faturasi + ellecleme_faturasi

        eski_ellec_desi = self.leg_ellecleme_desi.get(ellec_key, 0.0)
        yeni_ellec_desi = eski_ellec_desi + desi
        self.leg_ellecleme_desi[ellec_key] = yeni_ellec_desi

        if is_kiralik:
            eski_desi = self.leg_kiralik_desi.get(key, 0.0)
            yeni_desi = eski_desi + desi
            self.leg_kiralik_desi[key] = yeni_desi

            # Kiralık aracın seyir maliyeti baştan ödendi!
            # Bize sadece bu desiyi yüklemek/indirmek için harcanan zamanın maliyeti yansır.

            eski_ellecleme = ellecleme_maliyet_hesapla(eski_ellec_desi, p["rental_hourly"], dokunus_sayisi)
            yeni_ellecleme = ellecleme_maliyet_hesapla(yeni_ellec_desi, p["rental_hourly"], dokunus_sayisi)
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
            eski_maliyet = bacak_toplam_maliyeti(eski_adet, eski_ellec_desi)
            yeni_maliyet = bacak_toplam_maliyeti(yeni_adet, yeni_ellec_desi)
            marjinal_maliyet = yeni_maliyet - eski_maliyet

            if arac_turu == self.data.tir_arac_turu:
                delta_adet = yeni_adet - eski_adet
                # KISITLAR.md madde 22: ayni aracin ayni TM'de hareket etmeden
                # devam etmesi (milk_run) 1 birim sayilir - bu bacak bir onceki
                # bacaktan (skip_src_handling) devam ediyorsa, o TM'nin "giden"
                # tarafi zaten onceki bacagin "gelen"iyle karsilanmis demektir,
                # burada AYRICA saymiyoruz.
                if not skip_src_handling:
                    self.tir_usage_out[(src, gun)] = self.tir_usage_out.get((src, gun), 0) + delta_adet
                varis_g = arrival_day(self.data.route_lookup, self.data.gunler, (src, dst), gun, slot, arac_turu)
                if varis_g:
                    self.tir_usage_in[(dst, varis_g)] = self.tir_usage_in.get((dst, varis_g), 0) + delta_adet

        self._arac_maliyeti_toplam += marjinal_maliyet

        # if not skip_src_handling:
        #     self.handling_usage[(src, gun)] = self.handling_usage.get((src, gun), 0.0) + desi
        # varis_g = arrival_day(self.data.route_lookup, self.data.gunler, (src, dst), gun, slot, arac_turu)
        # if varis_g and not skip_dst_handling:
        #     self.handling_usage[(dst, varis_g)] = self.handling_usage.get((dst, varis_g), 0.0) + desi

        cikis_zamani = slot_datetime(gun, slot)
        # Yeni (oransal dağılım):
        if not skip_src_handling:
            sure_dk = ellecleme_suresi_dakika(desi, consolidation=False)
            dagilim = _ellecleme_dagilimi(cikis_zamani, desi, sure_dk)
            for gun_str, pay in dagilim.items():
                self.handling_usage[(src, gun_str)] = self.handling_usage.get((src, gun_str), 0.0) + pay

        # Varış elleçlemesi
        seyir_saat = seyir_suresi_saat(self.data.route_lookup, src, dst, arac_turu)
        varis_dt = varis_zamani(cikis_zamani, seyir_saat)
        varis_g = arrival_day(self.data.route_lookup, self.data.gunler, (src, dst), gun, slot, arac_turu)
        if not skip_dst_handling and varis_g is not None:
            sure_dk = ellecleme_suresi_dakika(desi, consolidation=False)
            dagilim = _ellecleme_dagilimi(varis_dt, desi, sure_dk)
            for gun_str, pay in dagilim.items():
                self.handling_usage[(dst, gun_str)] = self.handling_usage.get((dst, gun_str), 0.0) + pay

        real_key = key + (is_kiralik,)
        if not skip_src_handling:
            self.cikis_ellecleme_desi[real_key] = self.cikis_ellecleme_desi.get(real_key, 0.0) + desi
        if not skip_dst_handling:
            self.varis_ellecleme_desi[real_key] = self.varis_ellecleme_desi.get(real_key, 0.0) + desi

        return marjinal_maliyet


# ============================================================================
# Yol (path) seçenekleri ve maliyet/SLA hesabı
# ============================================================================
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


def leg_zaman_cizelgesi(data: ProblemData, legs: list, desi: float, milk_run: bool = False) -> list:
    """Her bacağın GERÇEK (elleçleme dahil) kalkış ve varış anını sırayla döndürür:
    [(kalkis_0, varis_0), (kalkis_1, varis_1), ...]. Rapor (alns_optimize.py) bu
    hesabı kullanır.

    milk_run=True ise: ara duraklarda (indir+yeniden yükle) hiç elleçleme YAPILMAZ -
    aynı fiziksel araç kargoyu üzerinde taşıyarak anında devam eder (zaman = varış anı,
    hiçbir gecikme eklenmez). İlk bacağın kalkışı (gerçek köken yükleme) ve son bacağın
    tamamlanması (gerçek nihai indirme) HER ZAMAN gerçek elleçleme süresi içerir -
    milk_run sadece ARADAKİ durakları etkiler."""
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
            zaman = varis if milk_run else ellecleme_tamamlanma_zamani(varis, desi, consolidation=True)
        else:
            zaman = varis

    return cizelge


def _bucket_aware_sla_cost(data, legs, desi, demand_gun, demand_slot, demand_hat,
                            spot_desi_map, kiralik_desi_map, milk_run=False) -> float:
    """SLA cezasini, her bacagin desi parametresi olarak KENDI (kucuk olabilen)
    payi yerine o bacaktaki GUNCEL yukten turetir - araç kapasitesiyle (kap)
    sinirli, cunku bucket birden fazla araca bolunebiliyor ve her arac sadece
    KENDI yukunu elleçliyor (bkz. _fresh_sla_cost ve try_insert_path/
    evaluate_path/cpsat_hat_repair'daki kullanim yerleri)."""
    zaman = None
    bucket_desi = desi
    for i, leg in enumerate(legs):
        key = (leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu)
        mevcut = kiralik_desi_map.get(key, 0.0) if leg.is_kiralik else spot_desi_map.get(key, 0.0)
        kap = data.arac_parametreleri[leg.arac_turu]["kapasite_desi"]
        bucket_desi = max(desi, min(mevcut, kap))
        slot_zamani = slot_datetime(leg.gun, leg.slot)

        if i == 0:
            kalkis = ellecleme_tamamlanma_zamani(slot_zamani, bucket_desi, consolidation=False)
        else:
            kalkis = max(zaman, slot_zamani)

        seyir = data.route_lookup[(leg.src, leg.dst)][leg.arac_turu]
        varis = varis_zamani(kalkis, seyir)

        if i < len(legs) - 1:
            zaman = varis if milk_run else ellecleme_tamamlanma_zamani(varis, bucket_desi, consolidation=True)
        else:
            zaman = varis

    tamamlanma = ellecleme_tamamlanma_zamani(zaman, bucket_desi, consolidation=False)
    talep_tamamlanma = slot_datetime(demand_gun, demand_slot)
    hedef_gun = data.route_lookup[demand_hat]["target_delivery_days"]
    deadline = sla_deadline(talep_tamamlanma, hedef_gun)
    gecikme = gecikme_saat(tamamlanma, deadline)
    return sla_cezasi_tl(desi, gecikme)


def _fresh_sla_cost(state: "State", a: Assignment) -> float:
    return _bucket_aware_sla_cost(
        state.data, a.legs, a.desi, a.demand_gun, a.demand_slot, a.demand_hat,
        state.leg_spot_desi, state.leg_kiralik_desi, milk_run=a.milk_run,
    )


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
        if best is None:
            # ONEMLI DUZELTME: _rank_spot_types_by_cost SADECE 'desi'ye (bu
            # parcaya) bakiyor, bu bacakta HALIHAZIRDA acik/yari-dolu bir arac
            # olup olmadigini HIC gormuyor (data alıyor, state degil) - bu
            # yuzden bacakta %25 dolu bir Kamyon dururken bile sistem, bu
            # kucuk parca icin sifirdan bir Kamyonet aciyordu (dusuk dolulugun
            # asil nedeni buydu, bkz. sohbet gecmisi). Duzeltme: her aday arac
            # turu icin GERCEK MARJINAL maliyeti (_commit_leg'deki AYNI formul)
            # hesaplayip, mevcut yuke eklemenin birim-desi maliyetine gore
            # sirala - boylece zaten acik olan bir aracin bos kapasitesine
            # eklemek (marjinal maliyet ~0'a yakin), yeni bir arac acmaktan
            # dogal olarak daha ucuz cikip tercih edilir.
            adaylar = []
            for arac_turu in data.arac_turleri:
                miktar = state.max_addable_on_leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, False)
                if miktar <= 0:
                    continue
                p = data.arac_parametreleri[arac_turu]
                kap = p["kapasite_desi"]
                mevcut = state.leg_spot_desi.get((leg_src, leg_dst, leg_gun, leg_slot, arac_turu), 0.0)
                onerilen = min(desi, miktar)

                eski_adet = spot_vehicle_count(mevcut, kap, MAX_SPOT)
                yeni_adet = spot_vehicle_count(mevcut + onerilen, kap, MAX_SPOT)
                birim = vehicle_leg_cost(data.route_lookup, (leg_src, leg_dst), arac_turu, p["spot_hourly"], p["spot_km"])
                eski_maliyet = eski_adet * birim + ellecleme_maliyet_hesapla(mevcut, p["spot_hourly"])
                yeni_maliyet = yeni_adet * birim + ellecleme_maliyet_hesapla(mevcut + onerilen, p["spot_hourly"])
                marjinal_maliyet = yeni_maliyet - eski_maliyet

                adaylar.append((marjinal_maliyet / onerilen, onerilen, arac_turu))

            if adaylar:
                adaylar.sort(key=lambda x: x[0])
                _, onerilen, arac_turu = adaylar[0]
                best = (onerilen, arac_turu, False)
        if best is None:
            return None
        leg_plans.append((leg_src, leg_dst, leg_gun, leg_slot, *best))

    tasinabilir = min(desi, min(p[4] for p in leg_plans))
    if tasinabilir <= 0:
        return None

    # YENİ EKLENECEK GÜVENLİK BLOĞU: Aktarma (Relay) merkezlerindeki çift sayımı engelle
    path_handling = {}
    for (leg_src, leg_dst, leg_gun, leg_slot, _miktar, arac_turu, is_kiralik) in leg_plans:
        # Bu rotanın hangi TM'de, hangi gün ne kadar çarpanla kapasite tükettiğini bul
        path_handling[(leg_src, leg_gun)] = path_handling.get((leg_src, leg_gun), 0.0) + 1.0
        varis_g = arrival_day(data.route_lookup, data.gunler, (leg_src, leg_dst), leg_gun, leg_slot, arac_turu)
        if varis_g:
             path_handling[(leg_dst, varis_g)] = path_handling.get((leg_dst, varis_g), 0.0) + 1.0
             
    # Eğer bir TM aynı gün hem varış hem çıkış alıyorsa (multiplier = 2.0 olur), taşınabilir miktarı tıraşla
    for (tm, g), multiplier in path_handling.items():
        avail = state.handling_available(tm, g)
        if tasinabilir * multiplier > avail:
            tasinabilir = avail / multiplier
        if tasinabilir <= 1e-6:
            return None # Daha fazla bakmaya gerek yok, bu yol zaten kapasiteyi aşıyor!

    legs = []
    vehicle_cost = 0.0
    for (leg_src, leg_dst, leg_gun, leg_slot, _miktar, arac_turu, is_kiralik) in leg_plans:
        vehicle_cost += state._commit_leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, tasinabilir, is_kiralik)
        legs.append(Leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, is_kiralik))

    sla_cost = _bucket_aware_sla_cost(
        data, legs, tasinabilir, demand_gun, demand_slot, (src, dst),
        state.leg_spot_desi, state.leg_kiralik_desi,
    )

    assignment = Assignment(
        demand_hat=hat, demand_gun=demand_gun, demand_slot=demand_slot, desi=tasinabilir,
        legs=tuple(legs), sla_cost=sla_cost, vehicle_cost=vehicle_cost, talep_id=talep_id,
    )
    state.assignments.append(assignment)
    return assignment


def insertion_options(data: ProblemData, hat: tuple, gun: str, slot: str):
    """Aynı slotta denenecek yol seçeneklerini üretir: direkt + tüm 1/2-aktarmalı
    adaylar. Her öğe (path, milk_run) çiftidir. Çok bacaklı (path boş olmayan) her
    aday için HEM mevcut konsolidasyon (milk_run=False: ara durakta indir+yeniden
    yükle) HEM uğrama (milk_run=True: aynı araç, ara durakta elleçleme yok) varyantı
    üretilir - arama motoru ikisini de deneyip hangisi daha ucuzsa onu seçer (bkz.
    evaluate_path). Sıra önemli değil — `_insert_chunk` bunların HEPSİNİ deneyip
    maliyete göre en ucuzunu seçiyor (bkz. o fonksiyonun docstring'i)."""
    yield ((), False)  # direkt
    for path in data.relay_candidates.get(hat, []):  # (r,) ya da (r1, r2) tuple'lari
        yield (path, False)
        if MILK_RUN_ENABLED:
            yield (path, True)


def _kiralik_bekleme_secenegi(state, hat, aktif_gun, aktif_slot, gun, slot, kalan, talep_id):
    """aktif_slot kiralık kalkış slotu (KIRALIK_DISPATCH_SLOT) DEĞİLSE, bir
    sonraki kiralık kalkışını (ilk sonraki günün ilk slotu) bulup, kargoyu
    ORAYA ERTELEYİP direkt kiralıkla göndermenin gerçek marjinal maliyetini
    (elleçleme + ERTELEMENİN SLA cezası dahil) döndürür.

    Sözleşmeli kiralık aracın seyir maliyeti zaten baştan (kullanılsa da
    kullanılmasa da) ödendiği için (bkz. State._kiralik_bos_seyir_maliyeti),
    bu kapasiteye erişebilen her kargo neredeyse her zaman spot'tan ucuzdur -
    ama _insert_chunk kargoyu KENDİ oluşum slotunda HEMEN (genelde bol spot
    kapasitesiyle) yerleştirdiği için, talep kiralığın SADECE kalktığı slotta
    OLUŞMADIĞI sürece bu ucuz kapasiteye hiç sıra gelmiyordu (bkz. sohbet
    geçmişi: İstanbul-Eskişehir/Yalova hatlarında talebin çoğu 17:00'de
    oluşuyor, kiralık ise sadece 09:00'da kalkıyor - 2 sözleşmeli Tır'dan biri
    hep boş kalıyordu). Bu fonksiyon o fırsatı AYRI bir aday olarak
    değerlendirmeyi mümkün kılar - çağıran taraf (_insert_chunk) bunun birim
    maliyetini normal (anlık) en iyi seçenekle karşılaştırıp gerçekten
    ucuzsa kullanır, değilse yok sayar.

    Sadece DİREKT yol (path=()) deneniyor - uğrama/aktarmalı adaylar bu
    erteleme mantığına dahil değil (kapsam: en yaygın/basit kiralık kullanım
    şekli). None döner: uygun bir sonraki kiralık slotu yoksa ya da o slotta
    (kiralık dahil) hiç kapasite yoksa."""
    if aktif_slot == KIRALIK_DISPATCH_SLOT:
        return None
    idx = state.data.zaman_sirali.index((aktif_gun, aktif_slot))
    kiralik_zamani = None
    for aday_gun, aday_slot in state.data.zaman_sirali[idx + 1:]:
        if aday_slot == KIRALIK_DISPATCH_SLOT:
            kiralik_zamani = (aday_gun, aday_slot)
            break
    if kiralik_zamani is None:
        return None
    kiralik_gun, kiralik_slot = kiralik_zamani
    eval_sonuc = evaluate_path(
        state, hat, kiralik_gun, kiralik_slot, kalan, (), talep_id,
        demand_gun=gun, demand_slot=slot, milk_run=False,
    )
    if eval_sonuc is None or eval_sonuc['desi'] <= 1e-9:
        return None
    return eval_sonuc


def _insert_chunk(state, hat, gun, slot, desi, rng, talep_id):
    kalan = desi
    idx = state.data.zaman_sirali.index((gun, slot))

    for aktif_gun, aktif_slot in state.data.zaman_sirali[idx:]:
        if kalan <= 1e-6:
            break

        secenekler = list(insertion_options(state.data, hat, aktif_gun, aktif_slot))
        # Aynı slot içinde birden fazla deneme yap
        while kalan > 1e-6:
            en_iyi_secenek = None
            en_iyi_desi = 0
            en_iyi_birim_maliyet = float('inf')
            en_iyi_eval = None

            for (secenek, milk_run) in secenekler:
                is_deferred = (aktif_gun != gun or aktif_slot != slot)
                eval_sonuc = evaluate_path(
                    state, hat, aktif_gun, aktif_slot, kalan, secenek, talep_id,
                    demand_gun = gun if is_deferred else None,
                    demand_slot = slot if is_deferred else None,
                    milk_run = milk_run,
                )
                if eval_sonuc is None or eval_sonuc['desi'] <= 1e-9:
                    continue
                birim_maliyet = eval_sonuc['maliyet'] / eval_sonuc['desi']
                # önce en çok desi, eşitse en ucuz
                if eval_sonuc['desi'] > en_iyi_desi or \
                   (abs(eval_sonuc['desi'] - en_iyi_desi) < 1e-9 and birim_maliyet < en_iyi_birim_maliyet):
                    en_iyi_desi = eval_sonuc['desi']
                    en_iyi_birim_maliyet = birim_maliyet
                    en_iyi_secenek = secenek
                    en_iyi_eval = eval_sonuc

            # YENİ: bu an kiralığın kalkmadığı bir slotsa, ileride (bir sonraki
            # kiralık kalkışında) DİREKT kiralıkla göndermenin normal en iyi
            # seçenekten ucuza mı geldiğine bak - öyleyse (kiralığın izin
            # verdiği kadarını) ONU kullan, kalanı normal akışa bırak.
            kiralik_eval = _kiralik_bekleme_secenegi(state, hat, aktif_gun, aktif_slot, gun, slot, kalan, talep_id)
            if kiralik_eval is not None:
                kiralik_birim = kiralik_eval['maliyet'] / kiralik_eval['desi']
                if en_iyi_secenek is None or kiralik_birim < en_iyi_birim_maliyet:
                    sla_cost = kiralik_eval.get("sla_cost")
                    commit_path(state, hat, kiralik_eval, talep_id,
                                demand_gun=gun, demand_slot=slot, sla_cost=sla_cost)
                    kalan -= kiralik_eval['desi']
                    continue

            if en_iyi_secenek is None:
                break

            sla_cost = en_iyi_eval.get("sla_cost")
            # Kazananı commit et
            commit_path(state, hat, en_iyi_eval, talep_id,
                        demand_gun=gun if (aktif_gun != gun or aktif_slot != slot) else gun,
                        demand_slot=slot if (aktif_gun != gun or aktif_slot != slot) else slot, sla_cost=sla_cost)
            kalan -= en_iyi_desi

    return kalan


def evaluate_path(state, hat, gun, slot, desi, path, talep_id="",
                  demand_gun=None, demand_slot=None, milk_run=False):
    """Verili bir yol (path) için state'i değiştirmeden (dry‑run) kapasite kontrolü
    ve marjinal maliyet hesabı yapar.

    Bu fonksiyon, state üzerinde HİÇBİR değişiklik yapmaz; yalnızca mevcut kapasite
    kullanımını (leg_spot_desi, leg_kiralik_desi, handling_usage, tir_usage)
    gölge sözlüklere kopyalayarak çalışır. Bu sayede birden çok aday yol hızlıca
    karşılaştırılabilir.

    Parametreler
    ----------
    state : State
        Anlık çözüm durumu (salt okunur).
    hat : tuple
        (kaynak_TM, hedef_TM)
    gun, slot : str
        FİİLİ kalkışın yapılacağı gün ve slot.
    desi : float
        Yerleştirilmek istenen maksimum desi.
    path : tuple
        Boş tuple = direkt, (r,) = 1‑aktarma, (r1, r2) = 2‑aktarma.
    talep_id : str
        Talebin benzersiz kimliği (raporlama için).
    demand_gun, demand_slot : str or None
        Talebin ORİJİNAL oluşum zamanı (SLA deadline'ı bunun üzerinden hesaplanır).
        None verilirse gun/slot ile aynı kabul edilir.
    milk_run : bool
        True ise (sadece path boş değilse anlamlı): TÜM bacaklar AYNI (tek) spot
        araç türüyle, kiralık olmadan gidilir - ara duraklarda araç hiç indirilmez/
        yeniden yüklenmez (kural: uğrama sadece spot araçlarla, aynı araç devam eder).
        False ise (varsayılan) mevcut konsolidasyon davranışı: her bacak için
        bağımsız araç türü seçilir, ara duraklarda tam indir+yeniden yükle olur.

    Dönüş
    -------
    dict veya None
        Başarılıysa:
        {
            'desi': float,          # bu yolda taşınabilecek gerçek desi miktarı
            'maliyet': float,       # marjinal araç maliyeti + SLA cezası
            'leg_plans': list,      # her bacak için (src,dst,gun,slot,miktar,
                                    #   arac_turu,is_kiralik) kaydı – commit_path
                                    #   için gerekli
            'legs': list[Leg],      # oluşturulan Leg nesneleri
            'sla_cost': float,      # SLA cezası
            'vehicle_cost': float,  # marjinal araç maliyeti
            'milk_run': bool,       # commit_path'in dokunus/skip mantığını
                                    #   tekrar üretebilmesi için
        }
        Hiçbir bacakta kapasite kalmamışsa veya rota geçersizse None döner.
    """
    demand_gun = gun if demand_gun is None else demand_gun
    demand_slot = slot if demand_slot is None else demand_slot

    if milk_run and not path:
        return None  # ugrama sadece coklu bacakli (aktarmali) yollarda anlamli

    src, dst = hat
    data = state.data

    # 1. Durakları ve bacak kalkış zamanlarını hesapla (try_insert_path'teki gibi)
    stops = [src, *path, dst]
    leg_departures = [(gun, slot)]
    for i in range(len(stops) - 1):
        if i == len(stops) - 2:
            break
        leg_src, leg_dst = stops[i], stops[i + 1]
        entry = data.route_lookup.get((leg_src, leg_dst))
        if entry is None:
            return None
        est_arac_turu = _rank_spot_types_by_cost(data, (leg_src, leg_dst), desi)[0]
        cur_gun, cur_slot = leg_departures[-1]
        sonraki = next_dispatch_slot(data.gunler, cur_gun, cur_slot, entry[est_arac_turu])
        if sonraki is None:
            return None
        leg_departures.append(sonraki)

    leg_pairs = [
        (stops[i], stops[i + 1], leg_departures[i][0], leg_departures[i][1])
        for i in range(len(stops) - 1)
    ]
    n_legs = len(leg_pairs)
    # Ugrama (milk_run) icin her bacagin POZISYONUNA gore hangi ucunun (cikis/varis)
    # gercek bir elleçleme oldugunu (skip=False) ya da ayni aracin devam ettigi bir
    # ara durak oldugunu (skip=True, hic elleçleme yok) belirler. Konsolidasyonda
    # (milk_run=False) ikisi de hep False - _commit_leg'deki mantikla BIREBIR ayni
    # (bkz. o fonksiyonun docstring'i).
    skip_src_per_leg = [milk_run and i > 0 for i in range(n_legs)]
    skip_dst_per_leg = [milk_run and i < n_legs - 1 for i in range(n_legs)]

    # 2. Geçici kapasite takip dict'leri oluştur (state'i değiştirme)
    temp_leg_spot = dict(state.leg_spot_desi)
    temp_leg_kiralik = dict(state.leg_kiralik_desi)
    temp_leg_ellecleme = dict(state.leg_ellecleme_desi)
    temp_handling = dict(state.handling_usage)
    # temp_tir = dict(state.tir_usage) if data.tir_arac_turu else {}
    temp_tir_in = dict(state.tir_usage_in) if data.tir_arac_turu else {}
    temp_tir_out = dict(state.tir_usage_out) if data.tir_arac_turu else {}

    # Yardımcı fonksiyonlar (geçici dict'leri kullanır)
    def temp_spot_capacity_left(leg_key, arac_turu):
        kap = data.arac_parametreleri[arac_turu]["kapasite_desi"]
        mevcut = temp_leg_spot.get(leg_key, 0.0)
        return max(0.0, MAX_SPOT * kap - mevcut)

    def temp_kiralik_available(hat, gun, arac_turu):
        stok = data.kiralik_stok_gunluk.get((hat, arac_turu), 0)
        if stok <= 0:
            return 0.0
        kap = data.arac_parametreleri[arac_turu]["kapasite_desi"]
        used = temp_leg_kiralik.get((hat[0], hat[1], gun, KIRALIK_DISPATCH_SLOT, arac_turu), 0.0)
        return max(0.0, stok * kap - used)

    def temp_handling_available(tm, gun):
        cap = data.handling_capacity.get(tm)
        if cap is None:
            return float("inf")
        return max(0.0, cap - temp_handling.get((tm, gun), 0.0))

    def temp_tir_available(tm, gun):
        if data.tir_arac_turu is None:
            return float("inf")
        cap = data.tir_capacity.get(tm)
        if cap is None:
            return float("inf")
        fixed = data.fixed_kiralik_tir_usage.get((tm, gun), 0)
        
        spot_in = temp_tir_in.get((tm, gun), 0)
        spot_out = temp_tir_out.get((tm, gun), 0)
        return max(0.0, cap - fixed - (spot_in + spot_out))

    def temp_max_addable_on_leg(src, dst, gun, slot, arac_turu, is_kiralik,skip_src_tir = False):
        if is_kiralik:
            limit = temp_kiralik_available((src, dst), gun, arac_turu)
        else:
            limit = temp_spot_capacity_left((src, dst, gun, slot, arac_turu), arac_turu)
        limit = min(limit, temp_handling_available(src, gun))
        varis_g = arrival_day(data.route_lookup, data.gunler, (src, dst), gun, slot, arac_turu)
        if varis_g is None:
            return 0.0
        limit = min(limit, temp_handling_available(dst, varis_g))
        if arac_turu == data.tir_arac_turu and not is_kiralik:
            kap = data.arac_parametreleri[arac_turu]["kapasite_desi"]
            mevcut_desi = temp_leg_spot.get((src, dst, gun, slot, arac_turu), 0.0)
            mevcut_adet = spot_vehicle_count(mevcut_desi, kap, MAX_SPOT)
            
            # YENİ: skip_src_tir mantığı tamamen temizlendi, doğrudan çağrılıyor
            departure_room = temp_tir_available(src, gun) 
            arrival_room = temp_tir_available(dst, varis_g)
            adet_limiti = min(departure_room, arrival_room)
            max_adet = mevcut_adet + adet_limiti
            limit = min(limit, max(0.0, max_adet * kap - mevcut_desi))
        return max(0.0, limit)

    # 3. Bacak planlarını oluştur (tıpkı try_insert_path'teki gibi)
    leg_plans = []
    if milk_run:
        # Ugrama: TUM bacaklar AYNI (tek) spot araç türüyle gidilir (kiralık asla -
        # bkz. kural: "kiralık araçlarla uğrama yapılamaz"). Önce her aday araç
        # türü için TÜM bacaklarda pozitif kapasite olup olmadığını kontrol et,
        # sonra gerçek (dokunuş-farkındalıklı) marjinal maliyete göre en ucuzunu seç.
        # YON-TUTARLILIGI KONTROLU: bu arac_turu ile HUB'daki her ara durak (junction),
        # state.milk_fwd/milk_bwd'de HALIHAZIRDA BASKA bir bacakla kilitli olmamali -
        # aksi halde ayni HUB bacagi, biri bu yone biri baska bir yone giden iki farkli
        # "elleclenmeden devam eden" talep tarafindan paylasilmis olur ki bu fiziksel
        # olarak imkansizdir (bkz. State.milk_fwd docstring'i).
        def _milk_junction_ok(arac_turu):
            for j in range(len(leg_pairs) - 1):
                j_src, j_dst, j_gun, j_slot = leg_pairs[j]
                k_src, k_dst, k_gun, k_slot = leg_pairs[j + 1]
                bucket_j = (j_src, j_dst, j_gun, j_slot, arac_turu)
                bucket_k = (k_src, k_dst, k_gun, k_slot, arac_turu)
                if state.milk_fwd.get(bucket_j, bucket_k) != bucket_k:
                    return False
                if state.milk_bwd.get(bucket_k, bucket_j) != bucket_j:
                    return False
            return True

        ortak_adaylar = []
        for arac_turu in data.arac_turleri:
            
            p = data.arac_parametreleri[arac_turu]
            miktarlar = []
            uygun = True
            for idx, (leg_src, leg_dst, leg_gun, leg_slot) in enumerate(leg_pairs):
                # skip_src_tir = i > 0
                miktar = temp_max_addable_on_leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, False)
                if miktar <= 0:
                    uygun = False
                    break
                miktarlar.append(miktar)
            if not uygun:
                continue
            if not _milk_junction_ok(arac_turu):
                continue
            onerilen = min(desi, min(miktarlar))
            if onerilen <= 0:
                continue

            toplam_marjinal = 0.0
            for i, (leg_src, leg_dst, leg_gun, leg_slot) in enumerate(leg_pairs):
                dokunus = (0 if skip_src_per_leg[i] else 1) + (0 if skip_dst_per_leg[i] else 1)
                key = (leg_src, leg_dst, leg_gun, leg_slot, arac_turu)
                kap = p["kapasite_desi"]
                mevcut = temp_leg_spot.get(key, 0.0)
                eski_adet = spot_vehicle_count(mevcut, kap, MAX_SPOT)
                yeni_adet = spot_vehicle_count(mevcut + onerilen, kap, MAX_SPOT)
                birim = vehicle_leg_cost(data.route_lookup, (leg_src, leg_dst), arac_turu, p["spot_hourly"], p["spot_km"])
                eski_ellec_desi = temp_leg_ellecleme.get(key + (dokunus,), 0.0)
                yeni_ellec_desi = eski_ellec_desi + onerilen
                eski_maliyet = eski_adet * birim + ellecleme_maliyet_hesapla(eski_ellec_desi, p["spot_hourly"], dokunus)
                yeni_maliyet = yeni_adet * birim + ellecleme_maliyet_hesapla(yeni_ellec_desi, p["spot_hourly"], dokunus)
                toplam_marjinal += (yeni_maliyet - eski_maliyet)

            ortak_adaylar.append((toplam_marjinal / onerilen, onerilen, arac_turu))

        if not ortak_adaylar:
            return None
        ortak_adaylar.sort(key=lambda x: x[0])
        _, onerilen, secilen_arac_turu = ortak_adaylar[0]
        leg_plans = [
            (leg_src, leg_dst, leg_gun, leg_slot, onerilen, secilen_arac_turu, False)
            for (leg_src, leg_dst, leg_gun, leg_slot) in leg_pairs
        ]
    else:
        for idx,(leg_src, leg_dst, leg_gun, leg_slot) in enumerate(leg_pairs):
            best = None
            if leg_slot == KIRALIK_DISPATCH_SLOT:
                for arac_turu in data.arac_turleri:
                    miktar = temp_max_addable_on_leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, True)
                    if miktar > 0:
                        best = (miktar, arac_turu, True)
                        break
            if best is None:
                adaylar = []
                for arac_turu in data.arac_turleri:
                    miktar = temp_max_addable_on_leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, False)
                    if miktar <= 0:
                        continue
                    # Marinal maliyet hesapla (mevcut desiye göre)
                    p = data.arac_parametreleri[arac_turu]
                    kap = p["kapasite_desi"]
                    mevcut = temp_leg_spot.get((leg_src, leg_dst, leg_gun, leg_slot, arac_turu), 0.0)
                    onerilen = min(desi, miktar)
                    eski_adet = spot_vehicle_count(mevcut, kap, MAX_SPOT)
                    yeni_adet = spot_vehicle_count(mevcut + onerilen, kap, MAX_SPOT)
                    birim = vehicle_leg_cost(data.route_lookup, (leg_src, leg_dst), arac_turu, p["spot_hourly"], p["spot_km"])
                    eski_maliyet = eski_adet * birim + ellecleme_maliyet_hesapla(mevcut, p["spot_hourly"])
                    yeni_maliyet = yeni_adet * birim + ellecleme_maliyet_hesapla(mevcut + onerilen, p["spot_hourly"])
                    marjinal_maliyet = yeni_maliyet - eski_maliyet
                    adaylar.append((marjinal_maliyet / onerilen, onerilen, arac_turu))
                if adaylar:
                    adaylar.sort(key=lambda x: x[0])
                    _, onerilen, arac_turu = adaylar[0]
                    best = (onerilen, arac_turu, False)
            if best is None:
                return None
            leg_plans.append((leg_src, leg_dst, leg_gun, leg_slot, *best))

    # 4. Taşınabilir desi miktarını bul (path_handling kontrolü dahil)
    tasinabilir = min(desi, min(p[4] for p in leg_plans))
    if tasinabilir <= 0:
        return None

    # path_handling kontrolü (orijinaldeki gibi, ama uğrama'da atlanan uçlar 0 katkı yapar)
    path_handling = {}
    for i, (leg_src, leg_dst, leg_gun, leg_slot, _miktar, arac_turu, is_kiralik) in enumerate(leg_plans):
        if not skip_src_per_leg[i]:
            path_handling[(leg_src, leg_gun)] = path_handling.get((leg_src, leg_gun), 0.0) + 1.0
        varis_g = arrival_day(data.route_lookup, data.gunler, (leg_src, leg_dst), leg_gun, leg_slot, arac_turu)
        if varis_g and not skip_dst_per_leg[i]:
            path_handling[(leg_dst, varis_g)] = path_handling.get((leg_dst, varis_g), 0.0) + 1.0

    for (tm, g), multiplier in path_handling.items():
        avail = temp_handling_available(tm, g)
        if tasinabilir * multiplier > avail:
            tasinabilir = avail / multiplier
        if tasinabilir <= 1e-6:
            return None

    # 5. Geçici olarak bacak desilerini güncelle ve maliyet hesapla
    toplam_vehicle_cost = 0.0
    legs = []
    for i, (leg_src, leg_dst, leg_gun, leg_slot, _miktar, arac_turu, is_kiralik) in enumerate(leg_plans):
        skip_src = skip_src_per_leg[i]
        skip_dst = skip_dst_per_leg[i]
        dokunus = (0 if skip_src else 1) + (0 if skip_dst else 1)
        # Geçici dict'leri güncelle (böylece sonraki bacaklar etkilenir)
        key = (leg_src, leg_dst, leg_gun, leg_slot, arac_turu)
        ellec_key = key + (dokunus,)
        p = data.arac_parametreleri[arac_turu]
        if is_kiralik:
            eski = temp_leg_kiralik.get(key, 0.0)
            yeni = eski + tasinabilir
            temp_leg_kiralik[key] = yeni
            eski_ellec_desi = temp_leg_ellecleme.get(ellec_key, 0.0)
            yeni_ellec_desi = eski_ellec_desi + tasinabilir
            temp_leg_ellecleme[ellec_key] = yeni_ellec_desi
            eski_ellec = ellecleme_maliyet_hesapla(eski_ellec_desi, p["rental_hourly"], dokunus)
            yeni_ellec = ellecleme_maliyet_hesapla(yeni_ellec_desi, p["rental_hourly"], dokunus)
            toplam_vehicle_cost += (yeni_ellec - eski_ellec)
        else:
            eski = temp_leg_spot.get(key, 0.0)
            yeni = eski + tasinabilir
            kap = p["kapasite_desi"]
            eski_adet = spot_vehicle_count(eski, kap, MAX_SPOT)
            yeni_adet = spot_vehicle_count(yeni, kap, MAX_SPOT)
            temp_leg_spot[key] = yeni
            birim = vehicle_leg_cost(data.route_lookup, (leg_src, leg_dst), arac_turu, p["spot_hourly"], p["spot_km"])
            eski_ellec_desi = temp_leg_ellecleme.get(ellec_key, 0.0)
            yeni_ellec_desi = eski_ellec_desi + tasinabilir
            temp_leg_ellecleme[ellec_key] = yeni_ellec_desi
            eski_maliyet = eski_adet * birim + ellecleme_maliyet_hesapla(eski_ellec_desi, p["spot_hourly"], dokunus)
            yeni_maliyet = yeni_adet * birim + ellecleme_maliyet_hesapla(yeni_ellec_desi, p["spot_hourly"], dokunus)
            toplam_vehicle_cost += (yeni_maliyet - eski_maliyet)
            if arac_turu == data.tir_arac_turu:
                delta_adet = yeni_adet - eski_adet
                # _commit_leg'deki skip_src mantığinin ayni: milk_run devaminda
                # bu bacagin cikisi ayrica sayilmaz (bkz. KISITLAR.md madde 22).
                if not skip_src:
                    temp_tir_out[(leg_src, leg_gun)] = temp_tir_out.get((leg_src, leg_gun), 0) + delta_adet
                varis_g = arrival_day(data.route_lookup, data.gunler, (leg_src, leg_dst), leg_gun, leg_slot, arac_turu)
                if varis_g:
                    temp_tir_in[(leg_dst, varis_g)] = temp_tir_in.get((leg_dst, varis_g), 0) + delta_adet
        # if not skip_src:
        #     temp_handling[(leg_src, leg_gun)] = temp_handling.get((leg_src, leg_gun), 0.0) + tasinabilir
        # varis_g = arrival_day(data.route_lookup, data.gunler, (leg_src, leg_dst), leg_gun, leg_slot, arac_turu)
        # if varis_g and not skip_dst:
        #     temp_handling[(leg_dst, varis_g)] = temp_handling.get((leg_dst, varis_g), 0.0) + tasinabilir
        # Çıkış elleçlemesi
        cikis_zamani = slot_datetime(leg_gun, leg_slot)
        if not skip_src:
            sure_dk = ellecleme_suresi_dakika(tasinabilir, consolidation=False)
            dagilim = _ellecleme_dagilimi(cikis_zamani, tasinabilir, sure_dk)
            for gun_str, pay in dagilim.items():
                temp_handling[(leg_src, gun_str)] = temp_handling.get((leg_src, gun_str), 0.0) + pay

        # Varış elleçlemesi
        seyir_saat = seyir_suresi_saat(data.route_lookup, leg_src, leg_dst, arac_turu)
        varis_dt = varis_zamani(cikis_zamani, seyir_saat)
        varis_g = arrival_day(data.route_lookup, data.gunler, (leg_src, leg_dst), leg_gun, leg_slot, arac_turu)
        if varis_g and not skip_dst:
            sure_dk = ellecleme_suresi_dakika(tasinabilir, consolidation=False)
            dagilim = _ellecleme_dagilimi(varis_dt, tasinabilir, sure_dk)
            for gun_str, pay in dagilim.items():
                temp_handling[(leg_dst, gun_str)] = temp_handling.get((leg_dst, gun_str), 0.0) + pay

        legs.append(Leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, is_kiralik))

    # 6. SLA maliyeti hesapla
    sla_cost = _bucket_aware_sla_cost(
        data, legs, tasinabilir, demand_gun, demand_slot, (src, dst),
        temp_leg_spot, temp_leg_kiralik, milk_run=milk_run,
    )

    return {
        'desi': tasinabilir,
        'maliyet': toplam_vehicle_cost + sla_cost,
        'leg_plans': leg_plans,  # bacak planlarını da döndür ki commit_path kullanabilsin
        'legs': legs,
        'sla_cost': sla_cost,
        'vehicle_cost': toplam_vehicle_cost,
        'milk_run': milk_run,  # commit_path'in ayni skip/dokunus mantigini tekrar uretmesi icin
    }

def commit_path(state, hat, eval_result, talep_id, demand_gun, demand_slot, sla_cost):
    """evaluate_path tarafından seçilen yolun gerçek state üzerinde kalıcı olarak
    uygulanmasını sağlar.

    evaluate_path'in döndürdüğü leg_plans kullanılarak state._commit_leg() çağrılır,
    böylece state'in kapasite takipçileri (leg_spot_desi, handling_usage, _arac_maliyeti_toplam
    vb.) doğru şekilde güncellenir. Daha sonra ilgili Assignment nesnesi oluşturulup
    state.assignments listesine eklenir.

    Parametreler
    ----------
    state : State
        Üzerinde değişiklik yapılacak çözüm durumu.
    hat : tuple
        (kaynak_TM, hedef_TM)
    eval_result : dict
        evaluate_path'ten dönen başarılı sonuç sözlüğü (yukarıdaki formatta).
    talep_id : str
        Talebin kimliği.
    demand_gun, demand_slot : str
        Talebin orijinal oluşum zamanı (SLA için).

    Dönüş
    -------
    Assignment
        Oluşturulan ve state'e eklenen Assignment nesnesi.
    """
    data = state.data
    src, dst = hat
    tasinabilir = eval_result['desi']
    leg_plans = eval_result['leg_plans']
    milk_run = eval_result.get('milk_run', False)
    n_legs = len(leg_plans)
    legs = []
    vehicle_cost = 0.0
    for i, (leg_src, leg_dst, leg_gun, leg_slot, _miktar, arac_turu, is_kiralik) in enumerate(leg_plans):
        # evaluate_path'teki skip_src_per_leg/skip_dst_per_leg ile BIREBIR ayni pozisyon
        # mantigi (bkz. o fonksiyonun ve _commit_leg'in docstring'i).
        skip_src_handling = milk_run and i > 0
        skip_dst_handling = milk_run and i < n_legs - 1
        # skip_src_tir = milk_run and i > 0
        cost = state._commit_leg(
            leg_src, leg_dst, leg_gun, leg_slot, arac_turu, tasinabilir, is_kiralik,
            skip_src_handling=skip_src_handling, skip_dst_handling=skip_dst_handling
        )
        vehicle_cost += cost
        legs.append(Leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, is_kiralik))

    if milk_run:
        # HUB junction'larini yon-tutarliligi kilidine kaydet (bkz. State.milk_fwd
        # docstring'i ve evaluate_path._milk_junction_ok) - _remove_assignment bu
        # kilidi, assignment kaldirildiginda tam olarak geri alir.
        for j in range(n_legs - 1):
            bucket_j = (legs[j].src, legs[j].dst, legs[j].gun, legs[j].slot, legs[j].arac_turu)
            bucket_k = (legs[j + 1].src, legs[j + 1].dst, legs[j + 1].gun, legs[j + 1].slot, legs[j + 1].arac_turu)
            state.milk_fwd[bucket_j] = bucket_k
            state.milk_bwd[bucket_k] = bucket_j
            jk = (bucket_j, bucket_k)
            state.milk_junction_desi[jk] = state.milk_junction_desi.get(jk, 0.0) + tasinabilir

    assignment = Assignment(
        demand_hat=hat, demand_gun=demand_gun, demand_slot=demand_slot,
        desi=tasinabilir, legs=tuple(legs), sla_cost=sla_cost,
        vehicle_cost=vehicle_cost, talep_id=talep_id, milk_run=milk_run,
    )
    state.assignments.append(assignment)
    return assignment

def dummy_initial_builder(state, rng, **kwargs):
    """
    ALNS döngüsü başlamadan ÖNCE, sistemi sıfır kapasite ihlali ile dolduran 
    özel 'Başlangıç İnşa' operatörüdür.
    
    Konsolidasyonu umursamaz. Eğer bir kargo o gün/slota sığmıyorsa, force_insert 
    yapmak yerine kargoyu zaman çizelgesinde (timeline) ileriye doğru kaydırarak 
    yasal boşluk arar.
    """
    state = state.copy()
    unassigned_items = list(state.unassigned)
    state.unassigned = []
    
    # 1. Kargoları büyükten küçüğe sırala (Greedy/Açgözlü yerleştirme mantığı)
    # Büyük desileri (Örn: 5000 desi) önce yerleştirmek her zaman daha güvenlidir, 
    # küçük desiler aralara rahatça sızabilir.
    unassigned_items.sort(key=lambda x: x[3], reverse=True)
    
    zamanlar = state.data.zaman_sirali  # Bütün (gun, slot) ikililerinin kronolojik listesi
    
    for item in unassigned_items:
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
    n_legs = len(a.legs)
    for i, leg in enumerate(a.legs):
        # _commit_leg ile AYNI dokunus mantigi (bkz. o fonksiyonun docstring'i) -
        # ugramasiz (a.milk_run=False) atamalarda skip_src/skip_dst hep False,
        # yani dokunus_sayisi hep 2 (mevcut davranis degismez).
        skip_src_handling = a.milk_run and i > 0
        skip_dst_handling = a.milk_run and i < n_legs - 1
        dokunus_sayisi = (0 if skip_src_handling else 1) + (0 if skip_dst_handling else 1)

        key = (leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu)
        ellec_key = key + (dokunus_sayisi,)
        p = state.data.arac_parametreleri[leg.arac_turu]

        eski_ellec_desi = state.leg_ellecleme_desi.get(ellec_key, 0.0)
        yeni_ellec_desi = max(0.0, eski_ellec_desi - a.desi)
        state.leg_ellecleme_desi[ellec_key] = yeni_ellec_desi

        if leg.is_kiralik:
            # DUZELTME: silme sirasinda da (ekleme sirasindaki gibi) GERCEK
            # maliyet farkini _arac_maliyeti_toplam'dan dusuyoruz - aksi halde
            # kalan taleplerin maliyeti BAYATLASIR (bkz. sohbet gecmisi).
            eski_desi = state.leg_kiralik_desi.get(key, 0.0)
            yeni_desi = max(0.0, eski_desi - a.desi)
            state.leg_kiralik_desi[key] = yeni_desi
            old_cost = ellecleme_maliyet_hesapla(eski_ellec_desi, p["rental_hourly"], dokunus_sayisi)
            new_cost = ellecleme_maliyet_hesapla(yeni_ellec_desi, p["rental_hourly"], dokunus_sayisi)
            delta_handling = new_cost - old_cost # Buradaki değer negatif çıkar. çünkü yeni elleçleme maliyetimiz, desi azaldığından dolayı eskisinden az
            state._arac_maliyeti_toplam += delta_handling # Araç maliyeti negatif deger ile toplandığı için azalır.
        else:
            eski = state.leg_spot_desi.get(key, 0.0)
            kap = p["kapasite_desi"]
            eski_adet = spot_vehicle_count(eski, kap, MAX_SPOT)
            yeni = max(0.0, eski - a.desi)
            yeni_adet = spot_vehicle_count(yeni, kap, MAX_SPOT)
            state.leg_spot_desi[key] = yeni

            birim = vehicle_leg_cost(state.data.route_lookup, (leg.src, leg.dst), leg.arac_turu, p["spot_hourly"], p["spot_km"])
            old_cost = eski_adet * birim + ellecleme_maliyet_hesapla(eski_ellec_desi, p["spot_hourly"], dokunus_sayisi)
            new_cost = yeni_adet * birim + ellecleme_maliyet_hesapla(yeni_ellec_desi, p["spot_hourly"], dokunus_sayisi)
            delta_cost = new_cost-old_cost
            state._arac_maliyeti_toplam += delta_cost

            if leg.arac_turu == state.data.tir_arac_turu:
                delta = yeni_adet - eski_adet # Bu negatiftir
                # _commit_leg'deki skip_src_handling atlamasinin simetrik geri alinisi.
                if not skip_src_handling:
                    state.tir_usage_out[(leg.src, leg.gun)] = state.tir_usage_out.get((leg.src, leg.gun), 0) + delta
                varis_g = arrival_day(state.data.route_lookup, state.data.gunler, (leg.src, leg.dst), leg.gun, leg.slot, leg.arac_turu)
                if varis_g:
                    state.tir_usage_in[(leg.dst, varis_g)] = state.tir_usage_in.get((leg.dst, varis_g), 0) + delta
        # if not skip_src_handling:
        #     state.handling_usage[(leg.src, leg.gun)] = state.handling_usage.get((leg.src, leg.gun), 0.0) - a.desi
        # varis_g = arrival_day(state.data.route_lookup, state.data.gunler, (leg.src, leg.dst), leg.gun, leg.slot, leg.arac_turu)
        # if varis_g and not skip_dst_handling:
        #     state.handling_usage[(leg.dst, varis_g)] = state.handling_usage.get((leg.dst, varis_g), 0.0) - a.desi
        # Çıkış elleçlemesi geri alma
        cikis_zamani = slot_datetime(leg.gun, leg.slot)
        if not skip_src_handling:
            sure_dk = ellecleme_suresi_dakika(a.desi, consolidation=False)
            dagilim = _ellecleme_dagilimi(cikis_zamani, a.desi, sure_dk)
            for gun_str, pay in dagilim.items():
                state.handling_usage[(leg.src, gun_str)] = state.handling_usage.get((leg.src, gun_str), 0.0) - pay

        # Varış elleçlemesi geri alma
        seyir_saat = state.data.route_lookup[(leg.src, leg.dst)][leg.arac_turu]
        varis_dt = varis_zamani(cikis_zamani, seyir_saat)
        varis_g = arrival_day(state.data.route_lookup, state.data.gunler, (leg.src, leg.dst), leg.gun, leg.slot, leg.arac_turu)
        if not skip_dst_handling and varis_g is not None:
            sure_dk = ellecleme_suresi_dakika(a.desi, consolidation=False)
            dagilim = _ellecleme_dagilimi(varis_dt, a.desi, sure_dk)
            for gun_str, pay in dagilim.items():
                state.handling_usage[(leg.dst, gun_str)] = state.handling_usage.get((leg.dst, gun_str), 0.0) - pay

        real_key = key + (leg.is_kiralik,)
        if not skip_src_handling:
            state.cikis_ellecleme_desi[real_key] = max(0.0, state.cikis_ellecleme_desi.get(real_key, 0.0) - a.desi)
        if not skip_dst_handling:
            state.varis_ellecleme_desi[real_key] = max(0.0, state.varis_ellecleme_desi.get(real_key, 0.0) - a.desi)

    if a.milk_run:
        # commit_path'te kurulan yon-tutarliligi kilidini (State.milk_fwd/milk_bwd)
        # tam olarak geri al - junction desi'si 0'a inince kilit tamamen acilir ki
        # baska bir talep bu HUB bacagini FARKLI bir yone milk_run ile kullanabilsin.
        for j in range(n_legs - 1):
            bucket_j = (a.legs[j].src, a.legs[j].dst, a.legs[j].gun, a.legs[j].slot, a.legs[j].arac_turu)
            bucket_k = (a.legs[j + 1].src, a.legs[j + 1].dst, a.legs[j + 1].gun, a.legs[j + 1].slot, a.legs[j + 1].arac_turu)
            jk = (bucket_j, bucket_k)
            kalan = state.milk_junction_desi.get(jk, 0.0) - a.desi
            if kalan <= 1e-9:
                state.milk_junction_desi.pop(jk, None)
                if state.milk_fwd.get(bucket_j) == bucket_k:
                    del state.milk_fwd[bucket_j]
                if state.milk_bwd.get(bucket_k) == bucket_j:
                    del state.milk_bwd[bucket_k]
            else:
                state.milk_junction_desi[jk] = kalan
    state.unassigned.append((a.demand_hat, a.demand_gun, a.demand_slot, a.desi, a.talep_id))


# ============================================================================
# Minimum spot doluluk kurali — KESIN (hard, parasal olmayan) uygulama
# ============================================================================
# CP-SAT motorundaki (src/optimization.py) "%10 doluluk" kurali orada SERT bir
# kisit (spot_y * kap <= tasinan_yuk * 10). ALNS parca-parca (chunk-by-chunk)
# insa ettigi icin ayni seyi objective() uzerinden PARASAL bir ceza ile
# yapmaya calismak (once denenen yontem) yanlis cikti: onlarca/yuzlerce
# (hat,gun,slot,tur) grubu ayni anda esigin altinda kalabiliyor, bunlarin
# TOPLAMI objective()'i milyonlarca TL sisiriyordu - bu YARISMANIN gercek
# maliyet semasinda olmayan sahte bir kalem. Onun yerine README'nin zaten
# tarif ettigi is kuralini AYNEN uyguluyoruz: "Doluluk oranini karsilamayan
# yukler o gun tasimaya alinmaz ve bir sonraki gune ertelenir" - yani PARA
# CEZASI degil, KESIN bir ERTELEME. Son (gun,slot) - CP-SAT'taki gibi - muaf.
def dogrula_min_spot_doluluk(state: "State") -> list:
    """Nihai state'te (kademeli/taper edilmis) minimum doluluk esigini ihlal
    eden TUM spot arac gruplarini dondurur - bkz. _min_doluluk_esigi (esik,
    son MIN_SPOT_DOLULUK_TAPER_GUN_SAYISI gun boyunca 0'a kadar azalir). Bos
    liste = kural %100 saglaniyor - bkz. enforce_min_spot_occupancy()."""
    zamanlar = state.data.zaman_sirali
    zaman_index = {gs: i for i, gs in enumerate(zamanlar)}
    hat_toplam = _hat_toplam_talep(state.data.demands)
    ihlaller = []
    for (src, dst, gun, slot, arac_turu), desi in state.leg_spot_desi.items():
        if desi <= 1e-9:
            continue
        idx = zaman_index.get((gun, slot))
        if idx is None:
            continue
        kap = state.data.arac_parametreleri[arac_turu]["kapasite_desi"]
        ust_sinir = min(1.0, hat_toplam.get((src, dst), 0.0) / kap) if kap else 1.0
        esik = _min_doluluk_esigi(idx, len(zamanlar), ust_sinir)
        if esik <= 1e-9:
            continue
        adet = spot_vehicle_count(desi, kap, MAX_SPOT)
        if adet <= 0:
            continue
        doluluk = desi / (adet * kap)
        if doluluk < esik - 1e-9:
            ihlaller.append({
                "src": src, "dst": dst, "gun": gun, "slot": slot, "arac_turu": arac_turu,
                "desi": desi, "adet": adet, "kapasite": kap, "doluluk": doluluk, "esik": esik,
            })
    return ihlaller


def _insert_chunk_from(state, hat, baslangic_idx, desi, rng, talep_id, gercek_gun, gercek_slot):
    """_insert_chunk ile ayni mantik, TEK farkla: arama baslangici (baslangic_idx)
    ile SLA icin kullanilan GERCEK talep olusum ani (gercek_gun/gercek_slot)
    BIRBIRINDEN BAGIMSIZ. _insert_chunk'ta bu ikisi ayni (gun,slot) parametresine
    baglidir - bu yuzden bir kargoyu zorla ILERI bir slottan aratmak, o slotta
    yerlesirse SLA'yi YANLIŞLIKLA "sanki orada olusmus gibi" (yani gecikme YOKMUS
    gibi) hesaplardi. enforce_min_spot_occupancy() bu fonksiyonu kullanir ki
    zorunlu erteleme GERCEK SLA cezasina dogru yansisin."""
    zamanlar = state.data.zaman_sirali
    kalan = desi

    for aktif_gun, aktif_slot in zamanlar[baslangic_idx:]:
        if kalan <= 1e-6:
            break

        secenekler = list(insertion_options(state.data, hat, aktif_gun, aktif_slot))
        while kalan > 1e-6:
            en_iyi_secenek = None
            en_iyi_desi = 0
            en_iyi_birim_maliyet = float('inf')
            en_iyi_eval = None

            for (secenek, milk_run) in secenekler:
                eval_sonuc = evaluate_path(
                    state, hat, aktif_gun, aktif_slot, kalan, secenek, talep_id,
                    demand_gun=gercek_gun, demand_slot=gercek_slot,
                    milk_run=milk_run,
                )
                if eval_sonuc is None or eval_sonuc['desi'] <= 1e-9:
                    continue
                birim_maliyet = eval_sonuc['maliyet'] / eval_sonuc['desi']
                if eval_sonuc['desi'] > en_iyi_desi or \
                   (abs(eval_sonuc['desi'] - en_iyi_desi) < 1e-9 and birim_maliyet < en_iyi_birim_maliyet):
                    en_iyi_desi = eval_sonuc['desi']
                    en_iyi_birim_maliyet = birim_maliyet
                    en_iyi_secenek = secenek
                    en_iyi_eval = eval_sonuc

            if en_iyi_secenek is None:
                break

            sla_cost = en_iyi_eval.get("sla_cost")

            commit_path(state, hat, en_iyi_eval, talep_id, demand_gun=gercek_gun, demand_slot=gercek_slot,sla_cost=sla_cost)
            kalan -= en_iyi_desi

    return kalan


def enforce_min_spot_occupancy(state: "State", rng) -> "State":
    """Minimum doluluk esigini (bkz. _min_doluluk_esigi - son
    MIN_SPOT_DOLULUK_TAPER_GUN_SAYISI gun boyunca kademeli olarak 0'a iner)
    KESIN (hard) olarak uygular - hicbir parasal ceza icermez, objective()'e
    hicbir katki yapmaz.

    Esigin altinda kalan spot arac gruplarini bulur, o gruplari kullanan
    atamalari soker ve ihlal edilen (gun,slot)'tan KESINLIKLE SONRAKI bir
    slottan itibaren yeniden dener. "Kesinlikle sonraki slottan itibaren"
    onemli: ayni slotta yeniden denenirse arama hala en ucuz secenek oldugu
    icin AYNI ihlali tekrar uretebilir (sonsuz donguye girer) - ileri zorlamak,
    talebin HER turda en az bir slot ilerlemesini garanti eder, bu da en fazla
    len(zaman_sirali) turda KESIN yakinsamayi saglar (en kotu durumda talep,
    esigin 0'a indigi SON slota kadar itilir - README: "karsilanmayan yuk bir
    sonraki gune ertelenir" kuralinin ta kendisi, ama artik TEK bir slota
    degil, son birkac gune YAYILARAK)."""
    zamanlar = state.data.zaman_sirali
    zaman_index = {gs: i for i, gs in enumerate(zamanlar)}
    tur_limiti = len(zamanlar)

    for _ in range(tur_limiti):
        ihlaller = dogrula_min_spot_doluluk(state)
        if not ihlaller:
            break
        ihlal_keyleri = {(v["src"], v["dst"], v["gun"], v["slot"], v["arac_turu"]) for v in ihlaller}

        etkilenen = [
            a for a in state.assignments
            if any(
                (leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu) in ihlal_keyleri and not leg.is_kiralik
                for leg in a.legs
            )
        ]
        if not etkilenen:
            break  # ihlal var ama eslesen atama yok (olmamali) - guvenlik agi

        # PERFORMANS: state.assignments.remove(a) DEGER esitligiyle (frozen
        # dataclass'in otomatik __eq__'i, ic ice legs tuple'i dahil) O(n)
        # tarama yapiyordu - binlerce elemanli listede yuzlerce silme
        # BASINA bunu tekrarlamak pratikte O(n^2) oluyordu (profil ile
        # dogrulandi: list.remove() + __eq__ TEK BASINA bir run'in ~%15-20'sini
        # yiyordu). id() bazli TEK GECISLI toplu silmeye cevrildi - O(n) toplam.
        etkilenen_id = {id(a) for a in etkilenen}
        state.assignments = [a for a in state.assignments if id(a) not in etkilenen_id]
        for a in etkilenen:
            _remove_assignment(state, a)
        # _remove_assignment() sokulen HER atamayi kendisi state.unassigned'a
        # geri ekler (normal sozlesmesi budur - repair operatorlerinin havuzdan
        # cekmesini bekler). Burada reinsert'i KENDIMIZ (asagida) yaptigimiz
        # icin, o otomatik eklenen (hayalet) girdileri temizliyoruz - yoksa
        # ayni talep hem gercek yeni atama olarak hem de unassigned'da
        # COKLANMIS olarak kalir (bkz. sohbet gecmisi: "100 desi 350 oldu" bulgusu).
        del state.unassigned[-len(etkilenen):]

        for a in etkilenen:
            zorunlu_idx = 1 + max(
                zaman_index[(leg.gun, leg.slot)]
                for leg in a.legs
                if (leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu) in ihlal_keyleri and not leg.is_kiralik
            )
            zorunlu_idx = min(zorunlu_idx, len(zamanlar) - 1)
            print(f"DEBUG minspot bump: {a.talep_id} eski=({a.legs[0].gun},{a.legs[0].slot}) yeni_min_idx={zorunlu_idx} ({zamanlar[zorunlu_idx]}) desi={a.desi:.0f}")
            kalan = _insert_chunk_from(
                state, a.demand_hat, zorunlu_idx, a.desi, rng, a.talep_id,
                a.demand_gun, a.demand_slot,
            )
            if kalan > 1e-6:
                # Zorunlu ILERI arama basarisiz oldu - kalan miktar icin GERCEK
                # kapasite (elleçleme/tir/kiralik stok) tukenmis demektir, bu
                # doluluk kuralindan BAGIMSIZ bir kisit. Bu durumda esigi
                # karsilamasa bile FIZIKSEL OLARAK SIGAN herhangi bir yere
                # yerlestirmek (standart _insert_chunk ile, orijinal olusum
                # anindan itibaren TUM slotlari tekrar dener), tamamen
                # yerlestirilememekten (devasa 30-gunluk SLA cezasi, bkz.
                # objective()) HER ZAMAN daha iyidir - bkz. sohbet gecmisi:
                # "155 satir / 1.43M TL yerlestirilemeyen talep" bulgusu.
                kalan = _insert_chunk(state, a.demand_hat, a.demand_gun, a.demand_slot, kalan, rng, a.talep_id)
            if kalan > 1e-6:
                state.unassigned.append((a.demand_hat, a.demand_gun, a.demand_slot, kalan, a.talep_id))

    kalan_ihlaller = dogrula_min_spot_doluluk(state)
    if kalan_ihlaller:
        print(
            f"UYARI: {len(kalan_ihlaller)} spot arac grubu {tur_limiti} zorunlu turdan sonra "
            f"HALA MIN_SPOT_DOLULUK_ORANI (%{MIN_SPOT_DOLULUK_ORANI*100:.0f}) altinda - "
            "bkz. dogrula_min_spot_doluluk()."
        )
    return state


# ============================================================================
# Gercek (elleçleme dahil) kapasite dogrulamasi ve zorunlu duzeltme
# ============================================================================
# arrival_day() (time_model.py) kapasiteyi nominal kalkis saati + yol suresine
# gore gune yaziyor, cikis elleçleme suresini atliyor - buyuk yuklerde bu sure
# gece yarisini asinca arac gercekte ertesi gune dusuyor ama arama sirasindaki
# kapasite takibi bunu yakalayamiyor. Asagidaki fonksiyonlar, State.cikis_
# ellecleme_desi/varis_ellecleme_desi'yi (milk-run dokunus ayrimini koruyan,
# bucket TOPLAMindan farkli sozlukler) kullanarak ALNS bittikten sonra gercek
# zamana gore kapasiteyi yeniden dogrular ve ihlalleri zorla erteler - arama
# mantigina dokunmaz (bkz. enforce_min_spot_occupancy ile ayni desen).
def _gercek_bacak_zamanlari(data: "ProblemData", key5: tuple, cikis_desi: float) -> tuple:
    """key5 = (src,dst,gun,slot,arac_turu). Kalkis, cikis_desi'nin (milk-run
    skip_src disarida) elleçleme suresi kadar gecikir; 0 ise gecikmez."""
    src, dst, gun, slot, arac_turu = key5
    slot_zamani = slot_datetime(gun, slot)
    if cikis_desi > 1e-9:
        kalkis = ellecleme_tamamlanma_zamani(slot_zamani, cikis_desi, consolidation=False)
    else:
        kalkis = slot_zamani
    seyir = seyir_suresi_saat(data.route_lookup, src, dst, arac_turu)
    varis = varis_zamani(kalkis, seyir)
    return kalkis, varis


def gercek_kapasite_kullanimlari(state: "State") -> tuple[dict, dict, dict]:
    """Gercek (elleçleme dahil) zamana gore elleçleme/tir kapasitesi kullanimini
    hesaplar. Dönüş: (handling_real, tir_real, bucket_zaman) - ucuncusu
    real_key -> (kalkis, varis), enforce_real_capacity_limits icin."""
    data = state.data
    handling_real: dict = {}
    tir_real: dict = {}
    bucket_zaman: dict = {}

    tum_keyler = (
        set(state.cikis_ellecleme_desi) | set(state.varis_ellecleme_desi)
        | {k + (False,) for k in state.leg_spot_desi if state.leg_spot_desi[k] > 1e-9}
        | {k + (True,) for k in state.leg_kiralik_desi if state.leg_kiralik_desi[k] > 1e-9}
    )

    for real_key in tum_keyler:
        src, dst, gun, slot, arac_turu, is_kiralik = real_key
        key5 = (src, dst, gun, slot, arac_turu)
        cikis_desi = state.cikis_ellecleme_desi.get(real_key, 0.0)
        varis_desi = state.varis_ellecleme_desi.get(real_key, 0.0)
        kalkis, varis = _gercek_bacak_zamanlari(data, key5, cikis_desi)
        bucket_zaman[real_key] = (kalkis, varis)

        if cikis_desi > 1e-9:
            slot_zamani = slot_datetime(gun, slot)
            cikis_sure_dk = (kalkis - slot_zamani).total_seconds() / 60.0
            for gun_str, pay in _ellecleme_dagilimi(slot_zamani, cikis_desi, cikis_sure_dk).items():
                handling_real[(src, gun_str)] = handling_real.get((src, gun_str), 0.0) + pay
        if varis_desi > 1e-9:
            varis_sure_dk = ellecleme_suresi_dakika(varis_desi, consolidation=False)
            for gun_str, pay in _ellecleme_dagilimi(varis, varis_desi, varis_sure_dk).items():
                handling_real[(dst, gun_str)] = handling_real.get((dst, gun_str), 0.0) + pay

        if arac_turu != data.tir_arac_turu:
            continue
        if is_kiralik:
            adet = data.kiralik_stok_gunluk.get(((src, dst), arac_turu), 0)
        else:
            bucket_desi = state.leg_spot_desi.get(key5, 0.0)
            if bucket_desi <= 1e-9:
                continue
            kap = data.arac_parametreleri[arac_turu]["kapasite_desi"]
            adet = spot_vehicle_count(bucket_desi, kap, MAX_SPOT)
        if adet <= 0:
            continue
        tir_real[(src, kalkis.date().isoformat())] = tir_real.get((src, kalkis.date().isoformat()), 0) + adet
        tir_real[(dst, varis.date().isoformat())] = tir_real.get((dst, varis.date().isoformat()), 0) + adet

    # Kiralik zorunlu her gun kalkar - o gun desi=0 olsa da tir kapasitesi tuketir.
    if data.tir_arac_turu is not None:
        for (hat, arac_turu), stok in data.kiralik_stok_gunluk.items():
            if arac_turu != data.tir_arac_turu or stok <= 0:
                continue
            src, dst = hat
            for gun in data.gunler:
                real_key = (src, dst, gun, KIRALIK_DISPATCH_SLOT, arac_turu, True)
                if real_key in tum_keyler:
                    continue  # yukarida zaten islendi
                kalkis = slot_datetime(gun, KIRALIK_DISPATCH_SLOT)
                seyir = seyir_suresi_saat(data.route_lookup, src, dst, arac_turu)
                varis = varis_zamani(kalkis, seyir)
                bucket_zaman[real_key] = (kalkis, varis)
                tir_real[(src, kalkis.date().isoformat())] = tir_real.get((src, kalkis.date().isoformat()), 0) + stok
                tir_real[(dst, varis.date().isoformat())] = tir_real.get((dst, varis.date().isoformat()), 0) + stok

    return handling_real, tir_real, bucket_zaman


def dogrula_gercek_kapasite(state: "State") -> dict:
    """gercek_kapasite_kullanimlari()'ni kapasite limitleriyle karsilastirip
    ihlal eden (tm, gun) ciftlerini dondurur: {"handling": [...], "tir": [...]}."""
    data = state.data
    handling_real, tir_real, _ = gercek_kapasite_kullanimlari(state)
    handling_ihlal = [
        {"tm": tm, "gun": gun, "kullanim": kullanim, "kapasite": data.handling_capacity[tm]}
        for (tm, gun), kullanim in handling_real.items()
        if tm in data.handling_capacity and kullanim > data.handling_capacity[tm] + 1e-6
    ]
    tir_ihlal = [
        {"tm": tm, "gun": gun, "kullanim": kullanim, "kapasite": data.tir_capacity[tm]}
        for (tm, gun), kullanim in tir_real.items()
        if tm in data.tir_capacity and kullanim > data.tir_capacity[tm] + 1e-6
    ]
    return {"handling": handling_ihlal, "tir": tir_ihlal}


def _yeni_atama_ihlal_cezasi(state: "State", yeni_atamalar: list, handling_asim: dict, tir_asim: dict) -> float:
    """yeni_atamalar (bu adayda YENİ eklenen Assignment'lar) icin, HER bacagin
    GERCEK cikis/varis gununun, TURUN BASINDA olculen ihlal kumesine
    (handling_asim/tir_asim) denk gelip gelmedigini kontrol edip objective()'in
    nominal kapasite asimlarina uyguladigi AYNI olcekte (x1000/x50000)
    cezalandirir. SADECE bu adayin KENDI (genelde 1-3) bacagina baktigi icin,
    tum state'i yeniden tarayan gercek_kapasite_kullanimlari()'ndan COK daha
    ucuzdur (performans: butun buketleri her adayda yeniden taramak 400s+
    butceyi asiyordu)."""
    if not handling_asim and not tir_asim:
        return 0.0
    ceza = 0.0
    data = state.data
    for a in yeni_atamalar:
        n = len(a.legs)
        for i, leg in enumerate(a.legs):
            if leg.is_kiralik:
                continue
            real_key = (leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu, False)
            cikis_desi = state.cikis_ellecleme_desi.get(real_key, 0.0)
            key5 = (leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu)
            kalkis, varis = _gercek_bacak_zamanlari(data, key5, cikis_desi)
            skip_src = a.milk_run and i > 0
            skip_dst = a.milk_run and i < n - 1
            cikis_gun = kalkis.date().isoformat()
            varis_gun = varis.date().isoformat()
            if not skip_src and (leg.src, cikis_gun) in handling_asim:
                ceza += a.desi * 1000.0
            if not skip_dst and (leg.dst, varis_gun) in handling_asim:
                ceza += a.desi * 1000.0
            if leg.arac_turu == data.tir_arac_turu:
                if (leg.src, cikis_gun) in tir_asim:
                    ceza += 50000.0
                if (leg.dst, varis_gun) in tir_asim:
                    ceza += 50000.0
    return ceza


def _yeniden_yerlestirme_maliyet_ekle(
    state: "State", a: Assignment, idx: int, rng, handling_asim: dict, tir_asim: dict
) -> float:
    """`a`yi idx'ten itibaren state UZERINDE (kopyasiz) yeniden yerlestirir,
    eklenen marjinal maliyeti (yeni atamalarin arac+SLA maliyeti + GERCEK
    ihlal cezasi, yerlesemezse agir sanal ceza) dondurur. Cagiran taraf secim
    icin KENDI kopyasini vermelidir."""
    onceki_sayisi = len(state.assignments)
    kalan = _insert_chunk_from(state, a.demand_hat, idx, a.desi, rng, a.talep_id, a.demand_gun, a.demand_slot)
    if kalan > 1e-6:
        kalan = _insert_chunk(state, a.demand_hat, a.demand_gun, a.demand_slot, kalan, rng, a.talep_id)
    maliyet = 0.0
    if kalan > 1e-6:
        state.unassigned.append((a.demand_hat, a.demand_gun, a.demand_slot, kalan, a.talep_id))
        maliyet += sla_cezasi_tl(kalan, 24 * 30)
    yeni_atamalar = state.assignments[onceki_sayisi:]
    for yeni in yeni_atamalar:
        maliyet += yeni.vehicle_cost + _fresh_sla_cost(state, yeni)
    maliyet += _yeni_atama_ihlal_cezasi(state, yeni_atamalar, handling_asim, tir_asim)
    return maliyet


def _en_iyi_yeniden_yerlestirme(
    state: "State", a: Assignment, zorunlu_idx: int, zamanlar: list, rng, handling_asim: dict, tir_asim: dict
) -> "State":
    """`a`yi zorunlu_idx'ten itibaren birkac FARKLI baslangic noktasindan
    yeniden yerlestirmeyi dener - hemen sonraki slot her zaman en ucuz
    olmuyor (sikisik bir hub'a tekrar tekrar yonlendirme riski), biraz daha
    ileri bir zaman cok daha ucuza gelebilir. Marjinal maliyeti (bu turun
    ihlal kumesini yeniden yaratan bir adayi elemek icin GERCEK ihlal cezasi
    dahil) en dusuk olan denemeyi kalici state olarak dondurur."""
    adaylar_idx = sorted({min(zorunlu_idx + adim, len(zamanlar) - 1) for adim in (0, 1, 2, 4)})
    en_iyi_state = None
    en_iyi_maliyet = float("inf")
    for idx in adaylar_idx:
        deneme = state.copy()
        maliyet = _yeniden_yerlestirme_maliyet_ekle(deneme, a, idx, rng, handling_asim, tir_asim)
        if maliyet < en_iyi_maliyet:
            en_iyi_maliyet = maliyet
            en_iyi_state = deneme
    return en_iyi_state


def enforce_real_capacity_limits(state: "State", rng) -> "State":
    """Gercek (elleçleme dahil) zamana gore ihlal eden TM/gun'lari bulur ve
    sadece asimi kapatmaya yetecek kadar spot atamayi - kucukten buyuge secerek,
    gereksiz sevkiyat ertelemesini onlemek icin - soker (kiralik zorunlu/sabit
    oldugundan sokulemez); enforce_min_spot_occupancy ile ayni ileri-zorlama
    mantigiyla bir sonraki uygun slottan itibaren yeniden yerlestirir.

    Saf kiralik kaynakli ihlaller (sokulecek spot atama yoksa) veri setindeki
    cozulemez bir celiskiyi isaret eder - sadece raporlanir, zorla duzeltilmez."""
    zamanlar = state.data.zaman_sirali
    zaman_index = {gs: i for i, gs in enumerate(zamanlar)}
    tur_limiti = len(zamanlar)
    data = state.data

    for _ in range(tur_limiti):
        handling_real, tir_real, bucket_zaman = gercek_kapasite_kullanimlari(state)
        handling_asim = {
            (tm, gun): kullanim - data.handling_capacity[tm]
            for (tm, gun), kullanim in handling_real.items()
            if tm in data.handling_capacity and kullanim > data.handling_capacity[tm] + 1e-6
        }
        tir_asim = {
            (tm, gun): kullanim - data.tir_capacity[tm]
            for (tm, gun), kullanim in tir_real.items()
            if tm in data.tir_capacity and kullanim > data.tir_capacity[tm] + 1e-6
        }
        if not handling_asim and not tir_asim:
            break

        etkilenen_map: dict = {}  # (tur,tm,gun) -> {id(a): a}
        for a in state.assignments:
            n = len(a.legs)
            for i, leg in enumerate(a.legs):
                if leg.is_kiralik:
                    continue  # kiralik zorunlu/sabit - sokulemez
                real_key = (leg.src, leg.dst, leg.gun, leg.slot, leg.arac_turu, False)
                kalkis, varis = bucket_zaman.get(real_key, (None, None))
                if kalkis is None:
                    continue
                skip_src = a.milk_run and i > 0
                skip_dst = a.milk_run and i < n - 1
                cikis_gun = kalkis.date().isoformat()
                varis_gun = varis.date().isoformat()
                if not skip_src and (leg.src, cikis_gun) in handling_asim:
                    etkilenen_map.setdefault(("h", leg.src, cikis_gun), {})[id(a)] = a
                if not skip_dst and (leg.dst, varis_gun) in handling_asim:
                    etkilenen_map.setdefault(("h", leg.dst, varis_gun), {})[id(a)] = a
                if leg.arac_turu == data.tir_arac_turu:
                    if (leg.src, cikis_gun) in tir_asim:
                        etkilenen_map.setdefault(("t", leg.src, cikis_gun), {})[id(a)] = a
                    if (leg.dst, varis_gun) in tir_asim:
                        etkilenen_map.setdefault(("t", leg.dst, varis_gun), {})[id(a)] = a

        # Her ihlal icin kucukten buyuge, asimi kapatmaya yetecek kadar sok.
        sokulecek: dict = {}
        asim_hepsi = [("h", tm, gun, deger) for (tm, gun), deger in handling_asim.items()]
        asim_hepsi += [("t", tm, gun, deger) for (tm, gun), deger in tir_asim.items()]
        for tur, tm, gun, asim_deger in asim_hepsi:
            adaylar = etkilenen_map.get((tur, tm, gun), {})
            if not adaylar:
                continue
            kalan_asim = asim_deger
            for aid, a in adaylar.items():
                if aid in sokulecek:
                    kalan_asim -= (a.desi if tur == "h" else 1)
            if kalan_asim <= 1e-9:
                continue
            kalan_adaylar = sorted(
                (a for aid, a in adaylar.items() if aid not in sokulecek),
                key=lambda a: a.desi,
            )
            for a in kalan_adaylar:
                if kalan_asim <= 1e-9:
                    break
                sokulecek[id(a)] = a
                kalan_asim -= (a.desi if tur == "h" else 1)

        if not sokulecek:
            break  # kalan ihlaller sadece kiralik kaynakli - sokulup duzeltilemez

        etkilenen = list(sokulecek.values())
        state.assignments = [a for a in state.assignments if id(a) not in sokulecek]
        for a in etkilenen:
            _remove_assignment(state, a)
        del state.unassigned[-len(etkilenen):]

        for a in etkilenen:
            zorunlu_idx = min(zaman_index[(a.demand_gun, a.demand_slot)] + 1, len(zamanlar) - 1)
            for leg in a.legs:
                zorunlu_idx = max(zorunlu_idx, min(zaman_index.get((leg.gun, leg.slot), 0) + 1, len(zamanlar) - 1))
            state = _en_iyi_yeniden_yerlestirme(state, a, zorunlu_idx, zamanlar, rng, handling_asim, tir_asim)

    kalan_ihlaller = dogrula_gercek_kapasite(state)
    if kalan_ihlaller["handling"] or kalan_ihlaller["tir"]:
        print(
            f"UYARI: gercek-zaman dogrulamasi sonrasi hala {len(kalan_ihlaller['handling'])} elleçleme "
            f"+ {len(kalan_ihlaller['tir'])} tir ihlali var (muhtemelen zorunlu kiralik kaynakli, "
            "veri setinde cozulemez bir celiski olabilir - bkz. dogrula_gercek_kapasite())."
        )
    return state


def random_removal(state: State, rng: rnd.Generator, **kwargs) -> State:
    state = state.copy()
    if not state.assignments:
        return state
    # DENEME GERI ALINDI: zamanla buyuyen yikim orani denendi ama olcumde
    # sonucu KOTULESTIRDI (21.5M -> 22.5M, ayni 450 sn butcede) - buyuyen
    # yikim, repair'in isini agirlastirip erken/kritik fazda iterasyon
    # sayisini dusurdu (bkz. sohbet gecmisi). Sabit kucuk orana donuldu.
    n = max(1, int(0.04 * len(state.assignments)))
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
    # KUCULTULDU (0.20 -> 0.08): zamanla buyuyen versiyon denendi, kotulestirdi
    # (bkz. random_removal'daki not) - sabit orana donuldu.
    max_removal_count = int(len(state.assignments) * 0.08) + 1
    num_to_remove = min(len(candidates_to_remove), max_removal_count)

    # Adaylar arasından rastgele bir kısmını seç (Böylece model her seferinde farklı kombinasyonlar dener)
    to_remove = rng.choice(candidates_to_remove, num_to_remove, replace=False)

    # 5. Seçilen atamaları State'ten çıkar
    # DUZELTME: _remove_assignment() sokulen atamayi KENDISI zaten
    # state.unassigned'a ekliyor (bkz. tanimi) - burada AYRICA elle
    # eklemek, ayni kargoyu unassigned havuzuna 2 KERE koyuyordu; repair
    # operatorleri unassigned'i sirayla islediginden aynen desi 2 kere
    # yerlestiriliyordu (kargo korunumu ihlali - bkz. sohbet gecmisi: "100
    # desi 350 oldu" bulgusu, ayni hatanin enforce_min_spot_occupancy'de
    # bilincli sekilde temizlendigi yer).
    # PERFORMANS: bkz. enforce_min_spot_occupancy'deki ayni duzeltmenin notu -
    # id() bazli tek gecisli toplu silme, deger esitligiyle O(n) x O(k) yerine.
    to_remove_id = {id(a) for a in to_remove}
    state.assignments = [a for a in state.assignments if id(a) not in to_remove_id]
    for a in to_remove:
        _remove_assignment(state, a)

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
    # toplam atamaların %3'ü ile %8'i arasında bir kısmını sökeceğiz.
    # KUCULTULDU (0.10-0.20 -> 0.03-0.08): zamanla buyuyen versiyon denendi,
    # kotulestirdi (bkz. random_removal'daki not) - sabit orana donuldu.
    min_remove = max(1, int(len(state.assignments) * 0.03))
    max_remove = max(2, int(len(state.assignments) * 0.08))
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

    # 5. SÖKME (Removal) İŞLEMİ
    # DUZELTME: _remove_assignment() sokulen atamayi KENDISI zaten
    # state.unassigned'a ekliyor - elle tekrar eklemek cift sayima
    # (kargo korunumu ihlaline) yol aciyordu, bkz. low_occupancy_removal'daki
    # ayni duzeltmenin notu.
    to_remove_id = {id(a) for a in to_remove}
    state.assignments = [a for a in state.assignments if id(a) not in to_remove_id]
    for a in to_remove:
        _remove_assignment(state, a)

    return state

def worst_removal(state: State, rng: rnd.Generator, **kwargs) -> State:
    state = state.copy()
    if not state.assignments:
        return state
    # KUCULTULDU (0.10 -> 0.04): zamanla buyuyen versiyon denendi, kotulestirdi
    # (bkz. random_removal'daki not) - sabit orana donuldu.
    n = max(1, int(0.04 * len(state.assignments)))
    # DUZELTME: a.sla_cost (donmus) yerine _fresh_sla_cost - aksi halde "en
    # kotu" siralama, ayni bacaga sonradan binen yukle gecikmesi buyumus
    # atamalari sistematik olarak KACIRIP sokulmeye aday bile gostermezdi.
    ranked = sorted(state.assignments, key=lambda a: -(_fresh_sla_cost(state, a) + a.vehicle_cost))
    to_remove = ranked[:n]
    to_remove_id = {id(a) for a in to_remove}
    state.assignments = [a for a in state.assignments if id(a) not in to_remove_id]
    for a in to_remove:
        _remove_assignment(state, a)
    return state


def tm_overload_removal(state: State, rng: rnd.Generator, **kwargs) -> State:
    """Sadece esigi (%85) FIILEN asan (tm, gun) ciftindeki yuku hedef alir.

    DUZELTME: eskiden aday secimi (tm, gun) bazinda yapilip, sokme filtresi
    SADECE tm bazinda (gun'den BAGIMSIZ) uygulaniyordu - bu da bir TM'nin
    SADECE 1 gununde asim varsa, o TM'ye dokunan ama HICBIR asim OLMAYAN
    baska gunlerdeki BASARILI atamalarin da sokulme riskine girmesine yol
    aciyordu (bkz. sohbet gecmisi). Artik to_remove filtresi de ayni (tm, gun)
    ciftine gore uygulaniyor - sadece gercekten asan gun etkileniyor."""
    state = state.copy()
    aday_tm_gun = [
        (tm, gun)
        for tm in state.data.merkezler
        for gun in state.data.gunler
        if state.data.handling_capacity.get(tm) is not None
        and state.handling_usage.get((tm, gun), 0.0) > 0.85 * state.data.handling_capacity[tm]
    ]
    if not aday_tm_gun:
        return random_removal(state, rng, **kwargs)
    tm, gun = aday_tm_gun[int(rng.integers(0, len(aday_tm_gun)))]
    to_remove = [
        a for a in state.assignments
        if any(leg.gun == gun and (leg.src == tm or leg.dst == tm) for leg in a.legs)
    ]
    rng.shuffle(to_remove)
    to_remove = to_remove[: max(1, len(to_remove) // 3)]
    to_remove_id = {id(a) for a in to_remove}
    state.assignments = [a for a in state.assignments if id(a) not in to_remove_id]
    for a in to_remove:
        _remove_assignment(state, a)
    return state


def regret_repair(state: State, rng: rnd.Generator, **kwargs) -> State:
    state = state.copy()
    unassigned_items = list(state.unassigned)
    state.unassigned = []
    zamanlar = state.data.zaman_sirali

    # --- HAFİF DELTA HESABI İÇİN YARDIMCI ---
    def _delta_objective(orijinal_state, deneme_state, yeni_assignments, yerlesen_desi):
        """Deneme state'inin objective farkını, tam objective() çağırmadan hesapla."""
        delta = 0.0
        
        # Araç maliyeti farkı
        delta += deneme_state._arac_maliyeti_toplam - orijinal_state._arac_maliyeti_toplam
        
        # Yeni eklenen SLA maliyetleri - DUZELTME: a.sla_cost (donmus) yerine
        # _fresh_sla_cost (deneme_state, insert sonrasi bucket yuklerini zaten
        # yansitiyor - bkz. o fonksiyonun docstring'i).
        for a in yeni_assignments:
            delta += _fresh_sla_cost(deneme_state, a)
        
        # Unassigned'dan kurtulan desi (30 günlük cezayı artık ödemiyoruz)
        if yerlesen_desi > 1e-6:
            delta -= sla_cezasi_tl(yerlesen_desi, 24 * 30)
        
        return delta

    # 1. Aşama: Regret skorlarını hesapla
    regret_scores = []
    
    for item in unassigned_items:
        hat, orj_gun, orj_slot, orj_desi, talep_id = item

        # Kargonun zaman çizelgesindeki başlangıç noktasını bul
        baslangic_idx = 0
        for z_idx, (g, s) in enumerate(zamanlar):
            if g == orj_gun and s == orj_slot:
                baslangic_idx = z_idx
                break

        # --- Opsiyon 1: Orijinal vaktinde yerleştirme ---
        onceki_assign_sayisi = len(state.assignments)
        deneme_1 = state.copy()
        gun = zamanlar[baslangic_idx][0]
        slot = zamanlar[baslangic_idx][1]
        kalan_1 = _insert_chunk(deneme_1, hat, gun, slot, orj_desi, rng, talep_id)
        yerlesen_1 = orj_desi - kalan_1
        yeni_assignments_1 = deneme_1.assignments[onceki_assign_sayisi:]
        
        if yerlesen_1 > 1e-6:
            maliyet_1 = _delta_objective(state, deneme_1, yeni_assignments_1, yerlesen_1) / yerlesen_1
        else:
            maliyet_1 = float('inf')

        # --- Opsiyon 2: Bir sonraki slotta yerleştirme ---
        maliyet_2 = float('inf')
        if baslangic_idx + 1 < len(zamanlar):
            onceki_assign_sayisi = len(state.assignments)
            deneme_2 = state.copy()
            kalan_2 = _insert_chunk(deneme_2, hat, zamanlar[baslangic_idx + 1][0], 
                                    zamanlar[baslangic_idx + 1][1], orj_desi, rng, talep_id)
            yerlesen_2 = orj_desi - kalan_2
            yeni_assignments_2 = deneme_2.assignments[onceki_assign_sayisi:]
            
            if yerlesen_2 > 1e-6:
                maliyet_2 = _delta_objective(state, deneme_2, yeni_assignments_2, yerlesen_2) / yerlesen_2

        # --- Regret hesabı ---
        if maliyet_1 == float('inf'):
            regret = -1.0
        elif maliyet_2 == float('inf'):
            regret = 1e9
        else:
            regret = maliyet_2 - maliyet_1

        agirlikli_regret = regret * orj_desi if regret > 0 else regret
        regret_scores.append((agirlikli_regret, item, baslangic_idx))

    # 2. ve 3. Aşama: Sırala ve yerleştir (değişiklik yok)
    regret_scores.sort(key=lambda x: x[0], reverse=True)
    
    for score, item, baslangic_idx in regret_scores:
        hat, orj_gun, orj_slot, orj_desi, talep_id = item
        kalan_desi = orj_desi
        
        for idx in range(baslangic_idx, len(zamanlar)):
            aktif_gun, aktif_slot = zamanlar[idx]
            kalan_desi = _insert_chunk(
                state, hat, aktif_gun, aktif_slot, kalan_desi, rng, talep_id
            )
            if kalan_desi <= 1e-6:
                break
                
        if kalan_desi > 1e-6:
            state.unassigned.append((hat, orj_gun, orj_slot, kalan_desi, talep_id))
            
    return state

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

    hat_counts: dict = {} # Unassigned içindeki hatların sayıları. Örn: İst-Ankara hattı 3 kere var.
    for (hat, *_rest) in state.unassigned: # _rest listesinin unassigned tuple'ındaki kullanmayacağımız şeyleri attık. 
        hat_counts[hat] = hat_counts.get(hat, 0) + 1
    target_hat = max(hat_counts, key=hat_counts.get) # En çok unassigned'ı olan hattı al.

    hat_items = [it for it in state.unassigned if it[0] == target_hat] # target hattın itemleri
    other_items = [it for it in state.unassigned if it[0] != target_hat] # target hatta ait olmayan itemler
    state.unassigned = other_items

    data = state.data
    src, dst = target_hat
    talep = {} # (gun,slot) zamanında ne kadar desi talebi var? sorusunu cevaplar
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

    kiralik_x, spot_y, ertelenen, biriken = {}, {}, {}, {}
    for (g, s) in zaman_sirali:
        for a in data.arac_turleri:
            stok = data.kiralik_stok_gunluk.get((target_hat, a), 0) if s == KIRALIK_DISPATCH_SLOT else 0
            kiralik_x[(g, s, a)] = model.NewIntVar(0, stok, f"kx_{g}_{s}_{a}")
            spot_y[(g, s, a)] = model.NewIntVar(0, MAX_SPOT, f"sy_{g}_{s}_{a}")
        ertelenen[(g, s)] = model.NewIntVar(0, max_talep, f"ert_{g}_{s}")
        biriken[(g, s)] = model.NewIntVar(0, max_talep, f"bir_{g}_{s}")

    # Paylasimli elleceleme kapasitesi: (TM, gun) basina TUM slotlarin toplam
    # katkisi, o an diger hatlarin kullandigi miktar dusuldukten sonra kalan
    # paya sigmali. Once tum slotlarin yuk terimlerini (TM,gun) bazinda topluyoruz,
    # sonra TEK bir kisit ekliyoruz (slot slot ayri kisitlamak yanlis olurdu -
    # ayni gunun iki slotu ayni kapasiteyi paylasir).

    handling_terimleri_by_tm_gun: dict = {} # "Bu TM'de bu günde bu kadar ellecleme yapılıyor" diyoruz.
    yuk_dict = {} # YENİ EKLENEN SÖZLÜK
    for idx, (g, s) in enumerate(zaman_sirali):
        demand_of_today = int(round(talep.get((g, s), 0.0)))
        if idx == 0:
            model.Add(biriken[(g, s)] == demand_of_today)
        else:
            g0, s0 = zaman_sirali[idx - 1]
            model.Add(biriken[(g, s)] == ertelenen[(g0, s0)] + demand_of_today)

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
        model.Add(biriken[(g, s)] == cp_model.LinearExpr.Sum(yuk_terimleri) + ertelenen[(g, s)])

    # Diger hatlarin MEVCUT kullanimi sabit kabul edilip kalan pay bu hatta
    # (tum slotlarin TOPLAMI uzerinden, gun bazinda) kisitlaniyor.
    for (tm, gun_for_cap), terimler in handling_terimleri_by_tm_gun.items():
        if data.handling_capacity.get(tm) is None:
            continue
        kalan_kapasite = state.handling_available(tm, gun_for_cap)
        model.Add(cp_model.LinearExpr.Sum(terimler) <= int(kalan_kapasite))

    idx_son = len(zaman_sirali) - 1
    model.Add(ertelenen[zaman_sirali[idx_son]] == 0)

    maliyet = []
    for a in data.arac_turleri:
        p = data.arac_parametreleri[a]
        # Seyir maliyetini senin güncellediğin vehicle_leg_cost fonksiyonundan alıyoruz
        spot_birim_maliyet = vehicle_leg_cost(data.route_lookup, target_hat, a, p["spot_hourly"], p["spot_km"])
        
        # CP-SAT sadece tamsayı kabul ettiği için katsayıyı yuvarlıyoruz
        ellecleme_katsayisi = int(round((0.01 / 60) * p["spot_hourly"])) # desi başına elleçleme maliyeti. Burada bu değeri tam sayıya yuvarlıyoruz. Bu direkt olarak maliyeti etkiler mi?
        
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
            maliyet.append(ertelenen[(g, s)] * katsayi)
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
        slotta_tasinan_yuk = max(0.0, float(solver.Value(biriken[(g, s)]) - solver.Value(ertelenen[(g, s)])))
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
                        piece_sla_cost = _bucket_aware_sla_cost(
                            data, [leg], piece_desi, demand_gun, demand_slot, target_hat,
                            state.leg_spot_desi, state.leg_kiralik_desi,
                        )
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
                        piece_sla_cost = _bucket_aware_sla_cost(
                            data, [leg], piece_desi, demand_gun, demand_slot, target_hat,
                            state.leg_spot_desi, state.leg_kiralik_desi,
                        )
                        state.assignments.append(
                            Assignment(target_hat, demand_gun, demand_slot, piece_desi, (leg,), piece_sla_cost, piece_vehicle_cost, tid)
                        )
                    toplam_yuk -= miktar
                    remaining_slot_load -= miktar

            if toplam_yuk > 1e-6:
                for tid, piece_desi, demand_gun, demand_slot in take_from_active_queue(toplam_yuk):
                    kalan2 = _insert_chunk(state, target_hat, demand_gun, demand_slot, piece_desi, rng, tid)
                    if kalan2 > 1e-6:
                        # DUZELTME: force_insert (kapasiteyi yok sayarak zorla ekleme)
                        # KULLANMIYORUZ artik - hicbir yere sigmayan kismi, mevcut
                        # "yerlestirilemedi -> gecikme cezasi öder" mekanizmasina
                        # (unassigned) gonderiyoruz. Boylece kapasite GERCEKTEN sert
                        # bir kisit oluyor (bkz. sohbet gecmisi, Task #1).
                        state.unassigned.append((target_hat, demand_gun, demand_slot, kalan2, tid))
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
            # DUZELTME: force_insert yerine unassigned'a gonder (bkz. yukaridaki not).
            state.unassigned.append((hat2, gun2, slot2, kalan2, talep_id2))

    return state