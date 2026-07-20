from dataclasses import dataclass, field
from typing import Optional


from src.alns.cost_model import spot_vehicle_count, vehicle_leg_cost, ellecleme_maliyet_hesapla
from src.alns.time_model import (
    DEMAND_ARRIVAL_TIMES,
    KIRALIK_DISPATCH_SLOT,
    RouteLookup,
    arrival_day,
    sla_cezasi_tl,
)

from src.alns.limits import (
    MAX_2HOP_CANDIDATES,
    MAX_RELAY_CANDIDATES,
    MAX_SPOT
)

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
        self.assignments: list = [] # Assigment
        self.unassigned: list = list(data.demands)
        self.leg_spot_desi: dict = {}      # (src,dst,gun,slot,arac_turu) -> desi
        self.leg_kiralik_desi: dict = {}   # (src,dst,gun,slot,arac_turu) -> desi
        self.handling_usage: dict = {}     # (tm,gun) -> desi
        self.tir_usage: dict = {}          # (tm,gun) -> adet (spot kaynakli, kiralik ayrica sabit)
        # self._fixed_kiralik_cost = self._compute_fixed_kiralik_cost()
        self._fixed_kiralik_cost = self._kiralik_bos_seyir_maliyeti()

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
                    total += asim * 10000.0 # 10K
        if self.data.tir_arac_turu is not None:
            for tm, cap in self.data.tir_capacity.items():
                for gun in self.data.gunler:
                    kullanim = self.tir_usage.get((tm, gun), 0) + self.data.fixed_kiralik_tir_usage.get((tm, gun), 0)
                    asim = kullanim - cap
                    if asim > 0:
                        total += asim * 500000.0 # 500K
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