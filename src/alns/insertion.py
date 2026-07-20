from typing import Optional

from src.alns.alns_engine import (
    State,
    Assignment,
    Leg,
    ProblemData,
    _rank_spot_types_by_cost,
    _completion_datetime

)
from src.alns.cost_model import spot_vehicle_count, vehicle_leg_cost, ellecleme_maliyet_hesapla
from src.alns.time_model import (
    DISPATCH_SLOTS,
    DEMAND_ARRIVAL_TIMES,
    KIRALIK_DISPATCH_SLOT,
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
        est_arac_turu = _rank_spot_types_by_cost(data, (leg_src, leg_dst), desi)[0] # 0. indis, en uygun araç
        cur_gun, cur_slot = leg_departures[-1] # Son eleman, yani bir önceki iterasyonda sonraki diye eklediğimiz zaman dilimi.
        sonraki = next_dispatch_slot(data.gunler, cur_gun, cur_slot, entry[est_arac_turu])
        if sonraki is None:
            return None
        leg_departures.append(sonraki)

    leg_pairs = [
        (stops[i], stops[i + 1], leg_departures[i][0], leg_departures[i][1])
        for i in range(len(stops) - 1)
    ] # Araç bu TM'den bu TM'ye şu günde şu saatte (09:00 veya 17:00) kalkacak. (src,dest,gun,slot)

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
        if best is None: # Kiraliklarda boş yer bulunamadiysa buna bakilir
            for arac_turu in _rank_spot_types_by_cost(data, (leg_src, leg_dst), desi):
                miktar = state.max_addable_on_leg(leg_src, leg_dst, leg_gun, leg_slot, arac_turu, False)
                if miktar <= 0: # Bu araç türünde yer yoksa, devam et.
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
