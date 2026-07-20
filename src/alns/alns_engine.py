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
from src.alns.insertion import(
    try_insert_path,
    insertion_options,
    _insert_chunk,
    force_insert
)
from src.alns.domain import(
    ProblemData,
    Assignment,
    Leg,
    State
)
from src.alns.limits import (
    MAX_2HOP_CANDIDATES,
    MAX_RELAY_CANDIDATES,
    MAX_SPOT
)

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
            kalkis = ellecleme_tamamlanma_zamani(slot_zamani, desi, consolidation=False) # ilk kalkış, slot anı yerine, "slot + ellecleme süresi" zamanında yaşanıyor. 
        else:
            kalkis = max(zaman, slot_zamani)

        seyir = data.route_lookup[(leg.src, leg.dst)][leg.arac_turu]
        varis = varis_zamani(kalkis, seyir)
        cizelge.append((kalkis, varis))

        if i < len(legs) - 1:
            zaman = ellecleme_tamamlanma_zamani(varis, desi, consolidation=True) # Burada neden consolidation true?
        else:
            zaman = varis

    return cizelge


def _completion_datetime(data: ProblemData, legs: list, desi: float):
    son_varis = leg_zaman_cizelgesi(data, legs, desi)[-1][1]
    return ellecleme_tamamlanma_zamani(son_varis, desi, consolidation=False)



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
    shuffeld_unassigned_items = rng.permutation(len(items)) if items else [] 
    zamanlar = state.data.zaman_sirali
    
    for i in shuffeld_unassigned_items:
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
    talep_id_by_gs = {}   # (gun, slot) -> bu talebe katki veren ID (varsa)
    for (_, gun, slot, desi, tid) in hat_items:
        talep[(gun, slot)] = talep.get((gun, slot), 0.0) + desi
        if (gun, slot) not in talep_id_by_gs:
            talep_id_by_gs[(gun, slot)] = tid
        elif talep_id_by_gs[(gun, slot)] != tid:
            talep_id_by_gs[(gun, slot)] = ""  # farkli ID'ler karisirsa guvenli tarafta kal

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
        for a in data.arac_turleri:
            k_adet = solver.Value(kiralik_x[(g, s, a)])
            s_adet = solver.Value(spot_y[(g, s, a)])
            if k_adet <= 0 and s_adet <= 0:
                continue
            kap = data.arac_parametreleri[a]["kapasite_desi"]
            toplam_yuk = min(solver.Value(bir[(g, s)]), (k_adet + s_adet) * kap)

            if k_adet > 0 and toplam_yuk > 0:
                istenen = min(toplam_yuk, k_adet * kap)
                miktar = min(istenen, state.max_addable_on_leg(src, dst, g, s, a, True))
                if miktar > 0:
                    state._commit_leg(src, dst, g, s, a, miktar, True)
                    varis_g = arrival_day(data.route_lookup, data.gunler, target_hat, g, s, a) or g
                    # NOT: deadline burada da (optimization.py'deki gibi) kalkis
                    # slotundan (g,s) hesaplaniyor - CP-SAT'in bu kucuk alt-modeli
                    # de talebin GERCEK olusum anini (yalnizca ne zaman sevk
                    # edildigini) ayrica takip etmiyor; bu, aggregate `bir`/`ert`
                    # degiskenlerinin bilinen bir sinirlamasi (bkz. optimization.py
                    # SLA bolumundeki not). Cikis/varis ellecleme suresi burada
                    # _completion_datetime uzerinden dogru ekleniyor.
                    leg = Leg(src, dst, g, s, a, True)
                    tamamlanma = _completion_datetime(data, [leg], miktar)
                    deadline = sla_deadline(slot_datetime(g, s), data.route_lookup[target_hat]["target_delivery_days"])
                    sla_cost = sla_cezasi_tl(miktar, gecikme_saat(tamamlanma, deadline))
                    state.assignments.append(Assignment(target_hat, g, s, miktar, (leg,), sla_cost, 0.0, talep_id_by_gs.get((g, s), "")))
                    toplam_yuk -= miktar

            if s_adet > 0 and toplam_yuk > 0:
                istenen = min(toplam_yuk, s_adet * kap)
                miktar = min(istenen, state.max_addable_on_leg(src, dst, g, s, a, False))
                if miktar > 0:
                    vehicle_cost = state._commit_leg(src, dst, g, s, a, miktar, False)
                    varis_g = arrival_day(data.route_lookup, data.gunler, target_hat, g, s, a) or g
                    leg = Leg(src, dst, g, s, a, False)
                    tamamlanma = _completion_datetime(data, [leg], miktar)
                    deadline = sla_deadline(slot_datetime(g, s), data.route_lookup[target_hat]["target_delivery_days"])
                    sla_cost = sla_cezasi_tl(miktar, gecikme_saat(tamamlanma, deadline))
                    state.assignments.append(Assignment(target_hat, g, s, miktar, (leg,), sla_cost, vehicle_cost, talep_id_by_gs.get((g, s), "")))
                    toplam_yuk -= miktar

            if toplam_yuk > 1e-6:
                # CP-SAT'in onerdigi miktarin bir kismi (baska hatlarin o an
                # kullandigi paylasimli kapasite yuzunden) clamp'lendi. Bu artigi
                # BURADA, hemen yerlestiriyoruz (once diger yol/slot secenekleri,
                # sonra son care force_insert) - "return state" ile birlikte
                # unassigned'da yari-islenmis birakmiyoruz (bkz. objective() guvenlik
                # agi + bu operatorun HER ZAMAN tam teslim garanti etmesi gerekliligi).
                kalan = _insert_chunk(state, target_hat, g, s, toplam_yuk, rng, "")
                if kalan > 1e-6:
                    # force_insert(state, target_hat, g, s, kalan, "")
                    state.unassigned.append((target_hat, g, s, kalan, ""))

    # Bu operator SADECE target_hat'i CP-SAT ile cozdu; destroy birden fazla
    # hattan parca kaldirmis olabilir - digerleri (other_items) hala
    # state.unassigned'da bekliyor olabilir. Bu fonksiyon da (greedy_repair gibi)
    # HER ZAMAN tam teslim garanti etmeli - kalanlari greedy sekilde yerlestiriyoruz.
    kalan_items = list(state.unassigned)
    state.unassigned = []
    for (hat2, gun2, slot2, desi2, talep_id2) in kalan_items:
        kalan2 = _insert_chunk(state, hat2, gun2, slot2, desi2, rng, talep_id2)
        if kalan2 > 1e-6:
            # force_insert(state, hat2, gun2, slot2, kalan2, talep_id2)
            state.unassigned.append((hat2, gun2, slot2, kalan2, talep_id2))

    return state
