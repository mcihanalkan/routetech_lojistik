# BU DOSYAYA, ProblemData CLASS'ININ INIT HESAPLAMALARINI YAPARKEN KULLANDIĞI FONKSİYONLAR BULUNUR.

from src.alns.domain import(
    RouteLookup,
    ProblemData
)
from src.alns.time_model import(
    arrival_day,
    KIRALIK_DISPATCH_SLOT
)

from src.alns.limits import (
    MAX_2HOP_CANDIDATES,
    MAX_RELAY_CANDIDATES,
    MAX_SPOT
)
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
