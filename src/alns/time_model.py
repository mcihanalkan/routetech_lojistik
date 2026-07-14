"""Faz 2 saat/dispatch zaman modeli.

Zaman ekseni CP-SAT'a serbest tamsayı değişken olarak girmez: kalkış epoğundan
(DISPATCH_SLOTS) başlayarak seyir + elleçleme süreleri analitik olarak toplanır.
Bu tasarım, durum uzayının (her saat için ayrı karar değişkeni yerine) küçük ve
sabit kalmasını sağlayan kilit karardır — bkz. plan dosyası.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pandas as pd

from src.config import (
    ELLECLEME_DAKIKA_PER_DESI,
    SLA_CEZA_TL_PER_DESI_PER_SAAT,
    VEHICLE_DURATION_COLUMNS,
)

# Talep günde yalnızca bu iki sabit saatte oluşur (PDF: "Talep Tamamlanma Saati").
# Kalkış kararları da bu epoklarda alınır.
DISPATCH_SLOTS = [
    "00:00", "04:00", "08:00", "12:00","16:00", "20:00"
]
DEMAND_ARRIVAL_TIMES = [
    "09:00", "17:00"
]

RouteLookup = dict[tuple[str, str], dict]


def slot_to_hour(slot: str) -> int:
    return int(str(slot).split(":")[0])


def build_route_lookup(route_matrix: pd.DataFrame) -> RouteLookup:
    """(source, destination) -> {distance_km, target_delivery_days, <araç türü>: seyir_saat}."""
    lookup: RouteLookup = {}
    for _, row in route_matrix.iterrows():
        entry = {
            "distance_km": float(row["distance_km"]),
            "target_delivery_days": float(row["target_delivery_days"]),
        }
        for vehicle_type, column in VEHICLE_DURATION_COLUMNS.items():
            entry[vehicle_type] = float(row[column])
        lookup[(row["source"], row["destination"])] = entry
    return lookup


def seyir_suresi_saat(route_lookup: RouteLookup, source: str, destination: str, vehicle_type: str) -> float:
    entry = route_lookup.get((source, destination))
    if entry is None:
        raise KeyError(f"Rota bulunamadı: {source} -> {destination}")
    return entry[vehicle_type]


def ellecleme_suresi_dakika(desi: float, consolidation: bool = False) -> float:
    """Elleçleme süresi = desi * 0.01 dk. Konsolidasyonda (indir + tekrar yükle) 2x sayılır."""
    sure = desi * ELLECLEME_DAKIKA_PER_DESI
    return sure * 2 if consolidation else sure




def varis_zamani(kalkis: datetime, seyir_saat: float) -> datetime:
    return kalkis + timedelta(hours=seyir_saat)


def ellecleme_tamamlanma_zamani(varis: datetime, desi: float, consolidation: bool = False) -> datetime:
    """SLA hesabında esas alınan an: araç varışı değil, elleçlemenin tamamlanma anı."""
    return varis + timedelta(minutes=ellecleme_suresi_dakika(desi, consolidation=consolidation))


def sla_deadline(talep_tamamlanma: datetime, hedef_teslim_gun: float) -> datetime:
    """SLA süresi, talebin tamamlanma anından (talep oluşum saati) başlar."""
    return talep_tamamlanma + timedelta(days=hedef_teslim_gun)


def gecikme_saat(fiili_tamamlanma: datetime, deadline: datetime) -> int:
    """Gecikme saat bazında hesaplanır; tam olmayan her gecikme bir üst tam saate yuvarlanır."""
    fark_dakika = (fiili_tamamlanma - deadline).total_seconds() / 60.0
    if fark_dakika <= 0:
        return 0
    return math.ceil(fark_dakika / 60.0)


def sla_cezasi_tl(geciken_desi: float, gecikme_saat_degeri: int) -> float:
    return geciken_desi * gecikme_saat_degeri * SLA_CEZA_TL_PER_DESI_PER_SAAT


def slot_datetime(gun: str, slot: str) -> datetime:
    return datetime.combine(pd.Timestamp(gun).date(), datetime.min.time()) + timedelta(
        hours=slot_to_hour(slot)
    )


def arrival_day(
    route_lookup: RouteLookup,
    valid_days: set[str] | list[str],
    hat: tuple[str, str],
    gun: str,
    slot: str,
    arac_turu: str,
) -> str | None:
    """Kalkış (gun, slot) + seyir süresine göre varışın düştüğü takvim günü (analitik).
    `valid_days` dışına düşerse None (o sefer, modellenen ufkun dışında kalır)."""
    entry = route_lookup.get(hat)
    if entry is None:
        return None
    toplam_saat = slot_to_hour(slot) + entry[arac_turu]
    gun_offset = int(toplam_saat // 24)
    varis_gun = (date.fromisoformat(gun) + timedelta(days=gun_offset)).isoformat()
    return varis_gun if varis_gun in valid_days else None


def next_dispatch_slot(
    valid_days: list[str],
    gun: str,
    slot: str,
    seyir_saat: float,
) -> tuple[str, str] | None:
    """Bir bacağın varışından SONRA, aktarma/konsolidasyon için bir sonraki kalkış
    penceresini (gun, slot) döndürür. Elleçleme süresi slot aralıklarına (8-16 saat)
    kıyasla çok kısa olduğundan (bkz. ellecleme_suresi_dakika), varış anından hemen
    sonraki dispatch penceresi güvenli bir yaklaşım olarak kullanılır — varışla aynı
    slotta asla kalkış yapılmaz (elleçlemeye zaman tanımak için).
    `valid_days` ufkunun dışına düşerse None."""
    toplam_saat = slot_to_hour(slot) + seyir_saat
    gun_offset, saat_of_day = divmod(toplam_saat, 24) 
    for aday_slot in DEMAND_ARRIVAL_TIMES:
        if slot_to_hour(aday_slot) > saat_of_day:
            aday_gun = (date.fromisoformat(gun) + timedelta(days=int(gun_offset))).isoformat()
            return (aday_gun, aday_slot) if aday_gun in valid_days else None
    # Günün tüm slotları geride kaldı -> ertesi günün ilk slotu
    aday_gun = (date.fromisoformat(gun) + timedelta(days=int(gun_offset) + 1)).isoformat()
    return (aday_gun, DEMAND_ARRIVAL_TIMES[0]) if aday_gun in valid_days else None
