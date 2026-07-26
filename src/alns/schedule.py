from src.alns.cost_model import spot_vehicle_count, vehicle_leg_cost, ellecleme_maliyet_hesapla
from src.alns.domain import Leg, ProblemData
from src.alns.time_model import (
    ellecleme_tamamlanma_zamani,
    slot_datetime,
    varis_zamani,
)


def _pick_vehicle_type(data: ProblemData, desi: float) -> list:
    """Araç türlerini büyükten küçüğe kapasiteye göre sırala."""
    return sorted(data.arac_turleri, key=lambda a: -data.arac_parametreleri[a]["kapasite_desi"])


def _rank_spot_types_by_cost(data: ProblemData, hat: tuple, desi: float) -> list:
    """Spot araç türlerini tahmini toplam maliyete göre artan sırada döndürür."""

    def tahmini_maliyet(arac_turu: str) -> float:
        p = data.arac_parametreleri[arac_turu]
        kap = p["kapasite_desi"]
        adet = spot_vehicle_count(desi, kap, 10 ** 9) if desi > 0 else 0
        birim_maliyet = vehicle_leg_cost(data.route_lookup, hat, arac_turu, p["spot_hourly"], p["spot_km"])
        return (adet * birim_maliyet) + ellecleme_maliyet_hesapla(desi, p["spot_hourly"])

    return sorted(data.arac_turleri, key=tahmini_maliyet)


def leg_zaman_cizelgesi(data: ProblemData, legs: list, desi: float) -> list:
    """Her bacağın gerçek kalkış ve varış anını sırayla döndürür."""
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