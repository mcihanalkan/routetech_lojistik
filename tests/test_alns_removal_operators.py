import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alns.alns_engine import (
    ProblemData, State, try_insert_path,
    tm_overload_removal, low_occupancy_removal, shaw_related_removal,
)


def _base_data(gunler, handling_capacity, kapasite_desi=100_000):
    """Ortak, minimal (A->B tek hatli) ProblemData kurulumu."""
    route_lookup = {
        ("A", "B"): {
            "distance_km": 450,
            "Kamyonet": 6.0,
            "target_delivery_days": 1,
        }
    }
    arac_parametreleri = {
        "Kamyonet": {
            "kapasite_desi": kapasite_desi,
            "spot_hourly": 500,
            "spot_km": 20,
            "rental_hourly": 0,
            "rental_km": 0,
        }
    }
    zaman_sirali = [(g, "09:00") for g in gunler]
    return ProblemData(
        route_lookup=route_lookup,
        arac_turleri=["Kamyonet"],
        arac_parametreleri=arac_parametreleri,
        kiralik_stok_gunluk={},
        handling_capacity=handling_capacity,
        tir_capacity={},
        tir_arac_turu=None,
        gunler=gunler,
        merkezler=["A", "B"],
        demands=[],
        zaman_sirali=zaman_sirali,
    )


class TestTmOverloadRemoval(unittest.TestCase):
    def test_only_overloaded_gun_is_targeted(self):
        """
        REGRESYON: tm_overload_removal eskiden bir TM'nin (A) SADECE 1
        gununde %85 esigi asilmis olsa bile, o TM'ye dokunan TUM gunlerdeki
        atamalari sokme adayi yapiyordu - diger gunlerde HICBIR asim
        olmamasina ragmen. Duzeltme sonrasi sadece gercekten asan (tm, gun)
        ciftine ait atamalar hedeflenmeli.

        1 asan gun + 9 asmayan gun kuruyoruz: sadece 1 "mesru" aday varken
        eski kod TUM 10 atamayi aday gosterip bunlardan max(1, 10//3)=3'unu
        sokuyordu - guvercin yuvasi ilkesiyle bu 3'un EN AZ 2'si zorunlu
        olarak asmayan bir gune ait olur (deterministik hata, seed'e bagli
        degil). Duzeltilmis kod ise SADECE 1 adayi (asan gunu) sokmeli.
        """
        gunler = [f"2026-05-{i+1:02d}" for i in range(10)]
        asan_gun = gunler[0]
        asmayan_gunler = gunler[1:]
        data = _base_data(gunler, handling_capacity={"A": 1000, "B": 1_000_000})
        state = State(data)

        for g in gunler:
            a = try_insert_path(state=state, hat=("A", "B"), gun=g, slot="09:00", desi=100.0, path=())
            self.assertIsNotNone(a)
        self.assertEqual(len(state.assignments), 10)

        # sadece asan_gun'de A merkezini FIILEN asima sok (%90); digerleri esigin (%85) ALTINDA (%10)
        state.handling_usage[("A", asan_gun)] = 900.0
        for g in asmayan_gunler:
            state.handling_usage[("A", g)] = 100.0

        rng = np.random.default_rng(7)
        new_state = tm_overload_removal(state, rng)

        kalan_gunler = {leg.gun for a in new_state.assignments for leg in a.legs}
        for g in asmayan_gunler:
            self.assertIn(
                g, kalan_gunler,
                f"HATA: {g} gunundeki atama, o gunde HICBIR kapasite asimi olmamasina ragmen "
                "sokuldu (tm_overload_removal gun bazinda degil, TM bazinda sokuyor)."
            )
        self.assertNotIn(
            asan_gun, kalan_gunler,
            f"HATA: {asan_gun} gunundeki (gercekten asan) atama sokulmedi."
        )
        self.assertEqual(len(new_state.assignments), 9)

        # kargo korunumu: toplam desi (assignments + unassigned) degismemeli
        toplam = sum(a.desi for a in new_state.assignments) + sum(d for (_h, _g, _s, d, _tid) in new_state.unassigned)
        self.assertAlmostEqual(toplam, 1000.0, places=3)

    def test_no_overload_falls_back_to_random_removal(self):
        gunler = ["2026-05-01"]
        data = _base_data(gunler, handling_capacity={"A": 1000, "B": 1000})
        state = State(data)
        a1 = try_insert_path(state=state, hat=("A", "B"), gun=gunler[0], slot="09:00", desi=10.0, path=())
        self.assertIsNotNone(a1)

        rng = np.random.default_rng(3)
        new_state = tm_overload_removal(state, rng)
        # Kapasitenin cok altinda oldugu icin random_removal'a dusmeli ve kargo kaybolmamali
        toplam = sum(a.desi for a in new_state.assignments) + sum(d for (_h, _g, _s, d, _tid) in new_state.unassigned)
        self.assertAlmostEqual(toplam, 10.0, places=3)


class TestUnassignedNoDoubleAppend(unittest.TestCase):
    """
    REGRESYON: low_occupancy_removal ve shaw_related_removal, _remove_assignment()
    zaten sokulen atamayi state.unassigned'a EKLEDIGI halde, AYNI kaydi elle
    bir kez daha ekliyordu. Bu, repair operatorlerinin ayni kargoyu 2 kere
    yerlestirmesine (kargo korunumu ihlaline / desi sisirilmesine) yol aciyordu.
    """

    def test_low_occupancy_removal_does_not_duplicate_unassigned(self):
        gunler = ["2026-05-01"]
        # Kapasitenin sadece %10'u kullanilsin (esik %40'in altinda -> sokme adayi)
        data = _base_data(gunler, handling_capacity={"A": 1_000_000, "B": 1_000_000}, kapasite_desi=1000)
        state = State(data)
        a1 = try_insert_path(state=state, hat=("A", "B"), gun=gunler[0], slot="09:00", desi=100.0, path=())
        self.assertIsNotNone(a1)
        self.assertEqual(len(state.assignments), 1)

        rng = np.random.default_rng(11)
        new_state = low_occupancy_removal(state, rng)

        self.assertEqual(
            len(new_state.assignments), 0,
            "On kosul basarisiz: dusuk doluluklu atama sokulmedi, asil test hic baslayamadi."
        )
        self.assertEqual(
            len(new_state.unassigned), 1,
            f"HATA: sokulen atama unassigned havuzuna 1 kez degil {len(new_state.unassigned)} "
            "kez eklendi (cift sayim / kargo korunumu ihlali)."
        )
        toplam_desi = sum(d for (_h, _g, _s, d, _tid) in new_state.unassigned)
        self.assertAlmostEqual(toplam_desi, 100.0, places=3)

    def test_shaw_related_removal_does_not_duplicate_unassigned(self):
        gunler = ["2026-05-01"]
        data = _base_data(gunler, handling_capacity={"A": 1_000_000, "B": 1_000_000})
        state = State(data)
        a1 = try_insert_path(state=state, hat=("A", "B"), gun=gunler[0], slot="09:00", desi=100.0, path=())
        self.assertIsNotNone(a1)
        self.assertEqual(len(state.assignments), 1)

        rng = np.random.default_rng(5)
        new_state = shaw_related_removal(state, rng)

        removed_count = 1 - len(new_state.assignments)
        self.assertGreaterEqual(removed_count, 0)
        if removed_count == 0:
            # Tek atama varken shaw_related_removal en azindan tohumu (seed) sokmelidir.
            self.fail("On kosul basarisiz: shaw_related_removal hicbir atamayi sokmedi.")

        self.assertEqual(
            len(new_state.unassigned), removed_count,
            f"HATA: {removed_count} atama sokuldu ama unassigned havuzunda "
            f"{len(new_state.unassigned)} kayit var (cift sayim / kargo korunumu ihlali)."
        )
        toplam_desi = sum(d for (_h, _g, _s, d, _tid) in new_state.unassigned)
        self.assertAlmostEqual(toplam_desi, 100.0 * removed_count, places=3)


if __name__ == '__main__':
    unittest.main()
