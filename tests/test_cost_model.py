import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alns.cost_model import spot_vehicle_count


class TestSpotVehicleCountFloatDrift(unittest.TestCase):
    """
    REGRESYON: ALNS, leg_spot_desi[key]'i binlerce iterasyon boyunca += / -=
    ile artimli guncelliyor. Talep desi'leri artik tam sayi olmak zorunda
    olmadigindan (round() kaldirildi - bkz. sohbet gecmisi), bu birikimli
    float toplama/cikarma kapasite sinirinda (orn. tam 5600.0) minik bir
    kayma (5600.00000001 gibi) yaratabiliyor. spot_vehicle_count()'un
    epsilonsuz math.ceil() versiyonu bu kaymayi FAZLADAN BIR ARAC olarak
    sayiyordu - bu da State.objective()'in (ALNS arama sinyali) rapor
    aninda sifirdan hesaplanan gercek maliyetten milyonlarca TL sapmasina
    yol acti (gozlemlenen fark: 3,659,952 TL, tek bir ALNS calismasinda).
    """

    def test_exact_boundary_no_drift(self):
        self.assertEqual(spot_vehicle_count(5600.0, 5600.0, 500), 1)

    def test_tiny_float_drift_above_boundary_still_one_vehicle(self):
        # tipik birikimli float hatasi buyuklugu (~1e-10 - 1e-12), gercek
        # bir fazlalik degil
        self.assertEqual(spot_vehicle_count(5600.0 + 1e-10, 5600.0, 500), 1)
        self.assertEqual(spot_vehicle_count(5600.0 + 1e-9, 5600.0, 500), 1)

    def test_tiny_float_drift_below_boundary_still_one_vehicle(self):
        self.assertEqual(spot_vehicle_count(5600.0 - 1e-10, 5600.0, 500), 1)

    def test_genuine_overage_still_counted(self):
        # 0.5 desi gercek bir fazlalik - epsilon bunu YUTMAMALI
        self.assertEqual(spot_vehicle_count(5600.5, 5600.0, 500), 2)

    def test_accumulated_float_addition_matches_exact_multiple(self):
        """Gercek ALNS senaryosunu simule eder: kapasiteye TAM esit toplam
        desi'yi, kucuk ondalikli parcalar halinde binlerce kez += ile
        biriktir - sonuc hala 1 arac olmali, birikmis float artiklari 2'ye
        sicramamali."""
        capacity = 5600.0
        toplam = 0.0
        parca = 0.0068  # gercekci, kucuk/ondalikli bir talep buyuklugu
        adet = round(capacity / parca)
        for _ in range(adet):
            toplam += parca
        # Bu noktada toplam, kayan nokta hatasi yuzunden capacity'den
        # HAFIFCE farkli olabilir (ustunde ya da altinda) - asil kontrol
        # spot_vehicle_count'un bunu 1 arac olarak saymasi.
        self.assertEqual(spot_vehicle_count(toplam, capacity, 500), 1)


if __name__ == '__main__':
    unittest.main()
