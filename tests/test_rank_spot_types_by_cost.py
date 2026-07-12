import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path

# Başındaki # işaretini kaldırdık ki Python ana dizini bulabilsin:
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alns.alns_engine import _rank_spot_types_by_cost, ProblemData

class TestSpotVehicleRanking(unittest.TestCase):
    
    def setUp(self):
        """
        Gerçek Excel verileriyle ('Araç_Kapasite_Maliyet_Saat.csv' ve 'sehirler_arasi_lojistik.csv')
        birebir eşleşen test ortamı.
        """
        self.mock_data = MagicMock(spec=ProblemData)
        self.mock_data.arac_turleri = ["Tır", "Kamyon", "Hafif Kamyon", "Kamyonet"]
        
        self.mock_data.arac_parametreleri = {
            "Tır":          {"kapasite_desi": 22400, "spot_hourly": 487.50, "spot_km": 25},
            "Kamyon":       {"kapasite_desi": 12000, "spot_hourly": 318.25, "spot_km": 21},
            "Hafif Kamyon": {"kapasite_desi": 7200,  "spot_hourly": 364.58, "spot_km": 20},
            "Kamyonet":     {"kapasite_desi": 5600,  "spot_hourly": 197.91, "spot_km": 18},
        }
        
        # GERÇEK VERİ: İstanbul -> Yalova rotası (60 km) ve araca özel saat süreleri
        self.mock_data.route_lookup = {
            ("İstanbul", "Yalova"): {
                "distance_km": 60,
                "Tır": 0.92, 
                "Kamyon": 0.86, 
                "Hafif Kamyon": 0.80, 
                "Kamyonet": 0.75
            }
        }

    def test_rank_spot_types_istanbul_yalova_scenarios(self):
        """
        Gerçek İstanbul-Yalova rotası üzerinde farklı desi senaryolarını test eder.
        """
        hat = ("İstanbul", "Yalova")
        
        # Gerçek maliyetlere göre (Mesafe: 60km) Tek Aracın Maliyetleri:
        # Kamyonet: ~1328 TL | Hafif Kamyon: ~1491 TL | Kamyon: ~1533 TL | Tır: ~1948 TL
        
        scenarios = [
            (100,   "Kamyonet",     "Çok küçük yük: Tek Kamyonet en ucuzudur (1328 TL)."),
            (5600,  "Kamyonet",     "Tam sınır yükü: Kamyonet %100 dolar, hala en ucuzudur."),
            (6000,  "Hafif Kamyon", "Kamyonet'i aşan yük: 2 Kamyonet (2656 TL) tutmaktansa 1 Hafif Kamyon (1491 TL) tutmak daha ucuzdur!"),
            (10000, "Kamyon",       "Orta-Büyük yük: 1 Kamyon (1533 TL) tutmak, 2 Kamyonet veya 2 Hafif Kamyon tutmaktan ucuzdur."),
            (15000, "Tır",          "Büyük yük: 1 Tır (1948 TL) tutmak, diğer tüm kombinasyonları ezer geçer.")
        ]
        
        for desi, best_vehicle, desc in scenarios:
            with self.subTest(desi=desi, desc=desc):
                siralama = _rank_spot_types_by_cost(self.mock_data, hat, desi)
                
                self.assertEqual(
                    siralama[0], best_vehicle,
                    f"\n🚨 HATA: {desi} desi için en uygun araç '{best_vehicle}' olmalıydı!\n"
                    f"Senaryo: {desc}\n"
                    f"Sistemin Seçtiği: '{siralama[0]}'\n"
                )

if __name__ == '__main__':
    unittest.main()