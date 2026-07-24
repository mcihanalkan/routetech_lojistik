import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Python'a bir üst klasöre (ana projeye) bakmasını söylüyoruz:
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alns.alns_engine import _rank_spot_types_by_cost, ProblemData

class TestSpotVehicleRanking(unittest.TestCase):
    
    def setUp(self):
        """
        Ahmet'in alns_engine.py dosyasındaki canlı koda tam uyumlu test ortamı.
        """
        self.mock_data = MagicMock(spec=ProblemData)
        self.mock_data.arac_turleri = ["Tır", "Kamyon", "Hafif Kamyon", "Kamyonet"]
        
        self.mock_data.arac_parametreleri = {
            "Tır":          {"kapasite_desi": 22400, "spot_hourly": 487.50, "spot_km": 25},
            "Kamyon":       {"kapasite_desi": 12000, "spot_hourly": 318.25, "spot_km": 21},
            "Hafif Kamyon": {"kapasite_desi": 7200,  "spot_hourly": 364.58, "spot_km": 20},
            "Kamyonet":     {"kapasite_desi": 5600,  "spot_hourly": 197.91, "spot_km": 18},
        }
        
        # GERÇEK VERİ: sehirler_arasi_lojistik.xlsx -> İstanbul-Yalova hattı (60 km)
        self.mock_data.route_lookup = {
            ("İstanbul", "Yalova"): {
                "distance_km": 60,
                "Tır": 0.92, 
                "Kamyon": 0.86, 
                "Hafif Kamyon": 0.80, 
                "Kamyonet": 0.75
            }
        }

    @patch('src.alns.alns_engine.spot_vehicle_count')
    @patch('src.alns.alns_engine.vehicle_leg_cost')
    def test_rank_spot_types_istanbul_yalova_scenarios(self, mock_vehicle_leg_cost, mock_spot_vehicle_count):
        """
        Canlı koddaki (3 parametreli spot_vehicle_count ve 5 parametreli vehicle_leg_cost)
        çağrılarını taklit eden hatasız senaryo testi.
        """
        hat = ("İstanbul", "Yalova")
        
        # AHMET'İN GERÇEK ÇAĞRISINA UYUM: (desi, kapasite, max_allowable) alıyor
        def sahte_arac_adedi(desi, kapasite, *args, **kwargs):
            import math
            return math.ceil(desi / kapasite)
            
        # AHMET'İN GERÇEK ÇAĞRISINA UYUM: (route_lookup, hat, arac_turu, hourly, km) alıyor
        def sahte_bacak_maliyeti(route_lookup, hat_tuple, arac_turu, hourly, km, *args, **kwargs):
            route = route_lookup[hat_tuple]
            km_cost = route["distance_km"] * km
            hourly_cost = route[arac_turu] * hourly
            return hourly_cost + km_cost

        # Sahte dublör fonksiyonlarımızı mock nesnelerine bağlıyoruz:
        mock_spot_vehicle_count.side_effect = sahte_arac_adedi
        mock_vehicle_leg_cost.side_effect = sahte_bacak_maliyeti
        
        # Senaryolarımız (Desi, Beklenen En Ekonomik Araç, Açıklama)
        #
        # ⚠️ DÜZELTME (6000 desi): Bu test _rank_spot_types_by_cost()'u çağırıyor,
        # ve o fonksiyon vehicle_leg_cost (yukarıda mock'landı) dışında AYRICA
        # gerçek (mock'lanmamış) ellecleme_maliyet_hesapla(desi, spot_hourly)
        # çağrısı da yapıyor (bkz. src/alns/cost_model.py — elleçleme süresi =
        # desi × 0.01 dk, dakika yukarı yuvarlanır, sure_saat × spot_hourly).
        # Bu terim önceden gözden kaçırılmış: 6000 desi'de "1 Hafif Kamyon
        # (1491 TL)" hesabı SADECE seyir+km maliyetini içeriyordu, elleçleme
        # maliyetini (Hafif Kamyon için ~365 TL, Kamyon için ~318 TL) atlıyordu.
        # Elleçleme dahil edilince gerçek toplamlar: Kamyon≈1852 TL, Hafif
        # Kamyon≈1857 TL — aralarında sadece ~5 TL fark var ve Kamyon kazanıyor.
        # Diğer 4 senaryo elle (kalem-kağıt, elleçleme dahil) yeniden doğrulandı,
        # hepsi hâlâ orijinal beklentiyle uyuşuyor — sadece bu senaryo değişti.
        scenarios = [
            (100,   "Kamyonet",     "Çok küçük yük: Tek Kamyonet en ucuzudur (~1231 TL, elleçleme dahil)."),
            (5600,  "Kamyonet",     "Tam sınır yükü: Kamyonet %100 dolar, elleçleme dahil hâlâ en ucuzudur (~1413 TL)."),
            (6000,  "Kamyon",       "Kamyonet'i aşan yük: elleçleme maliyeti dahil, 1 Kamyon (~1852 TL) 1 Hafif Kamyon'dan (~1857 TL) az farkla ucuzdur; 2 Kamyonet (~2655 TL) zaten elenir."),
            (10000, "Kamyon",       "Orta-Büyük yük: 1 Kamyon (~2064 TL, elleçleme dahil) tutmak, 2 Kamyonet/Hafif Kamyon'dan ucuzdur."),
            (15000, "Tır",          "Büyük yük: 1 Tır (~3168 TL, elleçleme dahil) tutmak, diğer tüm kombinasyonları eler geçer.")
        ]
        
        for desi, best_vehicle, desc in scenarios:
            with self.subTest(desi=desi, desc=desc):
                # Ahmet'in canlı fonksiyonunu çağırıyoruz:
                siralama = _rank_spot_types_by_cost(self.mock_data, hat, desi)
                
                self.assertEqual(
                    siralama[0], best_vehicle,
                    f"\n🚨 HATA: {desi} desi için en uygun araç '{best_vehicle}' olmalıydı!\n"
                    f"Senaryo: {desc}\n"
                    f"Sistemin Seçtiği: '{siralama[0]}'\n"
                )

if __name__ == '__main__':
    unittest.main()