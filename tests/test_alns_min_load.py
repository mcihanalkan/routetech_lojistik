import unittest
import sys
from pathlib import Path

# src modüllerine erişim için ana dizini path'e ekliyoruz
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alns.alns_engine import ProblemData, State, try_insert_path

class TestSpotMinLoad(unittest.TestCase):
    def test_spot_5_percent_load_accepted(self):
        """
        Gelişmiş Çözüm Aşamasında %10 doluluk kuralının kalktığını test eder.
        Kapasitenin %5'i kadar küçük bir talep reddedilmemeli ve ilk denemede atanmalıdır.
        """
        # 1. Mock (Sahte) Verileri Hazırla
        route_lookup = {
            ("Istanbul", "Ankara"): {
                "distance_km": 450,
                "Kamyonet": 6.0,  # 6 saat seyir süresi
                "target_delivery_days": 1
            }
        }
        
        arac_turleri = ["Kamyonet"]
        arac_parametreleri = {
            "Kamyonet": {
                "kapasite_desi": 1000, # Kamyonet kapasitesi 1000 desi
                "spot_hourly": 500,
                "spot_km": 20,
                "rental_hourly": 0,
                "rental_km": 0
            }
        }
        
        # Pandas'ın parse edebileceği standart tarih formatı (YYYY-MM-DD)
        test_tarihi = "2026-05-01" 
        
        # Problem Data'yı oluştur (Sadece test için gerekli kısımlar)
        data = ProblemData(
            route_lookup=route_lookup,
            arac_turleri=arac_turleri,
            arac_parametreleri=arac_parametreleri,
            kiralik_stok_gunluk={},
            handling_capacity={"Istanbul": 10000, "Ankara": 10000},
            tir_capacity={},
            tir_arac_turu=None,
            gunler=[test_tarihi],
            merkezler=["Istanbul", "Ankara"],
            demands=[],
            zaman_sirali=[(test_tarihi, "09:00")]
        )
        
        state = State(data)
        
        # %5 kapasite = 50 desi (Kamyonet kapasitesi 1000)
        desi_5_percent = 50.0 
        
        # 2. ALNS Motoru try_insert_path Fonksiyonunu Çağır
        assignment = try_insert_path(
            state=state,
            hat=("Istanbul", "Ankara"),
            gun=test_tarihi,
            slot="09:00",
            desi=desi_5_percent,
            path=() # Direkt rota (boş tuple)
        )
        
        # 3. Doğrulama (Assert)
        self.assertIsNotNone(
            assignment, 
            "HATA: %5'lik talep reddedildi! (Eski MVP %10 kuralı hala aktif olabilir veya kapasite hesabında sorun var)"
        )
        self.assertEqual(
            assignment.desi, 
            desi_5_percent, 
            "HATA: Talebin tamamı (50 desi) tek seferde araca yerleşmeliydi."
        )
        
        print("✅ TEST BAŞARILI: %5 kapasiteli (50 desi) küçük talep başarıyla spot araca atandı. MVP %10 kuralı sistemde yok!")

if __name__ == '__main__':
    unittest.main()