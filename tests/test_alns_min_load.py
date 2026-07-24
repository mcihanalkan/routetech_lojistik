import unittest
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# src modüllerine erişim için ana dizini path'e ekliyoruz
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alns.alns_engine import (
    ProblemData, State, try_insert_path, enforce_min_spot_occupancy,
    MIN_SPOT_DOLULUK_ORANI,
)

class TestSpotMinLoad(unittest.TestCase):
    def test_spot_5_percent_load_accepted(self):
        """
        İLK YERLEŞTİRME anında %10 doluluk kuralının bir REDDETME sebebi
        olmadığını test eder — kapasitenin %5'i kadar küçük bir talep, bu
        aşamada geri çevrilmemeli ve ilk denemede araca atanmalıdır.

        ⚠️ ÖNEMLİ SINIR: Bu test SADECE try_insert_path()'in (ilk yerleştirme)
        davranışını kontrol eder. Bu, "%10 doluluk kuralı sistemde tamamen
        yok" anlamına GELMEZ — enforce_min_spot_occupancy() bu YERLEŞTİRMEYİ
        SONRADAN sökup zorla erteleyebilir (bkz. aşağıdaki
        test_spot_5_percent_load_deferred_by_enforcement testi, ki bu ikisi
        BİRLİKTE okunmalı). MIN_SPOT_DOLULUK_ORANI hâlâ 0.10 ve ALNS
        pipeline'ında aktif.
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
        
        print("✅ TEST BAŞARILI: %5 kapasiteli (50 desi) küçük talep, İLK yerleştirme adımında reddedilmedi.")

    def test_spot_5_percent_load_deferred_by_enforcement(self):
        """
        MIN_SPOT_DOLULUK_ORANI (=0.10) kuralının, ilk yerleştirmeden SONRA
        enforce_min_spot_occupancy() aracılığıyla nasıl uygulandığını test
        eder. Bu, bu oturumda main.py'yi uçtan uca defalarca çalıştırarak
        ampirik olarak doğruladığımız asıl mekanizma: %5 doluluklu bir spot
        grup, ilk yerleştirmede kabul edilse bile optimizasyonun SONUNDA
        sökülüp KESİNLİKLE SONRAKİ bir zaman dilimine (gün/slot) zorla
        ertelenmeli (README kuralı: "doluluk oranını karşılamayan yükler o
        gün taşımaya alınmaz, bir sonraki güne ertelenir").

        Taper penceresinden (MIN_SPOT_DOLULUK_TAPER_GUN_SAYISI=2 gün=4 slot,
        bu pencerede eşik kademeli olarak 0'a iner) etkilenmemek için, talebi
        BİLEREK zaman çizelgesinin EN BAŞINA (son 4 slot'un çok gerisine)
        yerleştiriyoruz — aksi halde eşik zaten düşük/sıfır olur ve test
        hiçbir şey kanıtlamaz.
        """
        route_lookup = {
            ("Istanbul", "Ankara"): {
                "distance_km": 450,
                "Kamyonet": 6.0,
                "target_delivery_days": 1
            }
        }
        arac_turleri = ["Kamyonet"]
        arac_parametreleri = {
            "Kamyonet": {
                "kapasite_desi": 1000,
                "spot_hourly": 500,
                "spot_km": 20,
                "rental_hourly": 0,
                "rental_km": 0
            }
        }

        # Taper penceresinden (son 2 gün = 4 slot) rahatça uzakta kalmak için
        # 8 günlük (16 slotluk) bir zaman çizelgesi kuruyoruz; talep İLK
        # slotta (gün 1, 09:00) oluşacak.
        baslangic = datetime(2026, 5, 1)
        gunler = [(baslangic + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(8)]
        zaman_sirali = [(gun, slot) for gun in gunler for slot in ("09:00", "17:00")]

        ilk_gun, ilk_slot = zaman_sirali[0]
        desi_5_percent = 50.0  # Kamyonet kapasitesinin (1000) %5'i — 0.10 eşiğinin altında

        # DİKKAT: bu hat için TOPLAM ufuk talebini de veriyoruz (sadece bu 50
        # desi'lik parça değil) - dinamik alt limit (bkz. _hat_toplam_talep),
        # bir hat'ın TÜM ufukta ulaşabileceği azami dolulukla, kovaladığı
        # eşiği (%10) sınırlıyor. Bu hat için toplam talep (2050 desi),
        # kapasitenin (1000) %10'unu (100 desi) rahatça AŞTIĞI için erteleme
        # burada hâlâ MANTIKLI/beklenen davranış - testin asıl amacı bu.
        data = ProblemData(
            route_lookup=route_lookup,
            arac_turleri=arac_turleri,
            arac_parametreleri=arac_parametreleri,
            kiralik_stok_gunluk={},
            handling_capacity={"Istanbul": 100_000, "Ankara": 100_000},
            tir_capacity={},
            tir_arac_turu=None,
            gunler=gunler,
            merkezler=["Istanbul", "Ankara"],
            demands=[
                (("Istanbul", "Ankara"), ilk_gun, ilk_slot, desi_5_percent, "D_test_1"),
                (("Istanbul", "Ankara"), gunler[3], "09:00", 2000.0, "D_test_2"),
            ],
            zaman_sirali=zaman_sirali,
        )
        state = State(data)
        # data.demands'teki 2. (dolgu) talep, State.__init__ tarafından
        # otomatik olarak state.unassigned'a da eklendi - ama bu test SADECE
        # asagida try_insert_path ile yerlestirilecek 50 desi'lik parcayi
        # takip ediyor, o yuzden unassigned'i temizleyip sadece _hat_toplam_talep
        # icin gerekli olan data.demands bilgisini koruyoruz.
        state.unassigned = []

        assignment = try_insert_path(
            state=state,
            hat=("Istanbul", "Ankara"),
            gun=ilk_gun,
            slot=ilk_slot,
            desi=desi_5_percent,
            path=(),
        )
        self.assertIsNotNone(assignment, "Ön koşul başarısız: ilk yerleştirme reddedildi, asıl test hiç başlayamadı.")
        ilk_bacak = assignment.legs[0]
        self.assertEqual(
            (ilk_bacak.gun, ilk_bacak.slot), (ilk_gun, ilk_slot),
            "Ön koşul başarısız: talep beklenen ilk slotta yerleşmedi."
        )

        # enforce_min_spot_occupancy() SONRASI durumu kontrol et
        rng = np.random.default_rng(42)
        enforce_min_spot_occupancy(state, rng)

        kalan_ilk_slotta_mi = any(
            (leg.gun, leg.slot) == (ilk_gun, ilk_slot)
            for a in state.assignments
            for leg in a.legs
            if a.talep_id == assignment.talep_id or a is assignment
        )
        self.assertFalse(
            kalan_ilk_slotta_mi,
            f"HATA: %5 doluluklu ({desi_5_percent}/1000 desi = %{100*MIN_SPOT_DOLULUK_ORANI:.0f} eşiğinin altında) "
            f"yük, enforce_min_spot_occupancy() SONRASI hâlâ ilk slotunda ({ilk_gun} {ilk_slot}) duruyor — "
            f"kuralın zorla erteleme kısmı beklendiği gibi çalışmıyor."
        )

        toplam_desi_hala_sistemde = sum(a.desi for a in state.assignments) + sum(
            d for (_h, _g, _s, d, _tid) in state.unassigned
        )
        self.assertAlmostEqual(
            toplam_desi_hala_sistemde, desi_5_percent, places=3,
            msg="HATA: Erteleme sırasında desi miktarı kayboldu/değişti — kargo korunumu ihlali."
        )

        print(
            f"✅ TEST BAŞARILI: %5 doluluklu yük ilk slotundan ({ilk_gun} {ilk_slot}) söküldü/ertelendi "
            f"— MIN_SPOT_DOLULUK_ORANI={MIN_SPOT_DOLULUK_ORANI} zorla erteleme mekanizması çalışıyor."
        )

    def test_micro_hat_below_ceiling_not_endlessly_deferred(self):
        """
        DİNAMİK ALT LİMİT: bir hat'ın TÜM ufuktaki toplam talebi, kapasitenin
        %10'unu (burada 100/1000) hiçbir zaman aşamıyorsa (bu testte SADECE
        50 desi - başka hiç talep yok), %10 eşiğini kovalamak imkansız bir
        hedef kovalamak demektir - kural bu durumda esneyip hat'ın kendi
        ulaşabileceği tavana (50/1000 = %5) inmeli, mikro yükü ufkun sonuna
        kadar erteleyip gereksiz SLA cezası biriktirmemeli (bkz. sohbet
        geçmişi: round() düzeltmesi sonrası mikro taleplerin bu şekilde
        ertelenip SLA maliyetini şişirdiği gözlemlendi).

        Bu, test_spot_5_percent_load_deferred_by_enforcement testinin TAM
        TERSİ bir senaryo: oradaki hat için toplam talep eşiği rahatça
        aşıyordu (erteleme mantıklıydı), burada ise hat'ın kendisi hiçbir
        zaman eşiğe ulaşamıyor (erteleme mantıksız/imkansız bir hedef).
        """
        route_lookup = {
            ("Istanbul", "Ankara"): {
                "distance_km": 450,
                "Kamyonet": 6.0,
                "target_delivery_days": 1
            }
        }
        arac_turleri = ["Kamyonet"]
        arac_parametreleri = {
            "Kamyonet": {
                "kapasite_desi": 1000,
                "spot_hourly": 500,
                "spot_km": 20,
                "rental_hourly": 0,
                "rental_km": 0
            }
        }

        baslangic = datetime(2026, 5, 1)
        gunler = [(baslangic + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(8)]
        zaman_sirali = [(gun, slot) for gun in gunler for slot in ("09:00", "17:00")]

        ilk_gun, ilk_slot = zaman_sirali[0]
        desi_5_percent = 50.0

        # DİKKAT: data.demands'te bu hat için SADECE bu 50 desi var - bu hat
        # icin TUM ufukta baska HICBIR talep yok, yani %10 (100 desi) esigine
        # hicbir zaman ulasilamaz.
        data = ProblemData(
            route_lookup=route_lookup,
            arac_turleri=arac_turleri,
            arac_parametreleri=arac_parametreleri,
            kiralik_stok_gunluk={},
            handling_capacity={"Istanbul": 100_000, "Ankara": 100_000},
            tir_capacity={},
            tir_arac_turu=None,
            gunler=gunler,
            merkezler=["Istanbul", "Ankara"],
            demands=[(("Istanbul", "Ankara"), ilk_gun, ilk_slot, desi_5_percent, "D_test_micro")],
            zaman_sirali=zaman_sirali,
        )
        state = State(data)
        state.unassigned = []

        assignment = try_insert_path(
            state=state,
            hat=("Istanbul", "Ankara"),
            gun=ilk_gun,
            slot=ilk_slot,
            desi=desi_5_percent,
            path=(),
        )
        self.assertIsNotNone(assignment, "Ön koşul başarısız: ilk yerleştirme reddedildi, asıl test hiç başlayamadı.")

        rng = np.random.default_rng(42)
        enforce_min_spot_occupancy(state, rng)

        kalan_ilk_slotta_mi = any(
            (leg.gun, leg.slot) == (ilk_gun, ilk_slot)
            for a in state.assignments
            for leg in a.legs
            if a.talep_id == assignment.talep_id or a is assignment
        )
        self.assertTrue(
            kalan_ilk_slotta_mi,
            "HATA: bu hat için TÜM ufukta bu 50 desi'den başka talep yokken "
            "(yani %10 eşiğine hiçbir zaman ulaşılamaz), yük hâlâ ertelendi - "
            "dinamik alt limit (ust_sinir) beklendiği gibi çalışmıyor."
        )

        print(
            "✅ TEST BAŞARILI: hat'ın ulaşabileceği tavanın (%5) altında kalan "
            "%10 eşiği dinamik olarak gevşetildi, mikro yük gereksiz yere ertelenmedi."
        )



if __name__ == '__main__':
    unittest.main()