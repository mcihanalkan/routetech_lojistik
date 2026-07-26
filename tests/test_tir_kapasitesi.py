import pandas as pd
import os

def test_tir_kapasitesi_asimi():
    print("⏳ Gelişmiş Çözüm (Dataset B) - Tır Kapasitesi Testi Başlatılıyor...")
    
    # 1. Dosya Yolları (Root dizinine göre tanımlı)
    kapasite_file_path = os.path.join("data", "raw", "tir_kapasiteleri v2.xlsx")
    results_file_path = os.path.join("results", "optimization_results.xlsx")

    # Dosya varlık kontrolleri
    if not os.path.exists(kapasite_file_path):
        raise FileNotFoundError(f"❌ Kapasite dosyası bulunamadı: {kapasite_file_path}")
    if not os.path.exists(results_file_path):
        raise FileNotFoundError(f"❌ Sonuç dosyası bulunamadı: {results_file_path}")

    # 2. Verileri Oku
    print("📂 Excel dosyaları okunuyor...")
    df_kapasite = pd.read_excel(kapasite_file_path, sheet_name="table.tsv")
    df_plan = pd.read_excel(results_file_path, sheet_name="Teslim Plani (Talep Bazli)")

    # TM bazlı tır kapasite sözlüğü { 'İstanbul': 10, 'Yalova': 4 ... }
    kapasite_dict = dict(zip(df_kapasite['transfer_merkezi'].astype(str).str.strip(), df_kapasite['tir_kapasitesi']))

    # 3. Sadece "Tır" Araç Türünü Filtrele
    df_tir = df_plan[df_plan['Arac Turu'].astype(str).str.contains('Tir|Tır', case=False, na=False)].copy()

    if df_tir.empty:
        print("✅ TEST BAŞARILI: Planda hiç Tır kullanılmamış (Sadece Kamyon/Kamyonet var).")
        return

    toplam_arac_sayisi = df_tir['Arac ID'].nunique()
    print(f"🚛 Toplam {toplam_arac_sayisi} farklı Tır filosu / aracı tespit edildi. Rotalar analiz ediliyor...")

    gunluk_kullanim = {}

    # 4. Kronolojik Rota ve Kapasite Tüketim Takibi
    for arac_id, group in df_tir.groupby('Arac ID'):
        
        # Mükerrer kargo satırlarını eleyerek fiziksel bacakları teke düşür ve zaman sırasına diz
        bacaklar = group[['Tarih', 'Slot', 'Cikis TM', 'Varis Tarihi', 'Varis TM', 'Bacaktaki Arac Sayisi']].drop_duplicates().sort_values(by=['Tarih', 'Slot'])
        
        events = []
        
        for _, row in bacaklar.iterrows():
            cikis_tm = str(row['Cikis TM']).strip()
            varis_tm = str(row['Varis TM']).strip()
            cikis_gun = str(row['Tarih'])
            varis_gun = str(row['Varis Tarihi'])
            arac_sayisi = int(row['Bacaktaki Arac Sayisi'])
            
            if not events:
                # İlk kalkış ve varış olayını ekle
                events.append(((cikis_gun, cikis_tm), arac_sayisi))
                events.append(((varis_gun, varis_tm), arac_sayisi))
            else:
                (last_gun, last_tm), last_arac_sayisi = events[-1]
                
                # MILK RUN KONTROLÜ: Tarih değişse bile araç aynı TM'de kalıp hareket etmediyse (1 birim kuralı)
                if last_tm == cikis_tm:
                    # Konvoya ek araç eklendiyse (örn: 1 geldi, 2 çıkıyor), aradaki fark kadar ek kapasite düş
                    ekstra_arac = max(0, arac_sayisi - last_arac_sayisi)
                    if ekstra_arac > 0:
                        events.append(((cikis_gun, cikis_tm), ekstra_arac))
                        
                    # Varış noktasını ekle
                    events.append(((varis_gun, varis_tm), arac_sayisi))
                else:
                    # FARKLI SEFER / DÖNÜŞ KONTROLÜ: Araç başka TM'den geliyorsa ayrı bir birim daha tüketir
                    events.append(((cikis_gun, cikis_tm), arac_sayisi))
                    events.append(((varis_gun, varis_tm), arac_sayisi))

        # Olayları günlük TM kullanım havuzuna yansıt
        for (gun, tm), sayi in events:
            if gun not in gunluk_kullanim:
                gunluk_kullanim[gun] = {}
            gunluk_kullanim[gun][tm] = gunluk_kullanim[gun].get(tm, 0) + sayi

    # 5. Kısıt (Constraint) İhlal Kontrolü
    hatalar = []
    for tarih in sorted(gunluk_kullanim.keys()):
        for tm, kullanim in gunluk_kullanim[tarih].items():
            sinir = kapasite_dict.get(tm, 0)
            if kullanim > sinir:
                hatalar.append(f"Tarih: {tarih} | TM: {tm} | Limit: {sinir} | Kullanılan: {kullanim}")

    # 6. Raporlama
    if hatalar:
        print(f"\n🚨 KRİTİK HATA: Toplam {len(hatalar)} durumda Günlük Tır Kapasite Sınırları Aşıldı!")
        print("-" * 65)
        for hata in hatalar[:15]:
            print(f"  ❌ {hata}")
        if len(hatalar) > 15:
            print(f"  ... ve {len(hatalar) - 15} ihlal durumu daha gizlendi.")
        print("-" * 65)
        
        raise AssertionError("Algoritmanın ürettiği sonuçlar günlük Tır kapasitesi kısıtını ihlal ediyor!")
    else:
        print("\n✅ TEST BAŞARILI: Konsolidasyon, Milk Run ve çoklu gün rotaları dahil hiçbir TM'de günlük Tır kapasite sınırı aşılmadı.")

if __name__ == "__main__":
    test_tir_kapasitesi_asimi()