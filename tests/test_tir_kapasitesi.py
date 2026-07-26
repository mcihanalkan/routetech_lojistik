import pandas as pd
import os

def test_tir_kapasitesi_asimi():
    print("⏳ Gelişmiş Çözüm - Tır Kapasitesi Testi Başlatılıyor...")
    
    # 1. Dosya Yolları
    kapasite_file_path = os.path.join("data", "raw", "tir_kapasiteleri v2.xlsx")
    results_file_path = os.path.join("results", "optimization_results.xlsx")

    if not os.path.exists(kapasite_file_path):
        raise FileNotFoundError(f"❌ Kapasite dosyası bulunamadı: {kapasite_file_path}")
    if not os.path.exists(results_file_path):
        raise FileNotFoundError(f"❌ Sonuç dosyası bulunamadı: {results_file_path}")

    # 2. Verileri Oku
    print("📂 Excel dosyaları okunuyor...")
    df_kapasite = pd.read_excel(kapasite_file_path, sheet_name="table.tsv")
    df_plan = pd.read_excel(results_file_path, sheet_name="Teslim Plani (Talep Bazli)")

    kapasite_dict = dict(zip(df_kapasite['transfer_merkezi'].str.strip(), df_kapasite['tir_kapasitesi']))

    # 3. Tır Filtresi
    df_tir = df_plan[df_plan['Arac Turu'].str.contains('Tir|Tır', case=False, na=False)].copy()

    if df_tir.empty:
        print("✅ TEST BAŞARILI: Planda hiç Tır kullanılmamış.")
        return

    print(f"🚛 Toplam {df_tir['Arac ID'].nunique()} farklı Tır filosu / aracı tespit edildi. Uğramalı rotalar analiz ediliyor...")

    gunluk_kullanim = {}

    # 4. Gelişmiş Çözüm - Kronolojik Rota Takibi
    # Sadece 'Arac ID'ye göre grupluyoruz çünkü aynı araç 2-3 güne yayılan bir konsolidasyon turu atabilir.
    for arac_id, group in df_tir.groupby('Arac ID'):
        
        # Aynı araca yüklenen kargoları teke düşürüp bacakları (legs) kronolojik diziyoruz.
        # DİKKAT: Artık Varis Tarihi ve Bacaktaki Arac Sayisi'ni de takip ediyoruz!
        bacaklar = group[['Tarih', 'Slot', 'Cikis TM', 'Varis Tarihi', 'Varis TM', 'Bacaktaki Arac Sayisi']].drop_duplicates().sort_values(by=['Tarih', 'Slot'])
        
        events = []
        
        for _, row in bacaklar.iterrows():
            arac_sayisi = row['Bacaktaki Arac Sayisi']
            
            cikis_event = (row['Tarih'], row['Cikis TM'].strip())
            varis_event = (row['Varis Tarihi'], row['Varis TM'].strip())
            
            if not events:
                # İlk bacak: Çıkış ve Varış noktalarını (tarihleriyle birlikte) ekle
                events.append((cikis_event, arac_sayisi))
                events.append((varis_event, arac_sayisi))
            else:
                last_event, _ = events[-1]
                # KURAL KONTROLÜ: (Tarih ve TM birebir aynıysa) -> Araç hareket etmeden yeni yük alıyor demektir, kapasite yakmaz!
                if last_event == cikis_event:
                    events.append((varis_event, arac_sayisi))
                else:
                    # KURAL KONTROLÜ: Geri dönme durumu VEYA bir sonraki güne devredip sabah tekrar yük alma durumu
                    events.append((cikis_event, arac_sayisi))
                    events.append((varis_event, arac_sayisi))

        # 5. Toplanan olayları (events) gerçek tüketim havuzuna ekle
        for (gun, tm), sayi in events:
            if gun not in gunluk_kullanim:
                gunluk_kullanim[gun] = {}
            gunluk_kullanim[gun][tm] = gunluk_kullanim[gun].get(tm, 0) + sayi

    # 6. Kısıt (Constraint) Kontrolü
    hatalar = []
    for tarih, tm_kullanimlari in gunluk_kullanim.items():
        for tm, kullanim in tm_kullanimlari.items():
            sinir = kapasite_dict.get(tm, 0)
            if kullanim > sinir:
                hatalar.append(f"Tarih: {tarih} | TM: {tm} | Limit: {sinir} | Kullanılan: {kullanim}")

    # 7. Sonuç Raporlama
    if hatalar:
        print(f"🚨 KRİTİK HATA: {len(hatalar)} durumda Konsolidasyon yüzünden Tır kapasitesi aşıldı!")
        for hata in hatalar[:15]:
            print(f"  ❌ {hata}")
        if len(hatalar) > 15:
            print(f"  ... ve {len(hatalar) - 15} ihlal durumu daha gizlendi.")
        
        raise AssertionError("Algoritmanın konsolidasyon/uğrama kararları günlük tır limitlerini ihlal ediyor!")
    else:
        print("✅ TEST BAŞARILI: Konsolidasyon (Aktarma) ve çoklu gün rotaları dahil, hiçbir TM'de kapasite aşılmadı.")

if __name__ == "__main__":
    test_tir_kapasitesi_asimi()