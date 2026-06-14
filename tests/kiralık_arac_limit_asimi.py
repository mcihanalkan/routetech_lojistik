import pandas as pd
from pathlib import Path

def kiralik_arac_limit_haritasi_uret(df_kiralik):
    """
    Kiralık araçlar envanter tablosundan hızlı arama sözlüğü (Limit Haritası) üretir.
    Anahtar: (Çıkış, Varış, Araç Türü) -> Değer: Maksimum Araç Sayısı
    """
    limit_haritasi = {}
    
    for _, row in df_kiralik.iterrows():
        cikis = str(row['Çıkış Transfer Merkezi']).strip()
        varis = str(row['Varış Transfer Merkezi']).strip()
        arac_turu = str(row['Araç Türü']).strip()
        arac_sayisi = int(row['Araç sayısı'])
        
        # Üçlü benzersiz anahtarımız oluşturuluyor
        anahtar = (cikis, varis, arac_turu)
        limit_haritasi[anahtar] = arac_sayisi
        
    return limit_haritasi


def emniyet_kilidi_kontrol_gunluk(ahmet_model_ciktisi, limit_haritasi):
    """
    Ahmet'in model çıktısını GÜN GÜN (Tarih bazında) gruplayarak denetler.
    Kiralık araç sınırları her yeni gün için baştan sıfırlanır.
    """
    # 1. AŞAMA: Gruplamaya 'Tarih' sütununu da ekliyoruz.
    # Böylece kod, günleri birbirine karıştırmadan her günü kendi içinde toplar.
    gruplanmis_kararlar = ahmet_model_ciktisi.groupby(
        ['Tarih', 'Çıkış Transfer Merkezi', 'Varış Transfer Merkezi', 'Araç Türü']
    )['Atanan Araç Sayısı'].sum().reset_index()
    
    # 2. AŞAMA: Günlük kararları satır satır denetle
    for _, row in gruplanmis_kararlar.iterrows():
        tarih = str(row['Tarih']).strip()
        cikis = str(row['Çıkış Transfer Merkezi']).strip()
        varis = str(row['Varış Transfer Merkezi']).strip()
        arac_turu = str(row['Araç Türü']).strip()
        atanan_sayi = int(row['Atanan Araç Sayısı'])
        
        anahtar = (cikis, varis, arac_turu)
        
        # 1. KONTROL: Eğer bu hat şirketin kiralık araç listesinde varsa limiti denetle
        if anahtar in limit_haritasi:
            yasal_limit = limit_haritasi[anahtar]
            
            # Günlük limit aşımı kontrolü
            if atanan_sayi > yasal_limit:
                raise ValueError(
                    f"🚨 GÜNLÜK KURAL İHLALİ DETEKTÖRÜ:\n"
                    f"Tarih: {tarih}\n"
                    f"Hat: {cikis} → {varis} | Araç Türü: {arac_turu}\n"
                    f"Şirket Sözleşmesindeki Günlük Maksimum Sınır: {yasal_limit}\n"
                    f"Ahmet'in Modelinin O Gün Atamaya Çalıştığı: {atanan_sayi}\n"
                    f"Kritik ihlal nedeniyle sistem DURDURULDU!"
                )
                
        # 2. KONTROL: Eğer Ahmet envanterde kiralık olarak hiç var olmayan hayali bir araç türettiyse
        else:
            if atanan_sayi > 0:
                raise ValueError(
                    f"🚨 GEÇERSİZ ENVANTER ATAMASI:\n"
                    f"Tarih: {tarih}\n"
                    f"{cikis} → {varis} hattında normalde hiç kiralık '{arac_turu}' bulunmuyor!\n"
                    f"Model hayali envanter kullandı. Program DURDURULDU!"
                )
                
    print("✅ Emniyet Kilidi Raporu: Tüm günlerin günlük kiralık araç limitleri başarıyla doğrulandı. Sınır aşımı yoktur.")
    return True


# ==============================================================================
# SADECE DOĞRUDAN ÇALIŞTIRILDIĞINDA DEVREYE GİREN SİMÜLASYON VE TEST ALTYAPISI
# ==============================================================================
if __name__ == "__main__":
    # Test amaçlı kiralık araçlar envanter tablosunu taklit edelim
    # İstanbul-Yalova arası Tır sınırımız: 2
    envanter_data = {
        'Çıkış Transfer Merkezi': ['İstanbul', 'İstanbul', 'Kocaeli'],
        'Varış Transfer Merkezi': ['Yalova', 'Eskişehir', 'Yalova'],
        'Araç sayısı': [2, 2, 1],
        'Araç Türü': ['Tır', 'Tır', 'Tır']
    }
    df_kiralik = pd.DataFrame(envanter_data)
    yasal_limitler = kiralik_arac_limit_haritasi_uret(df_kiralik)
    
    print("--- 1. SENARYO: Günlük Limitleri Aşmayan Temiz Tablo Test Ediliyor ---")
    # Ahmet'in kurallara uyduğu; 11 Mayıs'ta 2, 12 Mayıs'ta yine 2 tır kaldırdığı temiz çıktı
    temiz_toplu_cikti = pd.DataFrame({
        'Tarih': ['2026-05-11', '2026-05-11', '2026-05-12'],
        'Çıkış Transfer Merkezi': ['İstanbul', 'Kocaeli', 'İstanbul'],
        'Varış Transfer Merkezi': ['Yalova', 'Yalova', 'Yalova'],
        'Araç Türü': ['Tır', 'Tır', 'Tır'],
        'Atanan Araç Sayısı': [2, 1, 2] # 11 Mayıs'ta 2, 12 Mayıs'ta 2. Toplamda 4 ama gün gün bakınca limit içi (<=2)
    })
    
    try:
        emniyet_kilidi_kontrol_gunluk(temiz_toplu_cikti, yasal_limitler)
    except ValueError as e:
        print(f"Hata Oluştu (Beklenmiyordu): {e}")

    print("\n--- 2. SENARYO: 12 Mayıs Günü Limiti Çiğneyen Tablo Test Ediliyor ---")
    # Ahmet'in 11 Mayıs'ta kurallara uyduğu ama 12 Mayıs'ta Yalova'ya 3 tır atadığı hatalı çıktı
    hatali_toplu_cikti = pd.DataFrame({
        'Tarih': ['2026-05-11', '2026-05-11', '2026-05-12'],
        'Çıkış Transfer Merkezi': ['İstanbul', 'Kocaeli', 'İstanbul'],
        'Varış Transfer Merkezi': ['Yalova', 'Yalova', 'Yalova'],
        'Araç Türü': ['Tır', 'Tır', 'Tır'],
        'Atanan Araç Sayısı': [2, 1, 3] # 12 Mayıs günü İstanbul-Yalova sınırı (2) aşılarak 3 yapıldı!
    })
    
    try:
        emniyet_kilidi_kontrol_gunluk(hatali_toplu_cikti, yasal_limitler)
    except ValueError as e:
        print("KOD BAŞARIYLA HATAYI YAKALADI VE SİSTEMİ DURDURDU! Detay:")
        print(e)
        