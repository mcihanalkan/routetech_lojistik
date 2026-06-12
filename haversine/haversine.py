from pathlib import Path
import numpy as np
import pandas as pd

def calculate_haversine_matrix(latitudes, longitudes):
    """
    Numpy Broadcasting (Vektörizasyon) kullanarak sıfır döngü (for loop) ile
    tüm noktaların birbirine olan Haversine mesafelerini hesaplar.
    
    Parametreler:
    latitudes: Enlem serisi/dizisi (Array-like)
    longitudes: Boylam serisi/dizisi (Array-like)
    
    Döndürdüğü değer:
    (N x N) boyutunda mesafe matrisi (km cinsinden)
    """
    R = 6371.0  # Dünya yarıçapı (km)
    
    # Giriş dizilerini numpy array formatına getirip radyana dönüştürüyoruz
    lat = np.radians(np.array(latitudes))
    lon = np.radians(np.array(longitudes))
    
    # Numpy Broadcasting için 1D dizileri (N, 1) kolon ve (1, N) satır matrislerine dönüştürüyoruz
    # Bu sayede numpy arka planda tüm kombinasyonları döngüsüz eşleştirir.
    lat_col = lat[:, np.newaxis]
    lon_col = lon[:, np.newaxis]
    
    lat_row = lat[np.newaxis, :]
    lon_row = lon[np.newaxis, :]
    
    # Fark matrisleri
    dlat = lat_row - lat_col
    dlon = lon_row - lon_col
    
    # Haversine Formülü (Tüm seriye aynı anda uygulanır)
    a = np.sin(dlat / 2)**2 + np.cos(lat_col) * np.cos(lat_row) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance_matrix = R * c
    print(distance_matrix)
    return distance_matrix

# ==============================================================================
# VERİ YÜKLEME VE STRATEJİK DOĞRULAMA (ASSERTION) AŞAMASI
# ==============================================================================

# 1. DOSYA YOLU DÜZENLEMESİ (haversine/ klasöründen data/raw/ klasörüne geçiş)
current_dir = Path(__file__).resolve().parent
dosya_yolu = current_dir.parent / "data" / "raw" / "Koordinatlar v2.xlsx"

# Veriyi oku
df_koordinat = pd.read_excel(dosya_yolu, sheet_name="Sheet1")


# Mesafeleri hesapla
mesafe_matrisi = calculate_haversine_matrix(df_koordinat['Enlem'], df_koordinat['Boylam'])

# Test için İstanbul ve Yalova'nın indekslerini dinamik olarak bulalım
ist_idx = df_koordinat[df_koordinat['Transfer Merkezi'] == 'İstanbul'].index[0]
yal_idx = df_koordinat[df_koordinat['Transfer Merkezi'] == 'Yalova'].index[0]

# Matrisimizden bu iki şehrin mesafesini çekelim
hesaplanan_mesafe = mesafe_matrisi[ist_idx, yal_idx]

# BİLİNEN GERÇEK: İstanbul-Yalova arası kuş uçuşu mesafe yaklaşık 46.63 km'dir.
beklenen_mesafe = 46.635


hata_payi = 0.5 # 500 metrelik bir tolerans tanıyoruz

print(f"-> Test Edilen Hat: İstanbul - Yalova")
print(f"-> Algoritmanın Hesapladığı Mesafe: {hesaplanan_mesafe:.4f} km")

# Sıkı bir Assert Kontrolü: Eğer formülde veya veri tipinde hata varsa sistem çöker ve uyarır!
assert np.isclose(hesaplanan_mesafe, beklenen_mesafe, atol=hata_payi), \
    f"KRİTİK HATA: Mesafe algoritması yanlış hesaplıyor! Beklenen: {beklenen_mesafe} km, Hesaplanan: {hesaplanan_mesafe:.2f} km"

print("✅ Başarılı: Assert doğrulamasından geçildi. Mesafe algoritması doğru çalışıyor.")
print(f"📊 Toplamda {len(df_koordinat)} Transfer Merkezi için {mesafe_matrisi.shape} boyutlu tam matris başarıyla oluşturuldu.")


