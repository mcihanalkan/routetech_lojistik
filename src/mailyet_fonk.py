import pulp

# Örnek Maliyet Parametreleri (Bu veriler ileride Eray veya sistemden gelecek)
sabit_kira = 5000     # Kiralık aracın günlük sabit maliyeti
km_maliyeti = 15      # Kiralık aracın kilometre başı maliyeti
spot_birim_maliyet = 8500  # Spot aracın hat bazlı toplam maliyeti

# Amaç Fonksiyonu için boş bir liste (Toplam Listesi)
maliyet_kalemleri = []

# Örnek veriler

# 1. İndeks Setleri: Problemin boyutlarını belirler
hatlar = ["IST-ANK", "ANK-IZM", "IST-IZM"]
gunler = ["Pazartesi", "Sali", "Carsamba", "Persembe", "Cuma"]
arac_turleri = ["Kucuk_Kamyon", "Buyuk_Tir"]

# 2. Mesafe Verisi: Eray'ın matrisinden gelecek veriyi simüle eder
# Her hat için gidiş-dönüş toplam mesafesi (km)
mesafe_verisi = {
    "IST-ANK": 450,
    "ANK-IZM": 590,
    "IST-IZM": 480
}

# 3. Maliyet Parametreleri: Şirketin finansal verileri
sabit_kira = 4500          # Aracın kapıda yatma maliyeti (TL)
km_maliyeti = 12           # Yakıt ve bakım maliyeti (TL/km)
spot_birim_maliyet = 12000 # Spot tırın tek seferlik piyasa fiyatı (TL)

# Tüm boyutlar üzerinde iç içe döngüler
for h in hatlar:
    # Eray'ın hesaplayacağı mesafe bilgisi (Örnek: dist_matrix[h])
    dist = mesafe_verisi[h] 
    
    for g in gunler:
        for a in arac_turleri:
            # 1. Kiralık Araç Maliyeti: Araç Sayısı * (Sabit + Mesafe * KM Maliyeti)
            kiralik_maliyet = kiralik_x[h][g][a] * (sabit_kira + dist * km_maliyeti)
            maliyet_kalemleri.append(kiralik_maliyet)
            
            # 2. Spot Araç Maliyeti: Spot Araç Sayısı * Spot Birim Maliyet
            spot_maliyet = spot_y[h][g][a] * spot_birim_maliyet
            maliyet_kalemleri.append(spot_maliyet)

# 3. Tüm listeyi lpSum ile toplayıp modele hedef olarak atama
model += pulp.lpSum(maliyet_kalemleri), "Toplam_Lojistik_Maliyeti"