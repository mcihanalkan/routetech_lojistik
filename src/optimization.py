from ortools.sat.python import cp_model

# 1. Modeli Başlatma
model = cp_model.CpModel()

# --- INPUT VERİLERİ (Arkadaşlarından ve Şartnameden Gelecek Kısımlar) ---

# Örnek İndeks Setleri (11-17 Mayıs haftası için günleri güncelledik)
hatlar = ["TM1-TM2", "TM2-TM3", "TM1-TM4"]
gunler = ["11_Mayis", "12_Mayis", "13_Mayis", "14_Mayis", "15_Mayis", "16_Mayis", "17_Mayis"]
arac_turleri = ["Tir", "Kamyon"]

# Eray'ın Haversine ile hesaplayacağı mesafe verisi (Örnek km'ler)
mesafe_verisi = {
    "TM1-TM2": 450,
    "TM2-TM3": 320,
    "TM1-TM4": 150
}

# Ebubekir'in ML ile tahmin edeceği 11-17 Mayıs haftası günlük desi talepleri
# Format: talep_verisi[(hat, gun)]
talep_verisi = {
    (h, g): 15000 for h in hatlar for g in gunler  # Test için her hatta günlük 15k desi varsayalım
}

# Şartnamede belirtilen araç özellikleri ve maliyet parametreleri
arac_parametreleri = {
    "Tir": {
        "sabit_kira": 6000,
        "kiralik_km_maliyet": 15,
        "spot_sabit_maliyet": 9000,
        "spot_km_maliyet": 22,
        "kapasite_desi": 24000  # Bir tırın taşıyabileceği max desi
    },
    "Kamyon": {
        "sabit_kira": 3500,
        "kiralik_km_maliyet": 10,
        "spot_sabit_maliyet": 5500,
        "spot_km_maliyet": 15,
        "kapasite_desi": 12000  # Bir kamyonun taşıyabileceği max desi
    }
}

# --- OPTİMİZASYON MOTORUNUN KURULMASI ---

# 2. Karar Değişkenlerinin Tanımlanması
kiralik_x = {}
spot_y = {}
max_arac = 50 # Çözüm hızını optimize etmek için makul bir üst sınır

for h in hatlar:
    for g in gunler:
        for a in arac_turleri:
            kiralik_x[(h, g, a)] = model.NewIntVar(0, max_arac, f'kiralik_{h}_{g}_{a}')
            spot_y[(h, g, a)] = model.NewIntVar(0, max_arac, f'spot_{h}_{g}_{a}')

# 3. Döngüyle Maliyet Denkleminin (Amaç Fonksiyonu) İnşası
maliyet_kalemleri = []

for h in hatlar:
    dist = mesafe_verisi[h]
    for g in gunler:
        for a in arac_turleri:
            # Araç türüne özgü parametreleri çekiyoruz
            p = arac_parametreleri[a]
            
            # Kiralık Araç Sefer Maliyeti = Sabit Kira + (Gidilen Mesafe * KM Maliyeti)
            kiralik_sefer_fiyati = p["sabit_kira"] + (dist * p["kiralik_km_maliyet"])
            maliyet_kalemleri.append(kiralik_x[(h, g, a)] * kiralik_sefer_fiyati)
            
            # Spot Araç Sefer Maliyeti = Spot Sabit + (Gidilen Mesafe * Spot KM Maliyeti)
            spot_sefer_fiyati = p["spot_sabit_maliyet"] + (dist * p["spot_km_maliyet"])
            maliyet_kalemleri.append(spot_y[(h, g, a)] * spot_sefer_fiyati)

model.Minimize(sum(maliyet_kalemleri))

# 4. Kısıtların Eklenmesi (Talebi Karşılama Kısıtı)
# Her hat ve her gün için: Çıkan araçların toplam kapasitesi, tahmini talepten büyük olmalı.
for h in hatlar:
    for g in gunler:
        toplam_tasima_kapasitesi = []
        for a in arac_turleri:
            kapasite = arac_parametreleri[a]["kapasite_desi"]
            
            # Kiralık ve spot araçların getirdiği toplam desi kapasitesi
            toplam_tasima_kapasitesi.append(kiralik_x[(h, g, a)] * kapasite)
            toplam_tasima_kapasitesi.append(spot_y[(h, g, a)] * kapasite)
        
        # Matematiksel Kısıt: Kapasite >= Ebubekir'in Tahmini Talebi
        model.Add(sum(toplam_tasima_kapasitesi) >= talep_verisi[(h, g)])

# 5. Çözüm Aşaması
solver = cp_model.CpSolver()
# Yarışmadaki 10 dakika (600 saniye) kısıtını aşmamak için emniyet sınırı koyuyoruz
solver.parameters.max_time_in_seconds = 60.0 

status = solver.Solve(model)

# 6. Sonuçların Alınması ve Excel Hazırlık Yapısı
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"--- OPTİMİZASYON BAŞARILI ---")
    print(f"README'ye Yazılacak Toplam Maliyet: {solver.ObjectiveValue()} TL\n")
    
    # Test için ilk birkaç kararı ekrana yazdıralım
    for h in hatlar:
        for g in gunler:
            for a in arac_turleri:
                k_adet = solver.Value(kiralik_x[(h, g, a)])
                s_adet = solver.Value(spot_y[(h, g, a)])
                if k_adet > 0 or s_adet > 0:
                    print(f"{g} günü {h} hattı için -> Kiralık {a}: {k_adet} adet | Spot {a}: {s_adet} adet")
else:
    print("Geçerli bir çözüm bulunamadı! Kısıtları veya verileri kontrol edin.")