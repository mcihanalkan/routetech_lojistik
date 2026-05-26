from ortools.sat.python import cp_model

# 1. Modeli Başlatma
model = cp_model.CpModel()

# Örnek İndeks Setleri
hatlar = ["TM1-TM2", "TM2-TM3", "TM1-TM4"]
gunler = ["Pazartesi", "Sali", "Carsamba"]
arac_turleri = ["Tir", "Kamyon"]

# 2. Karar Değişkenlerinin Tanımlanması
# OR-Tools'ta değişkenler sözlük (dictionary) yapısı içinde manuel veya döngüyle kurulur
kiralik_x = {}
spot_y = {}

# Maksimum araç sınırı (Üst sınır tanımlamak OR-Tools performansını artırır)
# Örneğin her hat için max 100 araç sınırı koyalım
max_arac = 100 

for h in hatlar:
    for g in gunler:
        for a in arac_turleri:
            # Kiralık değişkeni tanımlama (LowBound: 0, UpperBound: max_arac)
            kiralik_x[(h, g, a)] = model.NewIntVar(0, max_arac, f'kiralik_{h}_{g}_{a}')
            
            # Spot değişkeni tanımlama
            spot_y[(h, g, a)] = model.NewIntVar(0, max_arac, f'spot_{h}_{g}_{a}')

# Örnek Kullanım:
# print(kiralik_x[("TM1-TM2", "Pazartesi", "Tir")])


# Örnek Parametreler (Eray ve Finans biriminden gelecek veriler)
sabit_kira = 5000
km_maliyeti = 15
spot_birim_maliyet = 8500

# Daha önce tanımladığımız karar değişkenleri sözlükleri (Demsilidir)
# kiralik_x[(h, g, a)] ve spot_y[(h, g, a)] şeklinde tanımlandığını varsayıyoruz.

maliyet_kalemleri = []

# 2. Döngüyle Maliyet Denkleminin İnşası
for h in hatlar:
    dist = mesafe_verisi[h] # Eray'dan gelen mesafe bilgisi, hatlar arasındaki mesafeleri almak için kullanılıyor.
    for g in gunler:
        for a in arac_turleri:
            
            # Kiralık Araç Maliyet Katsayısı: (Sabit + Mesafe * KM Maliyeti)
            kiralik_birim_fiyat = sabit_kira + (dist * km_maliyeti)
            
            # Maliyet kalemini listeye ekle (Değişken * Katsayı)
            maliyet_kalemleri.append(kiralik_x[(h, g, a)] * kiralik_birim_fiyat)
            
            # Spot Araç Maliyet Kalemi
            maliyet_kalemleri.append(spot_y[(h, g, a)] * spot_birim_maliyet)

# 3. Ana Hedef: Toplam Maliyeti Minimize Et
model.Minimize(sum(maliyet_kalemleri))

# --- Çözüm Aşaması (Mantığı anlaman için ekliyorum) ---
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"Minimum Toplam Maliyet: {solver.ObjectiveValue()} TL")