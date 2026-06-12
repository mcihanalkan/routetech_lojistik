from ortools.sat.python import cp_model

# 1. Modeli Başlatma
model = cp_model.CpModel()

# --- INPUT VERİLERİ ---
hatlar = ["TM1-TM2", "TM2-TM3", "TM1-TM4"]
gunler = ["11_Mayis", "12_Mayis", "13_Mayis", "14_Mayis", "15_Mayis", "16_Mayis", "17_Mayis"]
arac_turleri = ["Tir", "Kamyon"]

# STOK BİLGİSİ: Her hatta her araç türünden GÜNLÜK kaç tane kiralık araç kullanabiliriz?
# Örnek: TM1-TM2 hattında günde max 1 Kamyon, TM1-TM4 hattında günde max 2 TIR + 1 Kamyon
kiralik_stok_gunluk = {
    ("TM1-TM2", "Kamyon"): 1,
    ("TM1-TM4", "Tir"): 2,
    ("TM1-TM4", "Kamyon"): 1,
}
# Stok günlük yenileniyor. Her gün bu kadar kiralık araç kullanabiliriz.

mesafe_verisi = {
    "TM1-TM2": 450,
    "TM2-TM3": 320,
    "TM1-TM4": 150
}

talep_verisi = {
    (h, g): 15000 for h in hatlar for g in gunler
}

arac_parametreleri = {
    "Tir": {
        "sabit_kira": 7000,      # Düzeltildi: sizin tablonuzda 7000
        "kiralik_km_maliyet": 13,
        "spot_sabit_maliyet": 11700,
        "spot_km_maliyet": 25,
        "kapasite_desi": 22400   # Düzeltildi
    },
    "Kamyon": {
        "sabit_kira": 5000,
        "kiralik_km_maliyet": 10,
        "spot_sabit_maliyet": 7638,
        "spot_km_maliyet": 21,
        "kapasite_desi": 12000
    }
}

# --- KARAR DEĞİŞKENLERİ ---
kiralik_x = {}
spot_y = {}
max_arac = 50

for h in hatlar:
    for g in gunler:
        for a in arac_turleri:
            kiralik_x[(h, g, a)] = model.NewIntVar(0, max_arac, f'kiralik_{h}_{g}_{a}')
            spot_y[(h, g, a)] = model.NewIntVar(0, max_arac, f'spot_{h}_{g}_{a}')

# --- KISIT 1: Kiralık Araç Stok Limiti (GÜNLÜK) ---
for h in hatlar:
    for g in gunler:
        for a in arac_turleri:
            if (h, a) in kiralik_stok_gunluk:
                limit = kiralik_stok_gunluk[(h, a)]
                model.Add(kiralik_x[(h, g, a)] <= limit)

# --- AMAÇ FONKSİYONU: Toplam Maliyet ---
maliyet_kalemleri = []

for h in hatlar:
    dist = mesafe_verisi[h]
    for g in gunler:
        for a in arac_turleri:
            p = arac_parametreleri[a]
            
            # Kiralık maliyet (HER ZAMAN EKLE - solver karar versin)
            kiralik_sefer_maliyeti = p["sabit_kira"] + (dist * p["kiralik_km_maliyet"])
            maliyet_kalemleri.append(kiralik_x[(h, g, a)] * kiralik_sefer_maliyeti)
            
            # Spot maliyet (HER ZAMAN EKLE)
            spot_sefer_maliyeti = p["spot_sabit_maliyet"] + (dist * p["spot_km_maliyet"])
            maliyet_kalemleri.append(spot_y[(h, g, a)] * spot_sefer_maliyeti)

model.Minimize(sum(maliyet_kalemleri))

# --- KISIT 2: Talebi Karşılama ---
for h in hatlar:
    for g in gunler:
        toplam_kapasite = []
        for a in arac_turleri:
            kap = arac_parametreleri[a]["kapasite_desi"]
            toplam_kapasite.append(kiralik_x[(h, g, a)] * kap)
            toplam_kapasite.append(spot_y[(h, g, a)] * kap)
        model.Add(sum(toplam_kapasite) >= talep_verisi[(h, g)])

# --- ÇÖZÜM ---
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
status = solver.Solve(model)

# --- SONUÇLARI YAZDIR ---
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"OPTIMIZASYON BASARILI")
    print(f"Toplam Maliyet: {solver.ObjectiveValue():,.0f} TL\n")
    
    for h in hatlar:
        for g in gunler:
            for a in arac_turleri:
                k_adet = solver.Value(kiralik_x[(h, g, a)])
                s_adet = solver.Value(spot_y[(h, g, a)])
                if k_adet > 0 or s_adet > 0:
                    kap = arac_parametreleri[a]["kapasite_desi"]
                    k_kap = k_adet * kap
                    s_kap = s_adet * kap
                    print(f"{g} | {h} | {a}: Kiralık={k_adet} ({k_kap} desi), Spot={s_adet} ({s_kap} desi)")
else:
    print("Çözüm bulunamadı! Talepler çok yüksek veya stok yetersiz.")