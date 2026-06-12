from pathlib import Path
import pandas as pd
from ortools.sat.python import cp_model

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "haversine"))
from haversine import GetDistanceMatrixAsList

# 1. Modeli Başlatma
model = cp_model.CpModel()

# --- INPUT VERİLERİ ---
tmler = [
    "Mersin", "Kütahya", "Kocaeli", "Eskişehir", "İstanbul", "Bilecik",
    "Balıkesir", "Şanlıurfa", "Tekirdağ", "Sivas", "Yalova", "Manisa",
    "Isparta", "Mardin", "Erzincan", "Zonguldak", "Karaman", "Denizli"
]

# ortools_payload.csv dosyasından hatları ve talepleri çek
payload_csv = Path(__file__).parent.parent / "src" / "predict_model" / "ortools_payload.csv"

if payload_csv.exists():
    df_payload = pd.read_csv(payload_csv)
    
    # Debug için terminale sütun isimlerini yazdıralım, gözümüzle görelim
    print(f"🔍 CSV Sütunları: {list(df_payload.columns)}")
    
    hatlar = []
    talep_verisi = {}
    gecici_gunler = set()
    
    for _, row in df_payload.iterrows():
        # Sütun adı ne olursa olsun güvenli bir şekilde çekmeyi dene
        source = row.get('source_tm') or row.get('kaynak_tm') or row.get('source') or row.get('kaynak')
        dest = row.get('destination_tm') or row.get('varis_tm') or row.get('destination') or row.get('varis')
        
        # Eğer yukarıdakiler de boş dönerse (None), CSV'nin ilk iki sütununu direkt al
        if pd.isna(source) or source is None:
            source = row.iloc[1] # Genellikle 0 tarih, 1 kaynak, 2 varıştır
        if pd.isna(dest) or dest is None:
            dest = row.iloc[2]
            
        date_str = str(row['date']) if 'date' in row else str(row.iloc[0])
        q50 = float(row.get('q50', 0))
        
        hat = f"{str(source).strip()}-{str(dest).strip()}"
        if hat not in hatlar:
            hatlar.append(hat)
        
        # Tarih dönüşümü (2026-05-11 -> 11_Mayis)
        tarih_obj = pd.to_datetime(date_str)
        gun_adi = f"{tarih_obj.day:02d}_Mayis" 
        
        gecici_gunler.add(gun_adi)
        talep_verisi[(hat, gun_adi)] = int(q50)
    
    hatlar = list(set(hatlar))
    gunler = sorted(list(gecici_gunler))
    
    print(f"✅ ortools_payload.csv'den başarıyla yüklendi:")
    print(f"   Hatlar: {len(hatlar)} adet benzersiz hat")
    print(f"   Günler: {gunler}")
    print(f"   Toplam Veri Noktası: {len(talep_verisi)}")
else:
    print(f"⚠️  {payload_csv} bulunamadı! Sabit veriler kullanılıyor.")
    hatlar = [f"{tm1}-{tm2}" for tm1 in tmler for tm2 in tmler if tm1 != tm2]
    gunler = ["11_Mayis", "12_Mayis", "13_Mayis", "14_Mayis", "15_Mayis", "16_Mayis", "17_Mayis"]
    talep_verisi = {(h, g): 15000 for h in hatlar for g in gunler}

arac_turleri = ["Tir", "Kamyon", "Hafif Kam", "Kamyonet"]

# Mesafe matrisi
centers, distances_2d = GetDistanceMatrixAsList()
tm_index = {tm: idx for idx, tm in enumerate(centers)}

# Hatlar için mesafe verisi hesapla (sadece mevcut hatlar)
mesafe_verisi = {}
for hat in hatlar:
    tm1, tm2 = hat.split("-")
    if tm1 in tm_index and tm2 in tm_index:
        idx1 = tm_index[tm1]
        idx2 = tm_index[tm2]
        mesafe_verisi[hat] = distances_2d[idx1][idx2]
    else:
        print(f"⚠️  {hat} için transfer merkezi koordinatı bulunamadı!")

# STOK BİLGİSİ (isteğe bağlı, belirlenen hatlar için)
kiralik_stok_gunluk = {
    ("İstanbul-Yalova", "Tir"): 2,
    ("İstanbul-Eskişehir", "Tir"): 2,
    ("Kocaeli-Yalova", "Tir"): 1,
    ("İstanbul-Manisa", "Tir"): 1,
    ("İstanbul-Balıkesir", "Tir"): 1,
    ("İstanbul-Tekirdağ", "Tir"): 1,
    ("Kocaeli-İstanbul", "Tir"): 1,
    ("Kocaeli-Tekirdağ", "Tir"): 1,
    ("Yalova-Eskişehir", "Kamyon"): 1,
    ("Kocaeli-Balıkesir", "Kamyon"): 1,
    ("Kocaeli-Eskişehir", "Kamyon"): 1,
    ("Yalova-Tekirdağ", "Kamyon"): 1,
}

# ARAÇ PARAMETRELERİ
arac_parametreleri = {
    "Tir":      {"sabit_kira": 7000,  "kiralik_km_maliyet": 13, "spot_sabit_maliyet": 11700, "spot_km_maliyet": 25, "kapasite_desi": 22400},
    "Kamyon":   {"sabit_kira": 5000,  "kiralik_km_maliyet": 10, "spot_sabit_maliyet": 7638,  "spot_km_maliyet": 21, "kapasite_desi": 12000},
    "Hafif Kam":{"sabit_kira": 5000,  "kiralik_km_maliyet": 10, "spot_sabit_maliyet": 8750,  "spot_km_maliyet": 20, "kapasite_desi": 7200},
    "Kamyonet": {"sabit_kira": 3750,  "kiralik_km_maliyet": 6,  "spot_sabit_maliyet": 4750,  "spot_km_maliyet": 18, "kapasite_desi": 5600},
}

SLA_GECIKME_CEZA_TL_PER_DESI = 0.5

# --- OPTİMİZASYON 1: Spot araç için gerçekçi üst limit ---
# Her hat için en kötü durumda (tüm biriken talep) kaç araç gerekebilir?
max_talep_hatta = max(talep_verisi.values()) * len(gunler)  # tüm haftanın talebi bir günde birikirse
min_kapasite = min(p["kapasite_desi"] for p in arac_parametreleri.values())
max_spot = -(-max_talep_hatta // min_kapasite) + 1  # ceiling division + 1 tampon
max_spot = min(max_spot, 20)  # mutlak tavan: 20 araç yeterli

max_ertelenen = max_talep_hatta

# --- KARAR DEĞİŞKENLERİ ---
kiralik_x = {}
spot_y = {}
ertelenen_talep = {}
biriken_talep = {}

for h in hatlar:
    for g in gunler:
        # OPTİMİZASYON 2: Sadece stoku olan hat-araç çiftleri için kiralik_x yarat
        for a in arac_turleri:
            if (h, a) in kiralik_stok_gunluk:
                limit = kiralik_stok_gunluk[(h, a)]
                kiralik_x[(h, g, a)] = model.NewIntVar(0, limit, f'kiralik_{h}_{g}_{a}')
            # Stok yoksa değişken bile yaratılmıyor → arama uzayı küçülüyor

            # Spot: gerçekçi üst limit
            spot_y[(h, g, a)] = model.NewIntVar(0, max_spot, f'spot_{h}_{g}_{a}')

        ertelenen_talep[(h, g)] = model.NewIntVar(0, max_ertelenen, f'ertelenen_{h}_{g}')
        biriken_talep[(h, g)] = model.NewIntVar(0, max_ertelenen + 5000, f'biriken_{h}_{g}')

# --- KISIT 1: Kiralık Araç Stok Kontrolü (sadece var olanlar için) ---
# Değişkenler zaten üst limitli yaratıldı → ekstra kısıta gerek yok

# --- KISIT 2: Talep ve Ertelenen İlişkisi ---
for h in hatlar:
    for idx, g in enumerate(gunler):
        if idx == 0:
            model.Add(biriken_talep[(h, g)] == talep_verisi[(h, g)])
        else:
            onceki_gun = gunler[idx - 1]
            model.Add(biriken_talep[(h, g)] == ertelenen_talep[(h, onceki_gun)] + talep_verisi[(h, g)])

# --- KISIT 3: Kapasite ≥ Biriken Talep - Ertelenen ---
for h in hatlar:
    for g in gunler:
        toplam_kapasite = []
        for a in arac_turleri:
            kap = arac_parametreleri[a]["kapasite_desi"]
            if (h, g, a) in kiralik_x:
                toplam_kapasite.append(kiralik_x[(h, g, a)] * kap)
            toplam_kapasite.append(spot_y[(h, g, a)] * kap)

        model.Add(sum(toplam_kapasite) + ertelenen_talep[(h, g)] >= biriken_talep[(h, g)])

# --- AMAÇ FONKSİYONU ---
# OPTİMİZASYON 3: LinearExpr.Sum kullan (daha hızlı model kurma)
maliyet_kalemleri = []

for h in hatlar:
    dist = mesafe_verisi[h]
    for g in gunler:
        for a in arac_turleri:
            p = arac_parametreleri[a]

            if (h, g, a) in kiralik_x:
                kiralik_maliyet = int(p["sabit_kira"] + dist * p["kiralik_km_maliyet"]) *1000
                maliyet_kalemleri.append(kiralik_x[(h, g, a)] * kiralik_maliyet)

            spot_maliyet = int(p["spot_sabit_maliyet"] + dist * p["spot_km_maliyet"])*1000
            maliyet_kalemleri.append(spot_y[(h, g, a)] * spot_maliyet)

        ceza = int(SLA_GECIKME_CEZA_TL_PER_DESI * 1000)  # integer aritmetik için ölçekle
        maliyet_kalemleri.append(ertelenen_talep[(h, g)] * ceza)

# LinearExpr.Sum, uzun liste toplama için Python sum()'dan çok daha hızlı
model.Minimize(cp_model.LinearExpr.Sum(maliyet_kalemleri))

# --- OPTİMİZASYON 4: Solver Parametreleri ---
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 120.0
solver.parameters.num_search_workers = 8      # çok çekirdekli paralel arama
solver.parameters.log_search_progress = True  # ilerlemeyi terminalde görmek için

status = solver.Solve(model)

# --- SONUÇLARI DOSYAYA YAZDIR ---
output_file = Path(__file__).parent.parent / "results" / "optimization_results.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        durum_etiketi = "OPTIMAL" if status == cp_model.OPTIMAL else "UYGUN (Optimal değil)"
        f.write("=" * 80 + "\n")
        f.write(f"OPTİMİZASYON BAŞARILI — {durum_etiketi}\n")
        f.write("=" * 80 + "\n\n")

        toplam_ertelenen_desi = 0

        for h in hatlar:
            for g in gunler:
                for a in arac_turleri:
                    k_adet = solver.Value(kiralik_x[(h, g, a)]) if (h, g, a) in kiralik_x else 0
                    s_adet = solver.Value(spot_y[(h, g, a)])
                    if k_adet > 0 or s_adet > 0:
                        kap = arac_parametreleri[a]["kapasite_desi"]
                        f.write(f"{g} | {h} | {a}: Kiralık={k_adet}, Spot={s_adet} "
                              f"(Toplam {k_adet * kap + s_adet * kap} desi)\n")

                ert = solver.Value(ertelenen_talep[(h, g)])
                if ert > 0:
                    ceza_tl = ert * SLA_GECIKME_CEZA_TL_PER_DESI
                    f.write(f"  -> {g} | {h}: {ert:.0f} desi ERTELENDİ (Ceza: {ceza_tl:.2f} TL)\n")
                    toplam_ertelenen_desi += ert

        f.write("\n" + "=" * 80 + "\n")
        f.write("ÖZET İSTATİSTİKLER\n")
        f.write("=" * 80 + "\n")
        
        gercek_maliyet = solver.ObjectiveValue() / 1000
        f.write(f"Toplam Maliyet: {gercek_maliyet:,.0f} TL\n")
        f.write(f"Toplam Ertelenen Desi: {toplam_ertelenen_desi:.0f} desi\n\n")

        f.write("Araç Kullanım İstatistikleri:\n")
        for a in arac_turleri:
            toplam_k = sum(
                solver.Value(kiralik_x[(h, g, a)])
                for h in hatlar for g in gunler
                if (h, g, a) in kiralik_x
            )
            toplam_s = sum(solver.Value(spot_y[(h, g, a)]) for h in hatlar for g in gunler)
            if toplam_k + toplam_s > 0:
                f.write(f"  {a}: Kiralık={toplam_k} sefer, Spot={toplam_s} sefer, "
                      f"Toplam={(toplam_k + toplam_s)} sefer\n")
    else:
        f.write("Çözüm bulunamadı! Kısıtları kontrol edin.\n")

print(f"Sonuçlar {output_file} dosyasına yazıldı.")