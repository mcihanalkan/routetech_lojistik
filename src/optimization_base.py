from pathlib import Path
import pandas as pd
from ortools.sat.python import cp_model
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "haversine"))
from haversine import GetDistanceMatrixAsList, GetCenters

# =============================================================================
# 1. MODELİ BAŞLATMA VE VERİ ÇEKİMİ (BASELINE - SAF MODEL)
# =============================================================================
model = cp_model.CpModel()
centers = GetCenters()
payload_csv = Path(__file__).parent.parent / "src" / "predict_model" / "ortools_payload.csv"

if payload_csv.exists():
    df_payload = pd.read_csv(payload_csv)
    hatlar_sirali = []
    talep_verisi = {} 
    gecici_gunler = set()

    for _, row in df_payload.iterrows():
        source = row.get('source_tm') or row.iloc[1]
        dest   = row.get('destination_tm') or row.iloc[2]
        date_str = str(row['date']) if 'date' in row else str(row.iloc[0])
        
        recommended = float(row.get('recommended_demand', row.get('q50', 0)))
        hat = f"{str(source).strip()}-{str(dest).strip()}"
        hatlar_sirali.append(hat)
        gun_adi = f"{pd.to_datetime(date_str).day:02d}_Mayis"
        gecici_gunler.add(gun_adi)
        talep_verisi[(hat, gun_adi)] = int(recommended)

    hatlar = list(dict.fromkeys(hatlar_sirali))
    gunler = sorted(list(gecici_gunler))
else:
    print("Payload bulunamadı.")
    sys.exit()

arac_turleri = ["Tir", "Kamyon", "Hafif Kam", "Kamyonet"]
distances_2d = GetDistanceMatrixAsList()
tm_index = {tm: idx for idx, tm in enumerate(centers)}

mesafe_verisi = {}
for hat in hatlar:
    parcalar = hat.split("-", maxsplit=1)
    if len(parcalar) == 2 and parcalar[0] in tm_index and parcalar[1] in tm_index:
        mesafe_verisi[hat] = distances_2d[tm_index[parcalar[0]]][tm_index[parcalar[1]]]

# --- Kiralık Stok ve Araç Parametreleri ---
rented_stoks_csv = Path(__file__).parent.parent / "data" / "static_datas" / "rented_stoks.csv"
kiralik_stok_gunluk = {}
if rented_stoks_csv.exists():
    df_rented = pd.read_csv(rented_stoks_csv)
    for _, row in df_rented.iterrows():
        kiralik_stok_gunluk[(row['route'], row['vehicle_type'])] = int(row['quantity'])

car_params_csv = Path(__file__).parent.parent / "data" / "static_datas" / "car_parameters.csv"
arac_parametreleri = {}
if car_params_csv.exists():
    df_car_params = pd.read_csv(car_params_csv)
    for _, row in df_car_params.iterrows():
        arac_parametreleri[row['vehicle_type']] = {
            "sabit_kira": int(row['sabit_kira']),
            "kiralik_km_maliyet": int(row['kiralik_km_maliyet']),
            "spot_sabit_maliyet": int(row['spot_sabit_maliyet']),
            "spot_km_maliyet": int(row['spot_km_maliyet']),
            "kapasite_desi": int(row['kapasite_desi']),
        }

# =============================================================================
# 2. KARAR DEĞİŞKENLERİ (SADECE DİREKT ARAÇLAR)
# =============================================================================
max_spot = 500
kiralik_x = {}
spot_y = {}

for h in hatlar:
    for g in gunler:
        for a in arac_turleri:
            spot_y[(h, g, a)] = model.NewIntVar(0, max_spot, f'spot_{h}_{g}_{a}')
            # Kiralıklar doğrudan sabit/zorunlu atama
            kiralik_x[(h, g, a)] = model.NewIntVar(0, kiralik_stok_gunluk.get((h,a),0), f'kiralik_{h}_{g}_{a}')

# =============================================================================
# 3. KISITLAR (SADE VE NET)
# =============================================================================
for h in hatlar:
    for g in gunler:
        talep = talep_verisi.get((h, g), 0)

        # 1. Kiralık Araç Kapasitesi ve Yükü
        kiralik_aktif_kapasite = cp_model.LinearExpr.Sum([
            kiralik_x[(h, g, a)] * arac_parametreleri[a]["kapasite_desi"] for a in arac_turleri
        ])
        kiralik_tasinan_yuk = model.NewIntVar(0, 500000, f'kiralik_net_yuk_{h}_{g}')
        model.Add(kiralik_tasinan_yuk <= kiralik_aktif_kapasite)

        # 2. Spot Araç Kapasitesi, Yükü ve %10 Kuralı
        spot_tasinan_yuk_listesi = []
        for a in arac_turleri:
            kap = arac_parametreleri[a]["kapasite_desi"]
            tasinan_yuk_a = model.NewIntVar(0, max_spot * kap, f'spot_net_yuk_{h}_{g}_{a}')
            spot_tasinan_yuk_listesi.append(tasinan_yuk_a)

            # Sınır: Kasa limitini aşamaz
            model.Add(tasinan_yuk_a <= spot_y[(h, g, a)] * kap)

            # KISIT A: %10 Minimum Doluluk
            # NOT: Eğer base test çökerse (INFEASIBLE), aşağıdaki satırı yorum satırı yap!
            # model.Add(spot_y[(h, g, a)] * kap <= tasinan_yuk_a * 10)

        # 3. KÜTLE DENGESİ: O günkü talep == Kiralık Taşıma + Spot Taşıma (Erteleme YOK)
        tasinan_toplam = cp_model.LinearExpr.Sum([kiralik_tasinan_yuk] + spot_tasinan_yuk_listesi)
        model.Add(talep == tasinan_toplam)

        # 4. Kiralık Araç Zimmet (Stok) Kısıtı
        for a in arac_turleri:
            model.Add(kiralik_x[(h, g, a)] <= kiralik_stok_gunluk.get((h, a), 0))

# =============================================================================
# 4. AMAÇ FONKSİYONU
# =============================================================================
maliyet_kalemleri = []
kiralik_sabit_toplam = 0

for h in hatlar:
    dist = mesafe_verisi.get(h, 0)
    for g in gunler:
        
        # Kiralık Muhasebesi (Batık Maliyet - Sunk Cost)
        for a in arac_turleri:
            adet = kiralik_stok_gunluk.get((h, a), 0)
            if adet > 0:
                gun_maliyet = adet * int(arac_parametreleri[a]["sabit_kira"] + dist * arac_parametreleri[a]["kiralik_km_maliyet"])
                kiralik_sabit_toplam += gun_maliyet

        # Spot Araç Faturası (Optimizasyona Giren Kısım)
        for a in arac_turleri:
            p = arac_parametreleri[a]
            spot_maliyet_katsayi = int(p["spot_sabit_maliyet"] + dist * p["spot_km_maliyet"])
            maliyet_kalemleri.append(spot_y[(h, g, a)] * spot_maliyet_katsayi)

model.Minimize(cp_model.LinearExpr.Sum(maliyet_kalemleri))

# =============================================================================
# 5. ÇÖZÜCÜ VE YAZDIRMA
# =============================================================================
solver = cp_model.CpSolver()
max_time_to_solve = 300
solver.parameters.max_time_in_seconds = max_time_to_solve
solver.parameters.num_search_workers = 4
solver.parameters.log_search_progress = True

status = solver.Solve(model)

output_file = Path(__file__).parent.parent / "results" / "optimization_base_results.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        durum = "OPTIMAL (Kusursuz)" if status == cp_model.OPTIMAL else "UYGUN (Süre Sınırı)"
        baslik = f"={'=' * 80}\nBASE MODEL BAŞARILI — {durum}\n={'=' * 80}\n"
        f.write(baslik + "\n")
        print(baslik)

        # Tablo Başlıkları
        sutun_basliklari = "Tarih | Araç Türü | Hat | Kapasite (Desi) | Maliyet"
        f.write(sutun_basliklari + "\n")
        f.write("-" * 80 + "\n")
        print(sutun_basliklari)
        print("-" * 80)

        spot_toplam_maliyet = 0

        for h in hatlar:
            dist = mesafe_verisi.get(h, 0)
            for g in gunler:
                for a in arac_turleri:
                    k_adet = solver.Value(kiralik_x[(h, g, a)])
                    s_adet = solver.Value(spot_y[(h, g, a)])
                    p = arac_parametreleri[a]
                    kapasite = p["kapasite_desi"]

                    # Kiralık Araçları Tek Tek Yazdır (Her araç için 1 satır)
                    for i in range(k_adet):
                        maliyet = int(p["sabit_kira"] + dist * p["kiralik_km_maliyet"])
                        metin = f"{g} | Kiralık {a} | {h} | {kapasite} Desi | {maliyet} TL\n"
                        f.write(metin)
                        print(metin.strip())

                    # Spot Araçları Tek Tek Yazdır (Her araç için 1 satır)
                    for i in range(s_adet):
                        maliyet = int(p["spot_sabit_maliyet"] + dist * p["spot_km_maliyet"])
                        spot_toplam_maliyet += maliyet
                        metin = f"{g} | Spot {a} | {h} | {kapasite} Desi | {maliyet} TL\n"
                        f.write(metin)
                        print(metin.strip())

        genel_toplam = kiralik_sabit_toplam + spot_toplam_maliyet

        ozet = f"""
{'=' * 80}
BASE MODEL (Uğramasız & Ertelemesiz) İSTATİSTİKLERİ
{'=' * 80}
  Kiralık Araç Sabit Maliyeti : {kiralik_sabit_toplam:>15,.0f} TL
  Spot Araç Maliyeti (Direkt) : {spot_toplam_maliyet:>15,.0f} TL
{'─' * 80}
  TOPLAM MALİYET              : {genel_toplam:>15,.0f} TL
{'=' * 80}
  Çözücü Süre                 : {solver.WallTime():>15.2f} sn
{'=' * 80}
"""
        f.write(ozet)
        print(ozet)
    else:
        hata = "❌ Çözüm bulunamadı (INFEASIBLE). Muhtemelen %10 kuralı erteleme olmadan aşılamadı.\n"
        f.write(hata)
        print(hata)