from pathlib import Path
import pandas as pd
from ortools.sat.python import cp_model
import sys
from datetime import datetime
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

kiralik_net_yuk_dict = {}
spot_net_yuk_dict = {}
for h in hatlar:
    for g in gunler:
        talep = talep_verisi.get((h, g), 0)

        # 1. Kiralık Araç Kapasitesi ve Yükü
        kiralik_aktif_kapasite = cp_model.LinearExpr.Sum([
            kiralik_x[(h, g, a)] * arac_parametreleri[a]["kapasite_desi"] for a in arac_turleri
        ])
        kiralik_tasinan_yuk = model.NewIntVar(0, 500000, f'kiralik_net_yuk_{h}_{g}')
        kiralik_net_yuk_dict[(h, g)] = kiralik_tasinan_yuk # <-- SÖZLÜĞE ALDIK
        model.Add(kiralik_tasinan_yuk <= kiralik_aktif_kapasite)

        # kiralik_tasinan_yuk_listesi = []
        # for a in arac_turleri:
        #     kap = arac_parametreleri[a]["kapasite_desi"]
        #     kiralik_tasinan_yuk_a = model.NewIntVar(0, kiralik_stok_gunluk.get((h,a)) * kap, f'kiralik_net_yuk_{h}_{g}_{a}')
        #     kiralik_net_yuk_dict[(h, g, a)] = kiralik_tasinan_yuk_a # <-- SÖZLÜĞE ALDIK
        #     kiralik_tasinan_yuk_listesi.append(kiralik_tasinan_yuk_a)

        #     # Sınır: Kasa limitini aşamaz
        #     model.Add(tasinan_yuk_a <= kiralik_x[(h, g, a)] * kap)

        # 2. Spot Araç Kapasitesi, Yükü ve %10 Kuralı
        spot_tasinan_yuk_listesi = []
        for a in arac_turleri:
            kap = arac_parametreleri[a]["kapasite_desi"]
            tasinan_yuk_a = model.NewIntVar(0, max_spot * kap, f'spot_net_yuk_{h}_{g}_{a}')
            spot_net_yuk_dict[(h, g, a)] = tasinan_yuk_a # <-- SÖZLÜĞE ALDIK
            spot_tasinan_yuk_listesi.append(tasinan_yuk_a)

            # Sınır: Kasa limitini aşamaz
            model.Add(tasinan_yuk_a <= spot_y[(h, g, a)] * kap)

            # KISIT A: %10 Minimum Doluluk
            # NOT: Eğer base test çökerse (INFEASIBLE), aşağıdaki satırı yorum satırı yap!
            # model.Add(spot_y[(h, g, a)] * kap <= tasinan_yuk_a * 10)

        # 3. KÜTLE DENGESİ: O günkü talep == Kiralık Taşıma + Spot Taşıma (Erteleme YOK)
        tasinan_toplam = cp_model.LinearExpr.Sum([kiralik_tasinan_yuk] + spot_tasinan_yuk_listesi)
        model.Add(talep == tasinan_toplam) # burada modele aslında bugün bu hatta taşıyacağın toplam yük bu hattaki bugünkü talebe eşit olmak zorunda diyoruz. O yüzden kiralik ve spot karar değişkenleri dolduruluyor.

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
                p = arac_parametreleri[a]
                kiralik_maliyet_katsayi = int(p["sabit_kira"] + dist * p["kiralik_km_maliyet"])
                # # kiralik_x[h, g, a] * maliyet_katsayi ekle!
                # maliyet_kalemleri.append(kiralik_x[(h, g, a)] * kiralik_maliyet_katsayi)
                kiralik_sabit_toplam += kiralik_maliyet_katsayi * adet
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
end_time = datetime.now()

output_file = Path(__file__).parent.parent / "results" / "optimization_base_results.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        durum = "OPTIMAL (Kusursuz)" if status == cp_model.OPTIMAL else "UYGUN (Süre Sınırı)"
        baslik = f"={'=' * 80}\nBASE MODEL BAŞARILI — {durum}\n MODEL HESAPLAMASI BİTİŞ TARİHİ: {end_time}\n={'=' * 80}\n"
        f.write(baslik + "\n")
        print(baslik)

        # Tablo Başlıkları
        sutun_basliklari = "Tarih | Araç Türü | Hat | TAŞINAN NET DESİ | Maliyet"
        f.write(sutun_basliklari + "\n")
        f.write("-" * 80 + "\n")
        print(sutun_basliklari)
        print("-" * 80)

        spot_toplam_maliyet = 0
        used_rented_count = 0
        used_spot_count = 0
        for h in hatlar:
            dist = mesafe_verisi.get(h, 0)
            for g in gunler:
                
                # 1. O gün o hatta Kiralık araçlara düşen TOPLAM net yükü çek
                k_yuk_kalan = solver.Value(kiralik_net_yuk_dict[(h, g)])

                for a in arac_turleri:
                    p = arac_parametreleri[a]
                    kapasite = p["kapasite_desi"]

                    # --- KİRALIK ARAÇLARIN PAYLAŞTIRILMASI ---
                    k_adet = solver.Value(kiralik_x[(h, g, a)])
                    used_rented_count += k_adet
                    for i in range(k_adet):
                        # Araca ya tam kapasite doldur, ya da elinde kalan son yükü ver
                        yuk_bu_araca = min(k_yuk_kalan, kapasite)
                        k_yuk_kalan -= yuk_bu_araca 
                        
                        maliyet = int(p["sabit_kira"] + dist * p["kiralik_km_maliyet"])
                        metin = f"{g} | Kiralık {a} | {h} | {yuk_bu_araca} Desi | {maliyet} TL\n"
                        f.write(metin)
                        print(metin.strip())

                    # --- SPOT ARAÇLARIN PAYLAŞTIRILMASI ---
                    s_adet = solver.Value(spot_y[(h, g, a)])
                    used_spot_count += s_adet
                    # O araç tipindeki spotlara düşen TOPLAM net yük
                    s_yuk_kalan = solver.Value(spot_net_yuk_dict[(h, g, a)]) 

                    for i in range(s_adet):
                        # Araca ya tam kapasite doldur, ya da elinde kalan son yükü ver
                        yuk_bu_araca = min(s_yuk_kalan, kapasite)
                        s_yuk_kalan -= yuk_bu_araca
                        
                        maliyet = int(p["spot_sabit_maliyet"] + dist * p["spot_km_maliyet"])
                        spot_toplam_maliyet += maliyet
                        metin = f"{g} | Spot {a} | {h} | {yuk_bu_araca} Desi | {maliyet} TL\n"
                        f.write(metin)
                        print(metin.strip())

        genel_toplam = kiralik_sabit_toplam + spot_toplam_maliyet

        ozet = f"""
            {'=' * 80}
            BASE MODEL (Uğramasız & Ertelemesiz) İSTATİSTİKLERİ
            {'=' * 80}
              Kiralık Araç Sabit Maliyeti : {kiralik_sabit_toplam:>15,.0f} TL
              Spot Araç Maliyeti (Direkt) : {spot_toplam_maliyet:>15,.0f} TL
              Kullanılan Kiralik Araç Sayısı: {used_rented_count}
              Kullanılan Spot Araç Sayısı: {used_spot_count}
            {'─' * 80}
              TOPLAM MALİYET              : {genel_toplam:>15,.0f} TL
            {'=' * 80}
            """
        f.write(ozet)
        print(ozet)
    else:
        hata = "❌ Çözüm bulunamadı (INFEASIBLE). Muhtemelen %10 kuralı erteleme olmadan aşılamadı.\n"
        f.write(hata)
        print(hata)