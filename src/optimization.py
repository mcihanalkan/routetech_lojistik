from pathlib import Path
import os
import pandas as pd
from ortools.sat.python import cp_model
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "haversine"))
from haversine import GetDistanceMatrixAsList, GetCenters, is_city_between

# =============================================================================
# 1. MODELİ BAŞLATMA
# =============================================================================
model = cp_model.CpModel()

# main.py üzerinden gelen ortam değişkenleri (varsa kullan, yoksa varsayılan)
ENV_MAX_TIME = float(os.environ.get("ROUTETECH_MAX_TIME_SECONDS", "300"))
ENV_LOG_PROGRESS = os.environ.get("ROUTETECH_LOG_SEARCH_PROGRESS", "1") == "1"

# =============================================================================
# 2. INPUT VERİLERİ
# =============================================================================
centers = GetCenters()

# Talep verilerinin çekimi
payload_csv = Path(__file__).parent.parent / "src" / "predict_model" / "ortools_payload.csv"

if payload_csv.exists():
    df_payload = pd.read_csv(payload_csv)
    hatlar_sirali = []
    talep_verisi = {} # (h,g) -> value
    gecici_gunler = set()

    for _, row in df_payload.iterrows():
        source = row.get('source_tm') or row.get('kaynak_tm') or row.get('source') or row.get('kaynak')
        dest   = row.get('destination_tm') or row.get('varis_tm') or row.get('destination') or row.get('varis')
        
        # source ve dest çekilemediyse 1 ve 2 indekse bak.
        if pd.isna(source) or source is None:
            source = row.iloc[1]
        if pd.isna(dest) or dest is None:
            dest = row.iloc[2]

        date_str = str(row['date']) if 'date' in row else str(row.iloc[0])
        # YZ modelimizin risk tamponlu "Önerilen" talebini alıyoruz, dümdüz q50'yi değil!
        recommended = float(row.get('recommended_demand', row.get('q50', 0)))

        # FIX #7: dict.fromkeys() ile sıra koruyarak tekil hat listesi
        hat = f"{str(source).strip()}-{str(dest).strip()}"
        hatlar_sirali.append(hat)

        tarih_obj = pd.to_datetime(date_str)
        gun_adi = f"{tarih_obj.day:02d}_Mayis"

        gecici_gunler.add(gun_adi)
        talep_verisi[(hat, gun_adi)] = round(recommended) # <--- Zeka buraya entegre oldu!

    # FIX #7: set() yerine dict.fromkeys() — sıra korunur, tekrarlar temizlenir
    hatlar = list(dict.fromkeys(hatlar_sirali))
    gunler = sorted(list(gecici_gunler))

else:
    print(f"⚠️  {payload_csv} bulunamadı! Sabit veriler kullanılıyor.")
    hatlar = [f"{tm1}-{tm2}" for tm1 in centers for tm2 in centers if tm1 != tm2]
    gunler = ["11_Mayis", "12_Mayis", "13_Mayis", "14_Mayis", "15_Mayis", "16_Mayis", "17_Mayis"]
    talep_verisi = {(h, g): 15000 for h in hatlar for g in gunler}

arac_turleri = ["Tir", "Kamyon", "Hafif Kam", "Kamyonet"]

# =============================================================================
# 3. MESAFE MATRİSİ
#    Teknofest Kural #6: Mesafeler kuş uçuşu (Haversine) hesaplanır.
# =============================================================================
distances_2d = GetDistanceMatrixAsList()

print(f"centers: {centers}")
print(f"distances_2d: {distances_2d}")

tm_index = {tm: idx for idx, tm in enumerate(centers)} # Key = tm -> value = index
center_matrix_df = pd.DataFrame(distances_2d, index=centers, columns=centers)

mesafe_verisi = {}
for hat in hatlar:
    # FIX #8: maxsplit=1 — tireli şehir adları güvenle ayrıştırılır
    parcalar = hat.split("-", maxsplit=1)
    if len(parcalar) == 2:
        tm1, tm2 = parcalar
        if tm1 in tm_index and tm2 in tm_index:
            mesafe_verisi[hat] = distances_2d[tm_index[tm1]][tm_index[tm2]]

# =============================================================================
# 4. KİRALIK STOK VE ARAÇ PARAMETRELERİ
#    Teknofest Kural #3: Kiralık araçlar tanımlı hatta zorunlu olarak çalışır.
#    Kiralık araç sayısı bu dict'teki ÜST SINIRI belirler; optimizer 0'a kadar
#    düşürebilir — ama Teknofest kuralına göre bunlar "kullanımda" sayılır ve
#    maliyetleri her halükarda ödenir (sabit_kira). Bu nedenle kiralık araç
#    maliyetleri amaç fonksiyonuna SABİT olarak eklenir, değişken olarak değil.
# =============================================================================
# CSV dosyasından kiralık stok verilerini oku
rented_stoks_csv = Path(__file__).parent.parent / "data" / "static_datas" / "rented_stoks.csv"

kiralik_stok_gunluk = {}

if rented_stoks_csv.exists():
    df_rented = pd.read_csv(rented_stoks_csv)
    for _, row in df_rented.iterrows():
        route = row['route']
        vehicle_type = row['vehicle_type']
        quantity = int(row['quantity'])
        kiralik_stok_gunluk[(route, vehicle_type)] = quantity
    print(f"✓ Kiralık stok verileri yüklendi: {rented_stoks_csv}")
else:
    print(f"⚠️ HATA.  {rented_stoks_csv} bulunamadı! Kiralık stok boş kalacak.")
    kiralik_stok_gunluk = {}


# CSV dosyasından araç parametrelerini oku
car_params_csv = Path(__file__).parent.parent / "data" / "static_datas" / "car_parameters.csv"

arac_parametreleri = {}

if car_params_csv.exists():
    df_car_params = pd.read_csv(car_params_csv)
    for _, row in df_car_params.iterrows():
        vehicle_type = row['vehicle_type']
        arac_parametreleri[vehicle_type] = {
            "sabit_kira": int(row['sabit_kira']),
            "kiralik_km_maliyet": int(row['kiralik_km_maliyet']),
            "spot_sabit_maliyet": int(row['spot_sabit_maliyet']),
            "spot_km_maliyet": int(row['spot_km_maliyet']),
            "kapasite_desi": int(row['kapasite_desi']),
        }
    print(f"✓ Araç parametreleri yüklendi: {car_params_csv}")
else:
    print(f"⚠️  HATA. {car_params_csv} bulunamadı! Araç parametreleri boş kalacak.")
    arac_parametreleri = {}

# Teknofest Kural #1: SLA ceza ağırlığı — spot araç maliyetiyle rekabetçi tutuldu.
# En ucuz spot araç (Kamyonet, 5600 desi): ~4750 TL sabit.
# Desi başı ~0.85 TL. Erteleme caydırıcı ama imkânsız kılmıyor.
SLA_GECIKME_CEZA_TL_PER_DESI = 4

# =============================================================================
# 5. ÜST SINIR HESAPLARI
# =============================================================================
# FIX #6: max_talep_hatta üst sınırı — teorik maksimum toplam talep
max_talep_hatta = sum(talep_verisi.values()) if talep_verisi else 0
max_spot = 500 # spot araç sayısı sınırsız. Bu yüzden yüksek bir değer olan 500 girildi

# =============================================================================
# 6. UĞRAMA ROTALARINI ÖN-İŞLEME
#    Teknofest Kural #4: Uğrama serbesttir (Multi-stop routing).
#    Konsolidasyon (farklı çıkış noktalarından merkeze toplama) yasaktır.
#    FIX #9: Uğrama rotaları bir kez önbelleklenir.
# =============================================================================
ugrama_rotalari = []

for h in hatlar:
    parcalar = h.split("-", maxsplit=1)
    if len(parcalar) != 2:
        continue
    tm1, tm2 = parcalar

    for c in centers:
        # C başlangıç veya bitiş noktası olamaz
        if c == tm1 or c == tm2:
            continue

        # A->C ve C->B rotası var mı ve C A ile B'nin arasında mı?
        if (f"{tm1}-{c}" in hatlar and f"{c}-{tm2}" in hatlar and
            is_city_between(source=tm1, destination=tm2, candidate=c, center_matrix=center_matrix_df)):
            ugrama_rotalari.append((tm1, c, tm2))


# Tekrar kontrolleri ortadan kaldır (eğer varsa)
ugrama_rotalari = list(dict.fromkeys(ugrama_rotalari))

print(f"ℹ️  Toplam uğrama rotası sayısı: {len(ugrama_rotalari)}")



# =============================================================================
# 7. KARAR DEĞİŞKENLERİ
# =============================================================================
# Teknofest Kural #3 + FIX #1:
#   Kiralık araçlar VALUE olarak modele giriyor (değişken değil).
#   Kapasiteleri yük kısıtlarına dahil edilir, ama spot_y optimizer'ın kararıdır.
#
# FIX #2: Yük değişkenleri kaldırıldı. Kapasite doğrudan araç sayısı × kapasite
#          üzerinden hesaplanıyor. Böylece yük-kapasite kopukluğu ortadan kalktı.
kiralik_x = {} # kiralık araç sayısı. Direkt gider. UĞRAMA YAPMAZ!
spot_y          = {}   # spot araç sayısı (karar değişkeni)
ertelenen_talep = {}   # güne ait karşılanamayan talep
biriken_talep   = {}   # önceki günden devreden + bugünkü talep
spot_yuk_by_type = {}        # spot araç türü başına taşınan yük (mevcut değişkenlerin referansı)
kiralik_tasinan_yuk_dict = {}  # kiralık toplam taşınan yük (mevcut değişkenlerin referansı)

# Modele karar değişkenlerini tanıtıyoruz.
for h in hatlar:
    for g in gunler:
        for a in arac_turleri:
            spot_y[(h, g, a)] = model.NewIntVar(0, max_spot, f'spot_{h}_{g}_{a}')
            kiralik_x[(h, g, a)] = model.NewIntVar(0, kiralik_stok_gunluk.get((h,a),0), f'kiralik_{h}_{g}_{a}')
        ertelenen_talep[(h, g)] = model.NewIntVar(0, max_talep_hatta, f'ert_{h}_{g}')
        # FIX #6: Birikmiş talep üst sınırı — tüm günlerin toplamı
        biriken_talep[(h, g)]   = model.NewIntVar(0, max_talep_hatta, f'bir_{h}_{g}')

# Uğrama araç değişkenleri
ugrama_spot_y = {}   # sadece spot araçlar için
ugrama_kiralik_y = {}  # kiralık araçlar için

ugrama_yuk_ac = {}
ugrama_yuk_cb = {}
ugrama_yuk_ab = {}

for (a, c, b) in ugrama_rotalari:
    for g in gunler:
        for arac in arac_turleri:
            ugrama_hat = f"{a}-{b}"
            max_kir = kiralik_stok_gunluk.get((ugrama_hat, arac), 0)
            ugrama_kiralik_y[(a, c, b, g, arac)] = model.NewIntVar(0, max_kir, f'uk_{a}_{c}_{b}_{g}_{arac}')
            ugrama_spot_y[(a, c, b, g, arac)] = model.NewIntVar(0, max_spot, f'uy_{a}_{c}_{b}_{g}_{arac}')
        # A'dan C'ye gidecek ve indirilecek yük
        ugrama_yuk_ac[(a, c, b, g)] = model.NewIntVar(0, max_talep_hatta, f'uac_{a}_{c}_{b}_{g}')
        # C'den yüklenip B'de indirilecek yük
        ugrama_yuk_cb[(a, c, b, g)] = model.NewIntVar(0, max_talep_hatta, f'ucb_{a}_{c}_{b}_{g}')
        # A'dan yüklenip C'de duruş yapıp B'de indirilecek yük.
        ugrama_yuk_ab[(a, c, b, g)] = model.NewIntVar(0, max_talep_hatta, f'uab_{a}_{c}_{b}_{g}')

# =============================================================================
# 8. KISITLAR
# =============================================================================

for h in hatlar:
    parcalar = h.split("-", maxsplit=1)
    if len(parcalar) != 2:
        continue
    tm1, tm2 = parcalar

    for g in gunler:

        # --- Kiralık araç kapasitesi (sabit) ---
        kiralik_toplam_kap = sum(
            kiralik_stok_gunluk.get((h, a), 0) * arac_parametreleri[a]["kapasite_desi"]
            for a in arac_turleri
        )

        # --- Spot araç kapasitesi ---
        spot_kap_ifadesi = cp_model.LinearExpr.Sum([
            spot_y[(h, g, a)] * arac_parametreleri[a]["kapasite_desi"]
            for a in arac_turleri
        ])

        # FIX #1: Spot araca tahsis edilen "Net Yük" izolasyonu
        spot_tasinan_yuk = model.NewIntVar(0, 500000, f'spot_net_yuk_{h}_{g}')
        
        # Fiziksel sınır: Taşınan yük, kapasiteyi aşamaz
        model.Add(spot_tasinan_yuk <= spot_kap_ifadesi)

        # ---------------------------------------------------------------
        # KISIT A — Teknofest Kural #1: %10 Minimum Doluluk (Sadece Spot)
        # ---------------------------------------------------------------
        spot_tasinan_yuk_listesi = []

        for a in arac_turleri:
            kap = arac_parametreleri[a]["kapasite_desi"]

            # Her araç türü (a) için taşınan spot yükü ayrı izole ediyoruz
            tasinan_yuk_a = model.NewIntVar(0, max_spot * kap, f'spot_net_yuk_{h}_{g}_{a}')
            spot_tasinan_yuk_listesi.append(tasinan_yuk_a)
            spot_yuk_by_type[(h, g, a)] = tasinan_yuk_a

            # Fiziksel sınır: Araç türünün taşıdığı yük, açılan kapasiteyi aşamaz
            model.Add(tasinan_yuk_a <= spot_y[(h, g, a)] * kap)

            # TEKNOFEST KURAL 1: %10 Minimum Doluluk (Her araç türü için ayrı denetim)
            if g != gunler[-1]:
                model.Add(spot_y[(h, g, a)] * kap <= tasinan_yuk_a * 10)    

        # ---------------------------------------------------------------
        # KISIT B — Talep Dengesi
        # ---------------------------------------------------------------
        bugun_talep = talep_verisi.get((h, g), 0)
        idx = gunler.index(g)
        if idx == 0:
            model.Add(biriken_talep[(h, g)] == bugun_talep)
        else:
            onceki_gun = gunler[idx - 1]
            model.Add(
                biriken_talep[(h, g)] ==
                ertelenen_talep[(h, onceki_gun)] + bugun_talep
            )

        # ---------------------------------------------------------------
        # KISIT C — Yük Dağıtım Dengesi (DÜZELTİLMİŞ)
        # ---------------------------------------------------------------
        ugrama_katkilari = []

        # DURUM 1: Bu hat (tm1-tm2), uğramanın İLK BACAĞI ise (tm1 -> tm2 -> X)
        for x in centers:
            if (tm1, tm2, x) in ugrama_rotalari:
                ugrama_katkilari.append(ugrama_yuk_ac[(tm1, tm2, x, g)])

        # DURUM 2: Bu hat (tm1-tm2), uğramanın İKİNCİ BACAĞI ise (W -> tm1 -> tm2)
        for w in centers:
            if (w, tm1, tm2) in ugrama_rotalari:
                ugrama_katkilari.append(ugrama_yuk_cb[(w, tm1, tm2, g)])

        # DURUM 3: Bu hat (tm1-tm2), uğramanın ANA ROTASI ise (tm1 -> C -> tm2)
        for c in centers:
            if (tm1, c, tm2) in ugrama_rotalari:
                ugrama_katkilari.append(ugrama_yuk_ab[(tm1, c, tm2, g)])

        kiralik_aktif_kapasite = cp_model.LinearExpr.Sum([
            kiralik_x[(h, g, a)] * arac_parametreleri[a]["kapasite_desi"]
            for a in arac_turleri
        ])

        kiralik_tasinan_yuk = model.NewIntVar(0, 500000, f'kiralik_net_yuk_{h}_{g}')
        model.Add(kiralik_tasinan_yuk <= kiralik_aktif_kapasite)
        kiralik_tasinan_yuk_dict[(h, g)] = kiralik_tasinan_yuk

        # FIX #2: tasinan_toplam içine SABİT kapasiteyi değil, DEĞİŞKEN olan net kiralık yükünü koyuyoruz.
        tasinan_toplam = cp_model.LinearExpr.Sum(
            [kiralik_tasinan_yuk] + spot_tasinan_yuk_listesi + ugrama_katkilari
        )

        model.Add(
            biriken_talep[(h, g)] == tasinan_toplam + ertelenen_talep[(h, g)]
        )
        
        model.Add(
            ertelenen_talep[(h, g)] <= biriken_talep[(h, g)]
        )

# ---------------------------------------------------------------
# KISIT D — Uğrama Araç Kapasite Kısıtları 
# ---------------------------------------------------------------
for (a, c, b) in ugrama_rotalari:
    for g in gunler:
        u_kap = cp_model.LinearExpr.Sum([
            ugrama_spot_y[(a, c, b, g, arac)] * arac_parametreleri[arac]["kapasite_desi"] +
            ugrama_kiralik_y[(a, c, b, g, arac)] * arac_parametreleri[arac]["kapasite_desi"]
            for arac in arac_turleri
        ])

        # A→C segmentinde: ac yükü + ab yükü (ab yükü tüm yol boyunca araçta)
        model.Add(ugrama_yuk_ac[(a, c, b, g)] + ugrama_yuk_ab[(a, c, b, g)] <= u_kap)

        # C→B segmentinde: cb yükü + ab yükü
        model.Add(ugrama_yuk_cb[(a, c, b, g)] + ugrama_yuk_ab[(a, c, b, g)] <= u_kap)

        # Uğrama yük-araç bağlantısı (tüm günler için geçerli)
        u_tasinan_net_listesi = []

        for arac in arac_turleri:
            kap = arac_parametreleri[arac]["kapasite_desi"]

            # --- Spot ugrama ---
            u_spot_net_a = model.NewIntVar(0, max_spot * kap * 2, f'u_spot_net_{a}_{c}_{b}_{g}_{arac}') # Toplam spot yükü
            u_tasinan_net_listesi.append(u_spot_net_a) 
            model.Add(u_spot_net_a <= ugrama_spot_y[(a, c, b, g, arac)] * kap * 2) # 

            # Teknofest Kural #1: %10 doluluk kuralı (son gün hariç)
            if g != gunler[-1]:
                model.Add(ugrama_spot_y[(a, c, b, g, arac)] * kap <= u_spot_net_a * 10) 
                # Uğrama spot araç sayısı * kap * 0.10 <= toplam spot yükü. Bunu araç araç ayrı bir şekilde yapmamız gerekir.
                # kap -> 5000, toplam spot yükü = 22000. 5000*5 * 0.1 <= 22000 = 2500 <= 22000 sağlandı fakat.
                # Araç 1: 5000, araç 2: 5000, araç 3: 5000, araç 4: 5000
            # --- Kiralık ugrama: %10 uygulanmaz (zorunlu kalkış, Teknofest Kural #3) ---
            ugrama_hat = f"{a}-{b}"
            max_kir = kiralik_stok_gunluk.get((ugrama_hat, arac), 0)
            if max_kir > 0:
                u_kir_net_a = model.NewIntVar(0, max_kir * kap * 2, f'u_kir_net_{a}_{c}_{b}_{g}_{arac}')
                u_tasinan_net_listesi.append(u_kir_net_a)
                model.Add(u_kir_net_a <= ugrama_kiralik_y[(a, c, b, g, arac)] * kap * 2)

        toplam_ugrama_yuk = cp_model.LinearExpr.Sum([
            ugrama_yuk_ac[(a, c, b, g)],
            ugrama_yuk_cb[(a, c, b, g)],
            ugrama_yuk_ab[(a, c, b, g)],
        ])
        model.Add(toplam_ugrama_yuk == cp_model.LinearExpr.Sum(u_tasinan_net_listesi))
# ---------------------------------------------------------------
# KISIT E — Teknofest: Son Gün Erteleme Yasağı
# ---------------------------------------------------------------
for h in hatlar:
    model.Add(ertelenen_talep[(h, gunler[-1])] == 0)

# =============================================================================
# KISIT F — Kiralık Araç Stok Kontrolü (ZORUNLU KALKIŞ)
# =============================================================================
for h in hatlar:
    for g in gunler:
        for a in arac_turleri:
            stok = kiralik_stok_gunluk.get((h, a), 0)
            
            direct_use = kiralik_x[(h, g, a)]
            ugrama_use = cp_model.LinearExpr.Sum([
                ugrama_kiralik_y[(tm1, tm2, tm3, g, a)]
                for (tm1, tm2, tm3) in ugrama_rotalari
                if f"{tm1}-{tm3}" == h
            ])
            
            # DÜZELTME: <= yerine == kullanıyoruz. Araçlar boş olsa bile KALKACAK!
            model.Add(direct_use + ugrama_use == stok)


# 9. AMAÇ FONKSİYONU
#
# Teknofest Kural #3 + FIX #1:
#   Kiralık araç maliyetleri SABİT (değişken değil) — her zaman ödenir.
#   Optimizer sadece spot araç sayısını ve ertelemeyi minimize eder.
#
# Teknofest Kural #2: Dönüş maliyeti hesaplanmaz (tek yön).
# Teknofest Kural #6: Mesafe = kuş uçuşu (Haversine).
# Teknofest Kural #5: Toplam maliyet raporlanır.
# =============================================================================
maliyet_kalemleri    = []
kiralik_sabit_toplam = 0  # Sabit kiralık maliyet (amaç fonksiyonuna girmez ama raporlanır)

for h in hatlar:
    dist = mesafe_verisi.get(h, 0)
    parcalar = h.split("-", maxsplit=1)
    if len(parcalar) != 2:
        continue

    for g in gunler:
        # # --- Kiralık araç maliyeti (optimizer tarafından kontrol edilir) ---
        for a in arac_turleri:
            adet = kiralik_stok_gunluk.get((h, a), 0)
            if adet > 0:
                p = arac_parametreleri[a]
                gun_maliyet = adet * int(p["sabit_kira"] + dist * p["kiralik_km_maliyet"])
                kiralik_sabit_toplam += gun_maliyet

        # --- Spot araç değişken maliyeti ---
        for a in arac_turleri:
            p = arac_parametreleri[a]
            spot_maliyet_katsayi = int(p["spot_sabit_maliyet"] + dist * p["spot_km_maliyet"])
            maliyet_kalemleri.append(spot_y[(h, g, a)] * spot_maliyet_katsayi)

        # --- SLA Gecikme Cezası ---
        # FIX #5: Ceza ağırlığı spot araçlarla rekabetçi seviyede
        ceza_katsayi = int(SLA_GECIKME_CEZA_TL_PER_DESI)
        maliyet_kalemleri.append(ertelenen_talep[(h, g)] * ceza_katsayi)

# --- Uğrama spot araç maliyetleri ---
# DÜZELTME: Kiralık araçlar için SADECE mesafe-tabanlı değişken maliyet ekle
# Sabit kira zaten amaç fonksiyonunun dışında ödeniyor

for (a, c, b) in ugrama_rotalari:
    dist_ac = distances_2d[tm_index[a]][tm_index[c]]
    dist_cb = distances_2d[tm_index[c]][tm_index[b]]
    dist_toplam = dist_ac + dist_cb

    for g in gunler:
        for arac in arac_turleri:
            p = arac_parametreleri[arac]
            ugrama_spot_maliyet = int(p["spot_sabit_maliyet"] + dist_toplam * p["spot_km_maliyet"])
            # A'dan B'ye direkt mesafe (Kaçınılmaz yol)
            dist_direkt_ab = distances_2d[tm_index[a]][tm_index[b]]

            # Sapmadan kaynaklanan EKSTRA uzama
            ekstra_mesafe = dist_toplam - dist_direkt_ab

            # Sadece uzayan yolun maliyeti faturaya yazılır!
            ugrama_kiralik_km_maliyet = int(ekstra_mesafe * p["kiralik_km_maliyet"])

            maliyet_kalemleri.append(ugrama_spot_y[(a, c, b, g, arac)] * ugrama_spot_maliyet)
            # ✅ Kiralık uğrama maliyeti: sabit değil, sadece KM maliyeti
            maliyet_kalemleri.append(ugrama_kiralik_y[(a, c, b, g, arac)] * ugrama_kiralik_km_maliyet)


model.Minimize(cp_model.LinearExpr.Sum(maliyet_kalemleri))

# =============================================================================
# 10. ÇÖZÜCÜ PARAMETRELERI
# =============================================================================
solver = cp_model.CpSolver()
max_time_to_solve = ENV_MAX_TIME
solver.parameters.max_time_in_seconds  = max_time_to_solve
solver.parameters.num_search_workers   = 4
solver.parameters.log_search_progress  = ENV_LOG_PROGRESS

status = solver.Solve(model)

# =============================================================================
# 11. SONUÇ YAZIMI VE CSV ÇIKTI
# =============================================================================
output_file = Path(__file__).parent.parent / "results" / "optimization_results.txt"
csv_output_file = Path(__file__).parent.parent / "results" / "optimization_results.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)

# CSV için veri topla
csv_records = []

with open(output_file, "w", encoding="utf-8") as f:

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        durum = "OPTIMAL (Kusursuz)" if status == cp_model.OPTIMAL else f"UYGUN ({max_time_to_solve}s)"
        baslik = f"{'=' * 80}\nOPTİMİZASYON BAŞARILI — {durum}\n{'=' * 80}\n\n"
        f.write(baslik)
        print(baslik)

        toplam_ertelenen_desi = 0
        spot_toplam_maliyet = 0
        ugrama_toplam_maliyet = 0
        kiralik_ugrama_ekstra_km_toplam = 0

        # --- Direkt Rota Sonuçları ---
        print("=" * 80)
        print("DİREKT ROTALAR")
        print("=" * 80)
        f.write("\n" + "=" * 80 + "\nDİREKT ROTALAR\n" + "=" * 80 + "\n")
        f.write("Tarih | Araç Türü | Hat | TAŞINAN NET DESİ | Maliyet\n")
        f.write("-" * 80 + "\n")
        print("Tarih | Araç Türü | Hat | TAŞINAN NET DESİ | Maliyet")
        print("-" * 80)

        u_spot_count = 0
        d_spot_count = 0
        u_rented_count = 0
        d_rented_count = 0

        for h in hatlar:
            dist = mesafe_verisi.get(h, 0)
            parcalar = h.split("-", maxsplit=1)
            if len(parcalar) != 2:
                continue
            tm1, tm2 = parcalar

            for g in gunler:
                # Kiralık yükü kapasiteye orantılı dağıt (önce hesapla, sonra yaz)
                kiralik_yuk_toplam = solver.Value(kiralik_tasinan_yuk_dict[(h, g)])
                toplam_kiralik_kap = sum(
                    solver.Value(kiralik_x[(h, g, ar)]) * arac_parametreleri[ar]["kapasite_desi"]
                    for ar in arac_turleri
                )
                kiralik_tip_yukleri = {}
                kiralik_dagitilan = 0
                son_tip = None
                for ar in arac_turleri:
                    ka = solver.Value(kiralik_x[(h, g, ar)])
                    if ka > 0:
                        tk = ka * arac_parametreleri[ar]["kapasite_desi"]
                        ty = kiralik_yuk_toplam * tk // toplam_kiralik_kap if toplam_kiralik_kap > 0 else 0
                        ty = min(ty, tk)
                        kiralik_tip_yukleri[ar] = ty
                        kiralik_dagitilan += ty
                        son_tip = ar
                if son_tip is not None and kiralik_dagitilan != kiralik_yuk_toplam:
                    fark = kiralik_yuk_toplam - kiralik_dagitilan
                    son_kap = solver.Value(kiralik_x[(h, g, son_tip)]) * arac_parametreleri[son_tip]["kapasite_desi"]
                    kiralik_tip_yukleri[son_tip] = min(kiralik_tip_yukleri[son_tip] + fark, son_kap)

                for a in arac_turleri:
                    k_adet = solver.Value(kiralik_x[(h, g, a)])
                    s_adet = solver.Value(spot_y[(h, g, a)])
                    d_rented_count += k_adet
                    d_spot_count += s_adet
                    p = arac_parametreleri[a]

                    # Kiralık araçlar
                    if k_adet > 0:
                        tip_yuk = kiralik_tip_yukleri.get(a, 0)
                        yuk_per_arac = tip_yuk // k_adet
                        yuk_kalan = tip_yuk - yuk_per_arac * k_adet
                        araç_maliyet = int(p["sabit_kira"] + dist * p["kiralik_km_maliyet"])
                        for i in range(k_adet):
                            yuk = yuk_per_arac + (1 if i < yuk_kalan else 0)
                            metin = f"{g} | Kiralık {a} | {h} | {yuk} | {araç_maliyet}\n"
                            f.write(metin)
                            print(metin.strip())

                            csv_records.append({
                                "Tarih": g,
                                "Araç_Tipi": f"Kiralık {a}",
                                "Çıkış_TM": tm1,
                                "Varış_TM": tm2,
                                "Araç_Sayısı": 1,
                                "Teslim_Edilen_Desi": yuk,
                                "Maliyet_TL": araç_maliyet,
                                "Rota_Tipi": "Direkt"
                            })

                    # Spot araçlar — per-tip yük zaten solver'dan geliyor
                    if s_adet > 0:
                        spot_yuk = solver.Value(spot_yuk_by_type[(h, g, a)])
                        spot_yuk_per_arac = spot_yuk // s_adet
                        spot_yuk_kalan = spot_yuk - spot_yuk_per_arac * s_adet
                        spot_araç_maliyet = int(p["spot_sabit_maliyet"] + dist * p["spot_km_maliyet"])
                        for i in range(s_adet):
                            yuk = spot_yuk_per_arac + (1 if i < spot_yuk_kalan else 0)
                            metin = f"{g} | Spot {a} | {h} | {yuk} | {spot_araç_maliyet}\n"
                            f.write(metin)
                            print(metin.strip())

                            spot_toplam_maliyet += spot_araç_maliyet
                            csv_records.append({
                                "Tarih": g,
                                "Araç_Tipi": f"Spot {a}",
                                "Çıkış_TM": tm1,
                                "Varış_TM": tm2,
                                "Araç_Sayısı": 1,
                                "Teslim_Edilen_Desi": yuk,
                                "Maliyet_TL": spot_araç_maliyet,
                                "Rota_Tipi": "Direkt"
                            })

                ert = solver.Value(ertelenen_talep[(h, g)])
                if ert > 0:
                    toplam_ertelenen_desi += ert
                    ert_metin = f"{g} | ERTELEME | {h} | {ert} | 0\n"
                    f.write(ert_metin)
                    print(ert_metin.strip())

        # --- Uğrama Rota Sonuçları ---
        print("=" * 80)
        print("UĞRAMA ROTALAR")
        print("=" * 80)
        f.write("\n" + "=" * 80 + "\nUĞRAMA ROTALAR\n" + "=" * 80 + "\n")
        f.write("Tarih | Araç Türü | Hat | TAŞINAN NET DESİ | Maliyet\n")
        f.write("-" * 80 + "\n")
        print("Tarih | Araç Türü | Hat | TAŞINAN NET DESİ | Maliyet")
        print("-" * 80)

        for (a, c, b) in ugrama_rotalari:
            dist_ac = distances_2d[tm_index[a]][tm_index[c]]
            dist_cb = distances_2d[tm_index[c]][tm_index[b]]
            dist_toplam = dist_ac + dist_cb
            ugrama_hat = f"{a}-{b}"

            for g in gunler:
                ugrama_toplam_yuk_gun = (
                    solver.Value(ugrama_yuk_ac[(a, c, b, g)]) +
                    solver.Value(ugrama_yuk_cb[(a, c, b, g)]) +
                    solver.Value(ugrama_yuk_ab[(a, c, b, g)])
                )
                ugrama_toplam_arac_gun = sum(
                    solver.Value(ugrama_kiralik_y[(a, c, b, g, ar)]) +
                    solver.Value(ugrama_spot_y[(a, c, b, g, ar)])
                    for ar in arac_turleri
                )
                # Uğrama yükü kapasiteye orantılı dağıt
                ugrama_toplam_kap = sum(
                    (solver.Value(ugrama_kiralik_y[(a, c, b, g, ar)]) + solver.Value(ugrama_spot_y[(a, c, b, g, ar)])) * arac_parametreleri[ar]["kapasite_desi"]
                    for ar in arac_turleri
                )
                ugrama_tip_yukleri = {}
                ugrama_dagitilan = 0
                ugrama_son_tip = None
                for ar in arac_turleri:
                    u_toplam_adet = solver.Value(ugrama_kiralik_y[(a, c, b, g, ar)]) + solver.Value(ugrama_spot_y[(a, c, b, g, ar)])
                    if u_toplam_adet > 0:
                        tk = u_toplam_adet * arac_parametreleri[ar]["kapasite_desi"]
                        ty = ugrama_toplam_yuk_gun * tk // ugrama_toplam_kap if ugrama_toplam_kap > 0 else 0
                        ty = min(ty, tk)
                        ugrama_tip_yukleri[ar] = ty
                        ugrama_dagitilan += ty
                        ugrama_son_tip = ar
                if ugrama_son_tip is not None and ugrama_dagitilan != ugrama_toplam_yuk_gun:
                    fark = ugrama_toplam_yuk_gun - ugrama_dagitilan
                    u_son_adet = solver.Value(ugrama_kiralik_y[(a, c, b, g, ugrama_son_tip)]) + solver.Value(ugrama_spot_y[(a, c, b, g, ugrama_son_tip)])
                    son_kap = u_son_adet * arac_parametreleri[ugrama_son_tip]["kapasite_desi"]
                    ugrama_tip_yukleri[ugrama_son_tip] = min(ugrama_tip_yukleri[ugrama_son_tip] + fark, son_kap)

                for arac in arac_turleri:
                    p = arac_parametreleri[arac]
                    u_k_adet = solver.Value(ugrama_kiralik_y[(a, c, b, g, arac)])
                    u_s_adet = solver.Value(ugrama_spot_y[(a, c, b, g, arac)])
                    u_toplam = u_k_adet + u_s_adet
                    u_rented_count += u_k_adet
                    u_spot_count += u_s_adet
                    tip_yuk = ugrama_tip_yukleri.get(arac, 0)

                    if u_toplam > 0:
                        yuk_per_arac = tip_yuk // u_toplam
                        yuk_kalan = tip_yuk - yuk_per_arac * u_toplam
                        arac_sira = 0

                    # Kiralık uğrama
                    if u_k_adet > 0:
                        araç_maliyet = int(p["sabit_kira"] + dist_toplam * p["kiralik_km_maliyet"])
                        dist_direkt_ab = distances_2d[tm_index[a]][tm_index[b]]
                        ekstra_km_maliyet = int((dist_toplam - dist_direkt_ab) * p["kiralik_km_maliyet"])
                        for i in range(u_k_adet):
                            kiralik_ugrama_ekstra_km_toplam += ekstra_km_maliyet
                            yuk = yuk_per_arac + (1 if arac_sira < yuk_kalan else 0)
                            arac_sira += 1
                            metin = f"{g} | Kiralık {arac} | {a}→{c}→{b} | {yuk} | {araç_maliyet}\n"
                            f.write(metin)
                            print(metin.strip())

                            csv_records.append({
                                "Tarih": g,
                                "Araç_Tipi": f"Kiralık {arac}",
                                "Çıkış_TM": a,
                                "Varış_TM": b,
                                "Araç_Sayısı": 1,
                                "Teslim_Edilen_Desi": yuk,
                                "Maliyet_TL": araç_maliyet,
                                "Rota_Tipi": f"Uğrama ({c})"
                            })

                    # Spot uğrama
                    if u_s_adet > 0:
                        spot_araç_maliyet = int(p["spot_sabit_maliyet"] + dist_toplam * p["spot_km_maliyet"])
                        for i in range(u_s_adet):
                            yuk = yuk_per_arac + (1 if arac_sira < yuk_kalan else 0)
                            arac_sira += 1
                            metin = f"{g} | Spot {arac} | {a}→{c}→{b} | {yuk} | {spot_araç_maliyet}\n"
                            f.write(metin)
                            print(metin.strip())

                            ugrama_toplam_maliyet += spot_araç_maliyet
                            csv_records.append({
                                "Tarih": g,
                                "Araç_Tipi": f"Spot {arac}",
                                "Çıkış_TM": a,
                                "Varış_TM": b,
                                "Araç_Sayısı": 1,
                                "Teslim_Edilen_Desi": yuk,
                                "Maliyet_TL": spot_araç_maliyet,
                                "Rota_Tipi": f"Uğrama ({c})"
                            })

        # --- Uğrama Yük Dağılımı: Gerçek teslimatları alt-rotalara ata ---
        for (a, c, b) in ugrama_rotalari:
            for g in gunler:
                yuk_ac = solver.Value(ugrama_yuk_ac[(a, c, b, g)])
                yuk_cb = solver.Value(ugrama_yuk_cb[(a, c, b, g)])

                if yuk_ac > 0:
                    csv_records.append({
                        "Tarih": g,
                        "Araç_Tipi": "Uğrama Teslimat",
                        "Çıkış_TM": a,
                        "Varış_TM": c,
                        "Araç_Sayısı": 0,
                        "Teslim_Edilen_Desi": yuk_ac,
                        "Maliyet_TL": 0,
                        "Rota_Tipi": f"Uğrama Katkısı ({a}→{c}→{b})"
                    })
                if yuk_cb > 0:
                    csv_records.append({
                        "Tarih": g,
                        "Araç_Tipi": "Uğrama Teslimat",
                        "Çıkış_TM": c,
                        "Varış_TM": b,
                        "Araç_Sayısı": 0,
                        "Teslim_Edilen_Desi": yuk_cb,
                        "Maliyet_TL": 0,
                        "Rota_Tipi": f"Uğrama Katkısı ({a}→{c}→{b})"
                    })

        # --- Teknofest Kural #5: Toplam Maliyet Özeti ---
        sla_ceza_toplam = int(toplam_ertelenen_desi * SLA_GECIKME_CEZA_TL_PER_DESI)
        degisken_toplam = spot_toplam_maliyet + ugrama_toplam_maliyet + kiralik_ugrama_ekstra_km_toplam + sla_ceza_toplam
        genel_toplam = kiralik_sabit_toplam + degisken_toplam

        ozet = f"""
            {'=' * 80}
            ÖZET İSTATİSTİKLER (Teknofest Kural #5 — Toplam Maliyet)
            {'=' * 80}
              Kiralık Araç Sabit Maliyeti : {kiralik_sabit_toplam:>15,.0f} TL  (her zaman ödenir)
              Kiralık Uğrama Ekstra KM    : {kiralik_ugrama_ekstra_km_toplam:>15,.0f} TL
              Spot Araç Maliyeti (Direkt) : {spot_toplam_maliyet:>15,.0f} TL
              Spot Araç Maliyeti (Uğrama) : {ugrama_toplam_maliyet:>15,.0f} TL
              SLA Gecikme Cezası          : {sla_ceza_toplam:>15,.0f} TL
            {'─' * 80}
              TOPLAM MALİYET              : {genel_toplam:>15,.0f} TL
            {'=' * 80}
              Toplam Ertelenen Yük        : {toplam_ertelenen_desi:>15,.0f} desi
              Çözücü Süre                 : {solver.WallTime():>15.2f} sn
              Objective Value (optimizer) : {solver.ObjectiveValue():>15,.0f}
              TOPLAM ARAÇ SAYISI          : {u_rented_count + d_rented_count + u_spot_count + d_spot_count}
              Direkt Kiralik Araç Sayısı  : {d_rented_count}
              Uğrama Kiralik Araç Sayısı  : {u_rented_count} 
              Direkt Spot Araç Sayısı  : {d_spot_count} 
              Uğrama Spot Araç Sayısı  : {u_spot_count} 
 
            {'=' * 80}
            """
        f.write(ozet)
        print(ozet)

    else:
        hata = (
            "❌ Çözüm bulunamadı! Lütfen kısıtları kontrol edin.\n"
            f"   Durum kodu: {status}\n"
        )
        f.write(hata)
        print(hata)

# CSV dosyasına yaz
if csv_records:
    df_csv = pd.DataFrame(csv_records)
    df_csv.to_csv(csv_output_file, index=False, encoding="utf-8")
    print(f"\n✅ CSV çıktısı kaydedildi: {csv_output_file}")
else:
    print("\n⚠️  CSV çıktısı için veri yok!")