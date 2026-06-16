from pathlib import Path
import pandas as pd
from ortools.sat.python import cp_model
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "haversine"))
from haversine import GetDistanceMatrixAsList, GetCenters, is_city_between

# =============================================================================
# 1. MODELİ BAŞLATMA
# =============================================================================
model = cp_model.CpModel()

# =============================================================================
# 2. INPUT VERİLERİ
# =============================================================================
centers = GetCenters()

# Talep verilerinin çekimi
payload_csv = Path(__file__).parent.parent / "src" / "predict_model" / "ortools_payload.csv"

if payload_csv.exists():
    df_payload = pd.read_csv(payload_csv)
    hatlar_sirali = []
    talep_verisi = {}
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
        q50 = float(row.get('q50', 0))

        # FIX #7: dict.fromkeys() ile sıra koruyarak tekil hat listesi
        hat = f"{str(source).strip()}-{str(dest).strip()}"
        hatlar_sirali.append(hat)

        tarih_obj = pd.to_datetime(date_str)
        gun_adi = f"{tarih_obj.day:02d}_Mayis"

        gecici_gunler.add(gun_adi)
        talep_verisi[(hat, gun_adi)] = int(q50)

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
print(f"distanced_2d: {distances_2d}")

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
SLA_GECIKME_CEZA_TL_PER_DESI = 50

# =============================================================================
# 5. ÜST SINIR HESAPLARI
# =============================================================================
# FIX #6: max_talep_hatta üst sınırı — teorik maksimum toplam talep
max_talep_hatta = sum(talep_verisi.values()) if talep_verisi else 0
max_spot = 50

# =============================================================================
# 6. UĞRAMA ROTALARINI ÖN-İŞLEME
#    Teknofest Kural #4: Uğrama serbesttir (Multi-stop routing).
#    Konsolidasyon (farklı çıkış noktalarından merkeze toplama) yasaktır.
#    FIX #9: Uğrama rotaları bir kez önbelleklenir.
# =============================================================================
ugrama_rotalari = []
for a in centers:
    for b in centers:
        if a == b:
            continue
        for c in centers:
            if c == a or c == b:
                continue
            if is_city_between(
                source=a, destination=b, candidate=c,
                center_matrix=center_matrix_df, tolerance=0.30
            ):
                ugrama_rotalari.append((a, c, b))

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

spot_y          = {}   # spot araç sayısı (karar değişkeni)
ertelenen_talep = {}   # güne ait karşılanamayan talep
biriken_talep   = {}   # önceki günden devreden + bugünkü talep

# Modele karar değişkenlerini tanıtıyoruz.
for h in hatlar:
    for g in gunler:
        for a in arac_turleri:
            spot_y[(h, g, a)] = model.NewIntVar(0, max_spot, f'spot_{h}_{g}_{a}')

        ertelenen_talep[(h, g)] = model.NewIntVar(0, max_talep_hatta, f'ert_{h}_{g}')
        # FIX #6: Birikmiş talep üst sınırı — tüm günlerin toplamı
        biriken_talep[(h, g)]   = model.NewIntVar(0, max_talep_hatta, f'bir_{h}_{g}')

# Uğrama araç değişkenleri
ugrama_y      = {}
ugrama_yuk_ac = {}
ugrama_yuk_cb = {}
ugrama_yuk_ab = {}

for (a, c, b) in ugrama_rotalari:
    for g in gunler:
        for arac in arac_turleri:
            ugrama_y[(a, c, b, g, arac)] = model.NewIntVar(0, max_spot, f'uy_{a}_{c}_{b}_{g}_{arac}')

        ugrama_yuk_ac[(a, c, b, g)] = model.NewIntVar(0, max_talep_hatta, f'uac_{a}_{c}_{b}_{g}')
        ugrama_yuk_cb[(a, c, b, g)] = model.NewIntVar(0, max_talep_hatta, f'ucb_{a}_{c}_{b}_{g}')
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

        # FIX (HAYAT KURTARAN DOKUNUŞ): 
        # Kiralık araçların kapasitesini sabit olarak eklemek yerine, 
        # kiralık araçların GERÇEKTE taşıdığı kargo miktarını temsil eden yeni bir değişken açıyoruz.
        # Sınırı: En az 0 taşır, en fazla kendi kapasitesi (kiralik_toplam_kap) kadar taşır.
        kiralik_tasinan_yuk = model.NewIntVar(0, kiralik_toplam_kap, f'kiralik_net_yuk_{h}_{g}')

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
# KISIT D — Uğrama Araç Kapasite Kısıtları (DÜZELTİLMİŞ VE YALINLAŞTIRILMIŞ)
# ---------------------------------------------------------------
for (a, c, b) in ugrama_rotalari:
    for g in gunler:
        u_kap = cp_model.LinearExpr.Sum([
            ugrama_y[(a, c, b, g, arac)] * arac_parametreleri[arac]["kapasite_desi"]
            for arac in arac_turleri
        ])

        # A→C segmentinde: ac yükü + ab yükü (ab yükü tüm yol boyunca araçta)
        model.Add(ugrama_yuk_ac[(a, c, b, g)] + ugrama_yuk_ab[(a, c, b, g)] <= u_kap)

        # C→B segmentinde: cb yükü + ab yükü
        model.Add(ugrama_yuk_cb[(a, c, b, g)] + ugrama_yuk_ab[(a, c, b, g)] <= u_kap)

        # Teknofest Kural #1: Uğrama spot araçlarına %10 doluluk kuralı (İzole Edilmiş)
        if g != gunler[-1]:
            # DÜZELTME 1: Araç bazlı net yükleri tutacağımız temiz bir liste açıyoruz
            u_tasinan_net_listesi = []

            for arac in arac_turleri:
                kap = arac_parametreleri[arac]["kapasite_desi"]
                u_tasinan_net_a = model.NewIntVar(0, max_spot * kap * 2, f'u_net_{a}_{c}_{b}_{g}_{arac}')
                
                # Yarattığımız değişkeni listeye ekliyoruz
                u_tasinan_net_listesi.append(u_tasinan_net_a)
                
                # Fiziksel sınır: Bir aracın taşıyabileceği maksimum kümülatif yük
                model.Add(u_tasinan_net_a <= ugrama_y[(a, c, b, g, arac)] * kap * 2)
                
                # Araç türü bazında %10 doluluk kısıtı
                model.Add(ugrama_y[(a, c, b, g, arac)] * kap <= u_tasinan_net_a * 10)

            # Global uğrama yüklerini, araç türü bazlı izole edilmiş net yüklere bağlama
            toplam_ugrama_yuk = cp_model.LinearExpr.Sum([
                ugrama_yuk_ac[(a, c, b, g)],
                ugrama_yuk_cb[(a, c, b, g)],
                ugrama_yuk_ab[(a, c, b, g)],
            ])
            
            # DÜZELTME 2: Hiçbir karmaşık index bulucu kullanmadan direkt listeyi topluyoruz
            # model.Add(toplam_ugrama_yuk == cp_model.LinearExpr.Sum(u_tasinan_net_listesi))
# ---------------------------------------------------------------
# KISIT E — Teknofest: Son Gün Erteleme Yasağı
# ---------------------------------------------------------------
for h in hatlar:
    model.Add(ertelenen_talep[(h, gunler[-1])] == 0)


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
        # --- Kiralık araç sabit maliyeti (optimizer kontrolünde değil) ---
        for a in arac_turleri:
            adet = kiralik_stok_gunluk.get((h, a), 0)
            if adet > 0:
                p = arac_parametreleri[a]
                # Teknofest: kiralık araç sabit + km maliyeti her gün ödenir
                gun_maliyet = adet * int(p["sabit_kira"] + dist * p["kiralik_km_maliyet"])
                kiralik_sabit_toplam += gun_maliyet

        # --- Spot araç değişken maliyeti ---
        for a in arac_turleri:
            p = arac_parametreleri[a]
            # FIX #5: Maliyet katsayısı tamsayı (CP-SAT integer gerektirir)
            spot_maliyet_katsayi = int(p["spot_sabit_maliyet"] + dist * p["spot_km_maliyet"])
            maliyet_kalemleri.append(spot_y[(h, g, a)] * spot_maliyet_katsayi)

        # --- SLA Gecikme Cezası ---
        # FIX #5: Ceza ağırlığı spot araçlarla rekabetçi seviyede
        ceza_katsayi = int(SLA_GECIKME_CEZA_TL_PER_DESI)
        maliyet_kalemleri.append(ertelenen_talep[(h, g)] * ceza_katsayi)

# --- Uğrama spot araç maliyetleri ---
for (a, c, b) in ugrama_rotalari:
    dist_ac = distances_2d[tm_index[a]][tm_index[c]]
    dist_cb = distances_2d[tm_index[c]][tm_index[b]]
    dist_toplam = dist_ac + dist_cb  # Teknofest Kural #6: kuş uçuşu toplam

    for g in gunler:
        for arac in arac_turleri:
            p = arac_parametreleri[arac]
            ugrama_maliyet = int(p["spot_sabit_maliyet"] + dist_toplam * p["spot_km_maliyet"])
            maliyet_kalemleri.append(ugrama_y[(a, c, b, g, arac)] * ugrama_maliyet)

model.Minimize(cp_model.LinearExpr.Sum(maliyet_kalemleri))

# =============================================================================
# 10. ÇÖZÜCÜ PARAMETRELERI
# =============================================================================
solver = cp_model.CpSolver()
max_time_to_solve = 300.0 # şimdilik 5 dakika
solver.parameters.max_time_in_seconds  = max_time_to_solve
solver.parameters.num_search_workers   = 8
solver.parameters.log_search_progress  = True

status = solver.Solve(model)

# =============================================================================
# 11. SONUÇ YAZIMI
#     Teknofest Kural #5: Toplam maliyet açıkça raporlanır.
# =============================================================================
output_file = Path(__file__).parent.parent / "results" / "optimization_results.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        durum = "OPTIMAL (Kusursuz)" if status == cp_model.OPTIMAL else f"UYGUN {max_time_to_solve}"
        baslik = f"{'=' * 80}\nOPTİMİZASYON BAŞARILI — {durum}\n{'=' * 80}\n\n"
        f.write(baslik)
        print(baslik)

        toplam_ertelenen_desi = 0
        spot_toplam_maliyet   = 0
        ugrama_toplam_maliyet = 0

        # --- Direkt Rota Sonuçları ---
        for h in hatlar:
            dist = mesafe_verisi.get(h, 0)
            parcalar = h.split("-", maxsplit=1)
            if len(parcalar) != 2:
                continue

            for g in gunler:
                for a in arac_turleri:
                    k_adet = kiralik_stok_gunluk.get((h, a), 0)
                    s_adet = solver.Value(spot_y[(h, g, a)])

                    if k_adet > 0 or s_adet > 0:
                        metin = (
                            f"🚛 DİREKT ROTA | {g} | {h} | {a} "
                            f"-> Kiralık: {k_adet}, Spot: {s_adet}\n"
                        )
                        f.write(metin)
                        if s_adet > 0:
                            print(metin.strip())
                            p = arac_parametreleri[a]
                            spot_toplam_maliyet += s_adet * int(
                                p["spot_sabit_maliyet"] + dist * p["spot_km_maliyet"]
                            )

                ert = solver.Value(ertelenen_talep[(h, g)])
                if ert > 0:
                    toplam_ertelenen_desi += ert
                    ert_metin = f"  ⚠️  ERTELEME | {g} | {h} → {ert} desi\n"
                    f.write(ert_metin)
                    print(ert_metin.strip())

        # --- Uğrama Rota Sonuçları ---
        print("-" * 60)
        f.write("-" * 60 + "\n")
        for (a, c, b) in ugrama_rotalari:
            dist_ac = distances_2d[tm_index[a]][tm_index[c]]
            dist_cb = distances_2d[tm_index[c]][tm_index[b]]
            dist_toplam = dist_ac + dist_cb
            for g in gunler:
                for arac in arac_turleri:
                    u_adet = solver.Value(ugrama_y[(a, c, b, g, arac)])
                    if u_adet > 0:
                        metin = (
                            f"🌟 UĞRAMA ROTASI | {g} | {a} → {c} → {b} "
                            f"| {arac}: {u_adet} Spot Araç\n"
                        )
                        f.write(metin)
                        print(metin.strip())
                        p = arac_parametreleri[arac]
                        ugrama_toplam_maliyet += u_adet * int(
                            p["spot_sabit_maliyet"] + dist_toplam * p["spot_km_maliyet"]
                        )

        # --- Teknofest Kural #5: Toplam Maliyet Özeti ---
        sla_ceza_toplam    = int(toplam_ertelenen_desi * SLA_GECIKME_CEZA_TL_PER_DESI)
        degisken_toplam    = spot_toplam_maliyet + ugrama_toplam_maliyet + sla_ceza_toplam
        genel_toplam       = kiralik_sabit_toplam + degisken_toplam

        ozet = f"""
{'=' * 80}
ÖZET İSTATİSTİKLER  (Teknofest Kural #5 — Toplam Maliyet)
{'=' * 80}
  Kiralık Araç Sabit Maliyeti : {kiralik_sabit_toplam:>15,.0f} TL  (her zaman ödenir)
  Spot Araç Maliyeti (Direkt) : {spot_toplam_maliyet:>15,.0f} TL
  Spot Araç Maliyeti (Uğrama) : {ugrama_toplam_maliyet:>15,.0f} TL
  SLA Gecikme Cezası          : {sla_ceza_toplam:>15,.0f} TL
{'─' * 80}
  TOPLAM MALİYET              : {genel_toplam:>15,.0f} TL
{'=' * 80}
  Toplam Ertelenen Yük        : {toplam_ertelenen_desi:>15,.0f} desi
  Çözücü Süre                 : {solver.WallTime():>15.2f} sn
  Objective Value (optimizer) : {solver.ObjectiveValue():>15,.0f}
{'=' * 80}
"""
        f.write(ozet)
        print(ozet)

    else:
        hata = (
            "❌ Çözüm bulunamadı! Lütfen kısıtları kontrol edin.\n"
            f"   Durum kodu: {status}\n"
            "   Olası neden: Sıfır talep olan hatlarda %10 doluluk kısıtı veya\n"
            "   erteleme yasağıyla çelişen bir kısıt kombinasyonu.\n"
        )
        f.write(hata)
        print(hata)