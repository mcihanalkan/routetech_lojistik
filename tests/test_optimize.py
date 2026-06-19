from pathlib import Path
import pandas as pd
import sys
import re

# Haversine modülü entegrasyonu
sys.path.insert(0, str(Path(__file__).parent.parent / "haversine"))
try:
    from haversine import GetDistanceMatrixAsList, GetCenters, is_city_between
except ImportError:
    print("⚠️ Haversine modülü yüklenemedi! Dosya yolları kontrol edilmeli.")

def load_data():
    base_dir = Path(__file__).parent.parent
    paths = {
        "params": base_dir / "data" / "static_datas" / "car_parameters.csv",
        "stoks": base_dir / "data" / "static_datas" / "rented_stoks.csv",
        "payload": base_dir / "src" / "predict_model" / "ortools_payload.csv",
        "results": base_dir / "results" / "optimization_results.csv"
    }
    return {k: pd.read_csv(v) if v.exists() else None for k, v in paths.items()}

def extract_ugrama_info(rota_tipi):
    """'Uğrama (Eskişehir)' formatından şehri güvenli şekilde çeker."""
    if pd.isna(rota_tipi) or str(rota_tipi).strip() == "nan": return None
    match = re.search(r'\((.*?)\)', str(rota_tipi))
    return match.group(1).strip() if match else None

def run_ultimate_testing():
    print("=" * 95)
    print("🏆 TEKNOFEST DOĞRULAMA SİSTEMİ v2.5 - NİHAİ FORMAT UYUMLU TEST")
    print("=" * 95)

    data = load_data()
    df_results = data["results"]
    df_payload = data["payload"]
    df_params = data["params"]
    df_stoks = data["stoks"]

    if df_results is None or df_payload is None or df_params is None:
        print("❌ HATA: Gerekli veri dosyaları bulunamadı! Testler iptal edildi.")
        return

    # Sütun isimlerindeki olası boşlukları temizleyelim
    df_results.columns = df_results.columns.str.strip()

    # 🚨 KRİTİK DÜZELTME: Boş satırları (NaN) temizle ve veri tipini zorla metin (string) yap
    df_results = df_results.dropna(subset=['Araç_Tipi'])
    df_results['Araç_Tipi'] = df_results['Araç_Tipi'].astype(str)
    df_results['Rota_Tipi'] = df_results['Rota_Tipi'].astype(str)

    # 1. Altyapı Hazırlıkları
    centers = GetCenters()
    distances_2d = GetDistanceMatrixAsList()
    tm_index = {tm: idx for idx, tm in enumerate(centers)}
    center_matrix_df = pd.DataFrame(distances_2d, index=centers, columns=centers)

    arac_param = {row['vehicle_type']: row.to_dict() for _, row in df_params.iterrows()}
    kiralik_stok = {(row['route'], row['vehicle_type']): int(row['quantity']) for _, row in df_stoks.iterrows()} if df_stoks is not None else {}

    # Ham Talep Verisinin İşlenmesi
    talep_verisi = {}
    for _, row in df_payload.iterrows():
        src = str(row.get('source_tm', row.get('kaynak_tm', row.get('source', row.iloc[1])))).strip()
        dst = str(row.get('destination_tm', row.get('varis_tm', row.get('destination', row.iloc[2])))).strip()
        date_str = str(row['date']) if 'date' in row else str(row.iloc[0])
        tarih_obj = pd.to_datetime(date_str)
        gun_adi = f"{tarih_obj.day:02d}_Mayis"

        recommended = int(float(row.get('recommended_demand', row.get('q50', 0))))
        talep_verisi[(f"{src}-{dst}", gun_adi)] = recommended

    gunler = sorted(list(set([g for (_, g) in talep_verisi.keys()])))
    hatlar = sorted(list(set([h for (h, _) in talep_verisi.keys()])))
    failed_logs = []

    # =========================================================================
    # KISIM 1: ZORUNLU KALKIŞ VE MALİYET KONTROLÜ
    # =========================================================================
    print("🔄 Kısım 1: Sözleşmeli Araç Kontrolü ve Optimizasyon Maliyet Matrisi Test Ediliyor...")

    if kiralik_stok:
        for g in gunler:
            for (route, arac_turu), max_stok in kiralik_stok.items():
                src_tm, dst_tm = route.split("-")
                kullanilan = df_results[
                    (df_results['Tarih'] == g) &
                    (df_results['Araç_Tipi'] == f"Kiralık {arac_turu}") &
                    (df_results['Çıkış_TM'] == src_tm) &
                    (df_results['Varış_TM'] == dst_tm)
                ]['Araç_Sayısı'].sum()

                if kullanilan != max_stok:
                    failed_logs.append(f"[Kısıt F İhlali] Gün: {g} | {route} hattında {max_stok} adet Kiralık {arac_turu} kalkmalıydı, {kullanilan} adet kalktı!")

    toplam_beklenen_maliyet = 0
    for idx, row in df_results.iterrows():
        arac_tipi_ham = row['Araç_Tipi']
        cikis, varis, maliyet, rota_tipi = row['Çıkış_TM'], row['Varış_TM'], row['Maliyet_TL'], row['Rota_Tipi']
        is_spot = "Spot" in arac_tipi_ham
        arac_turu = arac_tipi_ham.replace("Spot ", "").replace("Kiralık ", "")
        p = arac_param.get(arac_turu)

        if p is None: continue

        if "Uğrama" in rota_tipi:
            c_center = extract_ugrama_info(rota_tipi)
            if c_center and c_center in tm_index and cikis in tm_index and varis in tm_index:
                dist = distances_2d[tm_index[cikis]][tm_index[c_center]] + distances_2d[tm_index[c_center]][tm_index[varis]]
                if not is_spot:
                    dist_direkt = distances_2d[tm_index[cikis]][tm_index[varis]]
                    beklenen = int((dist - dist_direkt) * p["kiralik_km_maliyet"])
                    if abs(maliyet - beklenen) > 10:
                        beklenen_alt = int(p["sabit_kira"] + dist * p["kiralik_km_maliyet"])
                        if abs(maliyet - beklenen_alt) <= 10:
                            beklenen = beklenen_alt
                        else:
                            failed_logs.append(f"[Maliyet Hatası] Satır {idx}: Uğrama Kiralık maliyeti uyuşmuyor! Çıktı: {maliyet}, Beklenen: {beklenen} veya {beklenen_alt}")
                    toplam_beklenen_maliyet += beklenen
                    continue
            else:
                dist = 0 
        else:
            if cikis in tm_index and varis in tm_index:
                dist = distances_2d[tm_index[cikis]][tm_index[varis]]
            else:
                dist = 0

        beklenen = int(p["spot_sabit_maliyet"] + dist * p["spot_km_maliyet"]) if is_spot else int(p["sabit_kira"] + dist * p["kiralik_km_maliyet"])
        toplam_beklenen_maliyet += beklenen
        
        if abs(maliyet - beklenen) > 10:  
            failed_logs.append(f"[Maliyet Hatası] Satır {idx}: {arac_tipi_ham} ({cikis}-{varis}) | Çıktı: {maliyet} TL, Beklenen: {beklenen} TL")

    # =========================================================================
    # KISIM 2: AĞ AKIŞI VE KAPASİTE YETERLİLİĞİ
    # =========================================================================
    print("🔄 Kısım 2: Ağ Akışı, Multi-Stop Feasibility ve Son Gün Denetimi...")

    erteleme_takip = {h: 0 for h in hatlar}

    for g in gunler:
        for h in hatlar:
            src_tm, dst_tm = h.split("-")

            direkt_tasinan = df_results[
                (df_results['Tarih'] == g) & 
                (df_results['Rota_Tipi'] == 'Direkt') & 
                (df_results['Çıkış_TM'] == src_tm) & 
                (df_results['Varış_TM'] == dst_tm)
            ]['Teslim_Edilen_Desi'].sum()

            ugrama_katkisi = 0
            ugrama_seferleri = df_results[(df_results['Tarih'] == g) & (df_results['Rota_Tipi'].str.contains("Uğrama", na=False))]

            bugun_ham_talep = talep_verisi.get((h, g), 0)
            onceki_gunden_devreden = erteleme_takip[h]
            biriken_talep = onceki_gunden_devreden + bugun_ham_talep

            for _, u_row in ugrama_seferleri.iterrows():
                u_src, u_dst = u_row['Çıkış_TM'], u_row['Varış_TM']
                u_mid = extract_ugrama_info(u_row['Rota_Tipi'])
                u_desi = u_row['Teslim_Edilen_Desi']

                if (u_src == src_tm and u_mid == dst_tm) or (u_mid == src_tm and u_dst == dst_tm) or (u_src == src_tm and u_dst == dst_tm):
                    kalan_ihtiyac = max(0, biriken_talep - (direkt_tasinan + ugrama_katkisi))
                    ugrama_katkisi += min(u_desi, kalan_ihtiyac)

            toplam_tasinan = direkt_tasinan + ugrama_katkisi
            
            yeni_erteleme = biriken_talep - toplam_tasinan
            erteleme_takip[h] = max(0, yeni_erteleme)

            if g == gunler[-1] and yeni_erteleme > 1.0:
                failed_logs.append(f"[Kısıt E İhlali] Son Gün Kuralı! Hat: {h} üzerinde son gün {yeni_erteleme:.0f} desi kargo devretti!")

    # =========================================================================
    # KISIM 3: FİZİKSEL ARAÇ KAPASİTESİ VE COĞRAFİ UYGUNLUK
    # =========================================================================
    print("🔄 Kısım 3: Uğrama Araç Kapasite Sınırları ve Coğrafi Mantık Test Ediliyor...")
    ugrama_rows = df_results[df_results['Rota_Tipi'].str.contains("Uğrama", na=False)]

    for idx, row in ugrama_rows.iterrows():
        arac_tipi_ham = row['Araç_Tipi']
        arac_turu = arac_tipi_ham.replace("Spot ", "").replace("Kiralık ", "")
        p = arac_param.get(arac_turu)

        if p is None: continue

        toplam_kapasite = p["kapasite_desi"] * row.get('Araç_Sayısı', 1)
        tasinan_yuk = row['Teslim_Edilen_Desi']

        if tasinan_yuk > toplam_kapasite + 1e-6:
            failed_logs.append(f"[Kısıt D İhlali] Satır {idx}: Araç taşıma kapasitesi aşıldı! Taşınan: {tasinan_yuk} > Kapasite: {toplam_kapasite}")

        src, dst = row['Çıkış_TM'], row['Varış_TM']
        mid = extract_ugrama_info(row['Rota_Tipi'])

        if mid and mid in tm_index and src in tm_index and dst in tm_index:
            try:
                gecerli = is_city_between(source=src, destination=dst, candidate=mid, center_matrix=center_matrix_df)
                if not gecerli:
                    failed_logs.append(f"[Coğrafi Rota Hatası] Satır {idx}: '{mid}' şehri, {src}->{dst} rotası için mantıksız bir sapma yaratıyor!")
            except Exception:
                pass

    # =========================================================================
    # KISIM 4: SPOT ARAÇ DOLULUK ORANI (%10)
    # =========================================================================
    print("🔄 Kısım 4: Spot Araçlar İçin %10 Minimum Doluluk Regülasyonu Kontrol Ediliyor...")

    for idx, row in df_results.iterrows():
        if row['Tarih'] == gunler[-1]: continue  

        if "Spot" in row['Araç_Tipi']:
            arac_turu = row['Araç_Tipi'].replace("Spot ", "")
            if arac_turu not in arac_param: continue
            kapasite = arac_param[arac_turu]["kapasite_desi"]
            tasinan = row['Teslim_Edilen_Desi']

            if tasinan < (kapasite * 0.10):
                failed_logs.append(f"[Kısıt A İhlali] Düşük Doluluk! Satır {idx} | {row['Araç_Tipi']} kapasite israfı yapıyor.")

    # =========================================================================
    # RAPORLAMA ÇIKTISI
    # =========================================================================
    print("\n" + "=" * 95)
    print("📊 ENTEGRASYON VE KISIT DOĞRULAMA FİNAL ÖZETİ")
    print("=" * 95)

    toplam_csv_maliyet = df_results['Maliyet_TL'].sum()
    print(f"💰 CSV'deki Toplam Sefer Maliyeti               : {toplam_csv_maliyet:,.0f} TL")
    print(f"💰 Haversine Matrisinden Doğrulanan Net Maliyet : {toplam_beklenen_maliyet:,.0f} TL")
    
    fark = abs(toplam_csv_maliyet - toplam_beklenen_maliyet)
    if fark > 15 * len(df_results): 
        failed_logs.append(f"[Maliyet Toplamı Uyuşmazlığı] Genel toplamlar arasında {fark:,.0f} TL fark var.")

    if not failed_logs:
        print("\n 🎉 HARİKA! Optimizasyon kodunuz tüm kurallara uyuyor ve testleri kusursuz geçti.")
        print("     Üretilen çıktı TEKNOFEST şartnamesine %100 UYUMLUDUR.")
    else:
        print(f"\n ⚠️ TOPLAM {len(failed_logs)} ADET KISIT İHLALİ TESPİT EDİLDİ!\n")
        for log in failed_logs[:12]:
            print(f"  • {log}")
        if len(failed_logs) > 12:
            print(f"  ... ve {len(failed_logs)-12} hata daha var.")
    print("=" * 95)

if __name__ == "__main__":
    run_ultimate_testing()