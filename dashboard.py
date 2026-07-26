import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import folium
from streamlit_folium import st_folium

# --- SAYFA AYARLARI ---
st.set_page_config(layout="wide", page_title="Lojistik Karar Destek Arayüzü")
st.title("📦 Gelişmiş Çözüm: Konsolidasyon ve Talep İzleme Paneli")
st.markdown("Zaman serisi tahminleri ile optimizasyon kararlarının zaman akışına (09:00 -> 17:00 -> Ertesi Gün) ve haritaya göre analizi.")

# --- TÜRKİYE KOORDİNAT SÖZLÜĞÜ (Genişletilebilir) ---
TURKIYE_KOORDINATLAR = {
    "Adana": [37.0000, 35.3213], "Ankara": [39.9208, 32.8541], "Antalya": [36.8841, 30.7056],
    "Balıkesir": [39.6484, 27.8826], "Bilecik": [40.1451, 29.9799], "Bursa": [40.1824, 29.0633],
    "Denizli": [37.7765, 29.0864], "Erzincan": [39.7500, 39.5000], "Eskişehir": [39.7767, 30.5206],
    "Gaziantep": [37.0662, 37.3833], "Isparta": [37.7648, 30.5566], "İstanbul": [41.0082, 28.9784],
    "İzmir": [38.4192, 27.1287], "Karaman": [37.1811, 33.2222], "Kocaeli": [40.8533, 29.8815],
    "Konya": [37.8667, 32.4833], "Kütahya": [39.4167, 29.9833], "Manisa": [38.6191, 27.4289],
    "Mardin": [37.3122, 40.7339], "Mersin": [36.8000, 34.6333], "Sivas": [39.7477, 37.0179],
    "Şanlıurfa": [37.1674, 38.7955], "Tekirdağ": [40.9780, 27.5110], "Yalova": [40.6500, 29.2667],
    "Zonguldak": [41.4564, 31.7987]
}

# --- 1. VERİ OKUMA VE ÖN İŞLEME ---
@st.cache_data
def load_data():
    RESULTS_DIR = "results"

    talep_path = os.path.join(RESULTS_DIR, "Talep-tahmini.xlsx")
    opt_path = os.path.join(RESULTS_DIR, "optimization_results.csv")

    # 1. Talep Tahmin Verisi Okuma (EXCEL)
    # DUZELTME: sheet adi projeden projeye "Sheet1" olmayabilir (bkz. Kiralik_Araclar.xlsx
    # "Sayfa1" ornegi) - once "Sheet1" dene, yoksa ilk sheet'e (index 0) dus.
    try:
        df_talep = pd.read_excel(talep_path, sheet_name="Sheet1")
    except ValueError:
        df_talep = pd.read_excel(talep_path, sheet_name=0)

    df_talep['Tarih'] = pd.to_datetime(df_talep['Tarih'], format='%d.%m.%Y', errors='coerce').dt.date
    df_talep['Slot'] = df_talep['Talep Tamamlama Saati'].astype(str).str[:5]
    df_talep['Rota'] = df_talep['Çıkış Transfer Merkezi'] + " -> " + df_talep['Varış Transfer Merkezi']

    # 2. Optimizasyon Çıktısı Okuma (CSV)
    df_opt = pd.read_csv(opt_path)
    df_opt['Tarih'] = pd.to_datetime(df_opt['Tarih']).dt.date
    df_opt['Talep_Tarihi'] = pd.to_datetime(df_opt['Talep_Tarihi']).dt.date
    df_opt['Slot'] = df_opt['Slot'].astype(str).str[:5]
    df_opt['Talep_Slotu'] = df_opt['Talep_Slotu'].astype(str).str[:5]
    df_opt['Rota'] = df_opt['Nihai_Kaynak'] + " -> " + df_opt['Nihai_Varis']

    # DUZELTME (Bug 2 & 3 icin temel): bir satirin bu talebin ILK bacagi (asil
    # cikis noktasindan kalkis) mi, yoksa aktarma/relay bacagi mi oldugunu
    # ayirt eden bayrak. Nihai_Kaynak'tan kalkan bacak = ilk bacak.
    # "Rota" (Nihai_Kaynak->Nihai_Varis) alani AYNI KALIYOR aktarmali
    # sevkiyatin TUM bacaklarinda - bu yuzden bacak pozisyonunu bu bayrak
    # olmadan ayirt etmenin bir yolu yok (onceki versiyondaki hatanin kaynagi).
    df_opt['Ilk_Bacak_Mi'] = df_opt['Cikis_TM'] == df_opt['Nihai_Kaynak']

    return df_talep, df_opt

try:
    df_talep, df_opt = load_data()
except Exception as e:
    st.error(f"❌ Veriler yüklenirken hata oluştu! Dosya yollarını kontrol edin. Hata: {e}")
    st.stop()

# --- 2. HİYERARŞİK FİLTRELER ---
benzersiz_rotalar = sorted(df_talep['Rota'].dropna().astype(str).unique())
benzersiz_gunler = sorted(df_talep['Tarih'].dropna().unique())

st.sidebar.header("🔍 Filtreler")
secilen_rota = st.sidebar.selectbox("Rota Seçiniz (Nihai Kaynak -> Nihai Varış):", benzersiz_rotalar)


# --- YARDIMCI: gerçek kalkış saatini 09:00/17:00 "mantıksal slot"una eşler ---
def mantiksal_slota_esle(saat_str: str) -> str:
    """Gercek kalkis saatini (herhangi bir dakika olabilir, KISITLAR.md Bolum 4)
    09:00/17:00 nominal pencerelerinden birine yaklastirir.

    NOT: Bu bir HEURISTIKTIR - gercek kalkis dakika hassasiyetinde olabildigi
    icin kesin bir "doğru" esik yoktur. 13:00 (09:00-17:00 orta noktasi)
    kullaniliyor; sinira çok yakin saatlerde (12:55 / 13:05 gibi) yorum
    degisebilir - bu panel bunu kesin gecikme tespiti degil, YAKLASIK bir
    gorsellestirme olarak sunar.
    """
    try:
        saat_int = int(str(saat_str).split(":")[0])
    except (ValueError, IndexError):
        return "09:00"
    return '09:00' if saat_int < 13 else '17:00'


# --- 3. HARİTA (FOLIUM) GÖRSELLEŞTİRMESİ ---
def rota_haritasini_ciz(rota_adi, df_opt_filtered):
    # Ana rotayı ayır (Mantıksal Kaynak -> Varış)
    nihai_kaynak, nihai_varis = rota_adi.split(" -> ")

    # Harita Merkezini Bul
    merkez_koordinat = TURKIYE_KOORDINATLAR.get(nihai_kaynak, [39.0, 35.0])
    m = folium.Map(location=merkez_koordinat, zoom_start=6, tiles="CartoDB positron")

    # Eğer henüz hiç sefer planlanmadıysa sadece A->B düz çizgi çiz
    if df_opt_filtered.empty:
        varis_koordinat = TURKIYE_KOORDINATLAR.get(nihai_varis, [39.0, 35.0])
        folium.Marker(merkez_koordinat, tooltip=f"Çıkış: {nihai_kaynak}", icon=folium.Icon(color='gray', icon='truck', prefix='fa')).add_to(m)
        folium.Marker(varis_koordinat, tooltip=f"Varış: {nihai_varis}", icon=folium.Icon(color='gray', icon='flag', prefix='fa')).add_to(m)
        folium.PolyLine(locations=[merkez_koordinat, varis_koordinat], color="gray", weight=2, dash_array="10", tooltip="Bu Slot İçin Sefer Bulunamadı").add_to(m)
        return m

    df_harita = df_opt_filtered.copy()

    # Araç Tipi ve Türü varsa birleştir (Örn: Spot + Kamyonet = Spot Kamyonet)
    if 'Arac_Turu' in df_harita.columns:
        df_harita['Tam_Arac_Tipi'] = df_harita['Arac_Tipi'].astype(str) + " " + df_harita['Arac_Turu'].astype(str)
    else:
        df_harita['Tam_Arac_Tipi'] = df_harita['Arac_Tipi'].astype(str)

    # Dinamik kolon kontrolü (Eski ve Yeni CSV'ye uyumlu olması için)
    group_cols = ['Cikis_TM', 'Varis_TM']
    for col in ['Arac_ID', 'Slot', 'Varis_Saati', 'Yolculuk_Suresi_Dk']:
        if col in df_harita.columns:
            group_cols.append(col)

    # DUZELTME (Bug 1): 'Bu_Talebin_Desisi' toplami SADECE secili rotaya/gune
    # ait kismi verir - aracin GERCEK toplam yukunu YANLIS temsil eder (bir
    # arac ayni anda birden fazla farkli nihai varisa yuk tasiyabilir, bkz.
    # KISITLAR.md Bolum 6). CSV'deki 'Bacak_Toplam_Desi' kolonu, aracin
    # FILTRELERDEN BAGIMSIZ gercek toplam yukunu tasir - onu da (varsa) ayrica
    # aliyoruz ki tooltip'te ikisini birbirinden ayirt ederek gosterebilelim.
    agg_map = {
        'Bu_Talebin_Desisi': 'sum',
        'Tam_Arac_Tipi': 'first',
        'Talep_ID': lambda x: ", ".join(x.astype(str).unique()),
        'Rota_Tipi': lambda x: "<br>".join(x.astype(str).unique()),
    }
    if 'Bacak_Toplam_Desi' in df_harita.columns:
        agg_map['Bacak_Toplam_Desi'] = 'first'

    bacak_ozetleri = df_harita.groupby(group_cols).agg(agg_map).reset_index()

    cizilen_noktalar = set()

    for _, row in bacak_ozetleri.iterrows():
        c_tm = row['Cikis_TM']
        v_tm = row['Varis_TM']
        bu_rotaya_ait_yuk = row['Bu_Talebin_Desisi']

        # Yeni veriler (Varsa al, yoksa Bilinmiyor yaz)
        arac_id = row.get('Arac_ID', 'Bilinmiyor')
        cikis_saati = row.get('Slot', 'Bilinmiyor')
        varis_saati = row.get('Varis_Saati', 'Bilinmiyor')
        yolculuk_sure = row.get('Yolculuk_Suresi_Dk', 'Bilinmiyor')

        arac_tipi = row['Tam_Arac_Tipi']
        talep_idler = row['Talep_ID']
        konsolidasyon_durumu = row['Rota_Tipi']

        c_koordinat = TURKIYE_KOORDINATLAR.get(c_tm, [39.0, 35.0])
        v_koordinat = TURKIYE_KOORDINATLAR.get(v_tm, [39.0, 35.0])

        if c_tm not in cizilen_noktalar:
            folium.Marker(c_koordinat, tooltip=f"{c_tm} (Transfer Merkezi)", icon=folium.Icon(color='green' if c_tm==nihai_kaynak else 'blue', icon='building', prefix='fa')).add_to(m)
            cizilen_noktalar.add(c_tm)
        if v_tm not in cizilen_noktalar:
            folium.Marker(v_koordinat, tooltip=f"{v_tm} (Transfer Merkezi)", icon=folium.Icon(color='red' if v_tm==nihai_varis else 'blue', icon='building', prefix='fa')).add_to(m)
            cizilen_noktalar.add(v_tm)

        # DUZELTME (Bug 1): gercek arac toplami varsa VE secili rotaya ait
        # yukten FARKLIYSA, ikisini ayri ayri goster - "Toplam Yuk" etiketini
        # sadece gercekten aracin TEK basina o rotaya ait oldugu durumda kullan.
        gercek_arac_toplami = row.get('Bacak_Toplam_Desi', None)
        if gercek_arac_toplami is not None and abs(gercek_arac_toplami - bu_rotaya_ait_yuk) > 0.01:
            yuk_satiri = (
                f"<b style='color: #16a34a; font-size: 14px;'>📦 Bu Rotaya Ait Yük:</b> {bu_rotaya_ait_yuk:.2f} Desi<br>"
                f"<b style='color: #6b7280; font-size: 12px;'>ℹ️ Aracın Gerçek Toplam Yükü:</b> {gercek_arac_toplami:.2f} Desi "
                f"<span style='font-size:10px;'>(bu araç başka rotalara giden yük de taşıyor)</span><br>"
            )
        else:
            yuk_satiri = f"<b style='color: #16a34a; font-size: 14px;'>📦 Araçtaki Toplam Yük:</b> {bu_rotaya_ait_yuk:.2f} Desi<br>"

        # Zenginleştirilmiş HTML Tooltip (CSS İle Şıklaştırıldı)
        html_tooltip = f"""
        <div style='font-family: Arial, sans-serif; font-size: 13px; min-width: 300px; padding: 5px;'>
            <b style='color: #2563eb; font-size: 14px;'>📍 Fiziksel Bacak:</b> {c_tm} ➡️ {v_tm}<br>
            <hr style='margin: 6px 0; border: 0; border-top: 1px solid #e5e7eb;'>
            <b style='color: #4b5563;'>🚚 Araç:</b> {arac_id} ({arac_tipi})<br>
            <b style='color: #4b5563;'>⏱️ Gerçek Kalkış:</b> {cikis_saati} | <b style='color: #4b5563;'>🏁 Varış:</b> {varis_saati}<br>
            <b style='color: #4b5563;'>⏳ Yolculuk Süresi:</b> {yolculuk_sure} Dk<br>
            {yuk_satiri}
            <hr style='margin: 6px 0; border: 0; border-top: 1px solid #e5e7eb;'>
            <b style='color: #d97706;'>🔖 Taşıdığı Talep ID'ler:</b> <span style='font-size: 11px;'>{talep_idler}</span><br>
            <b style='color: #dc2626;'>🚦 Yüklerin Durumu:</b><br><span style='font-size: 11px;'>{konsolidasyon_durumu}</span>
        </div>
        """

        # İçinde herhangi bir "Uğramalı" veya "Konsolidasyon" geçen satır varsa rengi turuncu yap
        is_konsolide = "Uğramalı" in konsolidasyon_durumu or "Konsolidasyon" in konsolidasyon_durumu

        folium.PolyLine(
            locations=[c_koordinat, v_koordinat],
            color="orange" if is_konsolide else "blue",
            weight=5,
            opacity=0.8,
            tooltip=html_tooltip
        ).add_to(m)

    return m

# --- 4. GÜN SEKMELERİ VE KART GÖRSELLEŞTİRME STİLLERİ ---
sekme_isimleri = [g.strftime('%d %B %Y') for g in benzersiz_gunler]
sekmeler = st.tabs(sekme_isimleri)

def css_stilleri():
    st.markdown("""
        <style>
        .aktif-kart {
            background-color: #f0fdf4; border: 2px solid #22c55e; border-radius: 10px; padding: 20px;
        }
        .pasif-kart {
            background-color: #f3f4f6; border: 2px dashed #9ca3af; border-radius: 10px; padding: 20px; opacity: 0.7;
        }
        .kismi-kart {
            background-color: #fffbeb; border: 2px solid #f59e0b; border-radius: 10px; padding: 20px;
        }
        .badge-gelen {
            background-color: #3b82f6; color: white; padding: 4px 8px; border-radius: 5px; font-size: 12px; font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

css_stilleri()


def slot_analizi_yap(secilen_tarih, slot_saati, rota, df_talep, df_opt):
    """Bir (gun, slot, rota) icin talep vs gerceklesen sevkiyat karsilastirmasi.

    DUZELTME (Bug 2 + Bug 3): Onceki versiyon, bu talebe ait TUM bacak
    satirlarini (Rota == Nihai_Kaynak->Nihai_Varis filtresi TUM bacaklarda
    ayni oldugu icin) ayirt etmeden `.iloc[0]` ile TEK bir satir aliyordu -
    bu, aktarmali (relay) sevkiyatlarda HANGI bacagin alindigini belirsiz
    kiliyordu ve bolunmus taleplerin (D0001-1/-2 gibi) FARKLI kaderlerini
    (biri zamaninda, biri ertelenmis) gizliyordu. Ayrica "gelen yuk" hesabi,
    normal bir aktarmali sevkiyatin 2. bacagini da yanlislikla "gecikmeden
    gelen yuk" sayabiliyordu.

    Bu versiyon, load_data()'da eklenen 'Ilk_Bacak_Mi' bayragini kullanarak
    SADECE bu talebin ILK (asil cikis noktasindan kalkan) bacagina bakiyor -
    aktarmali sevkiyatlarda sonraki bacaklar bu analize hic girmiyor (onlar
    zaten dogal bir aktarma hareketi, "gecikme" degil).
    """
    base_demand_row = df_talep[(df_talep['Tarih'] == secilen_tarih) & (df_talep['Slot'] == slot_saati) & (df_talep['Rota'] == rota)]
    base_demand = base_demand_row['Tahmin Edilen Desi'].sum() if not base_demand_row.empty else 0.0

    # DUZELTME (Bug 2): sadece ILK bacak satirlari - aktarmali sevkiyatlarin
    # sonraki bacaklari burada YOK, cunku onlar "bu talebin nereden kalktigi"
    # sorusuyla ilgili degil.
    opt_ilk_bacaklar = df_opt[
        (df_opt['Talep_Tarihi'] == secilen_tarih)
        & (df_opt['Talep_Slotu'] == slot_saati)
        & (df_opt['Rota'] == rota)
        & (df_opt['Ilk_Bacak_Mi'])
    ]

    iptal_edildi = False
    kismi_ertelendi = False
    aktarilan_hedef = ""
    zamaninda_desi = 0.0
    ertelenen_desi = 0.0

    if base_demand > 0:
        if opt_ilk_bacaklar.empty:
            iptal_edildi = True
            aktarilan_hedef = "Ertesi Gün veya Daha Sonraki Sefer"
        else:
            # Bir talep (D0001) kapasite nedeniyle ayni ilk bacakta BIRDEN FAZLA
            # araca (farkli Arac_ID'ler) bolunmus olabilir; VEYA aslinda
            # birden fazla ayri parcaya (D0001-1, D0001-2, ...) bolunmus olup
            # her biri FARKLI bir zamanda kalkmis olabilir. AYRICA ayni
            # Talep_ID'nin arac-bolunmesi parcalari bile TEORIDE farkli
            # (Tarih, Slot) kombinasyonuna dusebilir (bu veri setinde 0 vaka
            # olarak dogruladik ama kod bunu VARSAYMAMALI) - bu yuzden
            # ('Talep_ID','Tarih','Slot') ucuyle grupluyoruz, TEK bir satirin
            # (.iloc[0]) tum grubu temsil ettigini varsaymiyoruz.
            parca_sonuclari = []  # (talep_id, gercek_tarih, gercek_mantiksal_slot, desi)
            for (talep_id, g_tarih, g_slot_kesin), grp in opt_ilk_bacaklar.groupby(['Talep_ID', 'Tarih', 'Slot']):
                g_mantiksal = mantiksal_slota_esle(g_slot_kesin)
                g_desi = grp['Bu_Talebin_Desisi'].sum()
                parca_sonuclari.append((talep_id, g_tarih, g_mantiksal, g_desi))

            for talep_id, g_tarih, g_mantiksal, g_desi in parca_sonuclari:
                zamaninda_mi = (g_tarih == secilen_tarih) and (g_mantiksal == slot_saati)
                if zamaninda_mi:
                    zamaninda_desi += g_desi
                else:
                    ertelenen_desi += g_desi

            if ertelenen_desi > 0 and zamaninda_desi > 0:
                kismi_ertelendi = True
                # farkli hedeflerin ozetini olustur
                hedefler = set()
                for talep_id, g_tarih, g_mantiksal, g_desi in parca_sonuclari:
                    if not ((g_tarih == secilen_tarih) and (g_mantiksal == slot_saati)):
                        if g_tarih == secilen_tarih and g_mantiksal == '17:00':
                            hedefler.add("bugün 17:00 seferine")
                        else:
                            hedefler.add(f"{g_tarih.strftime('%d.%m')} {g_mantiksal} seferine")
                aktarilan_hedef = ", ".join(sorted(hedefler))
            elif ertelenen_desi > 0 and zamaninda_desi == 0:
                iptal_edildi = True
                hedefler = set()
                for talep_id, g_tarih, g_mantiksal, g_desi in parca_sonuclari:
                    if g_tarih == secilen_tarih and g_mantiksal == '17:00':
                        hedefler.add("17:00 seferine")
                    else:
                        hedefler.add(f"Ertesi gün ({g_tarih.strftime('%d.%m')}) {g_mantiksal} seferine")
                aktarilan_hedef = ", ".join(sorted(hedefler))

    # DUZELTME (Bug 3): "gelen yuk" hesabi da SADECE ILK bacak satirlarindan
    # yapiliyor - bir aktarmali sevkiyatin 2. (relay) bacagi burada asla
    # "gecikmeden gelen yuk" olarak sayilmiyor, cunku o zaten normal/planli
    # bir aktarma hareketi, gecikme degil.
    opt_bugun_ilk_bacak = df_opt[
        (df_opt['Tarih'] == secilen_tarih) & (df_opt['Rota'] == rota) & (df_opt['Ilk_Bacak_Mi'])
    ]

    gelen_yuk = 0.0
    gelen_mesaj = ""

    # NOT: 'Talep_Tarihi'/'Talep_Slotu' bir Talep_ID'nin sabit ozellikleridir
    # (arac-bolunmesinden etkilenmez), ama 'Slot' (gercek kalkis saati) FARKLI
    # arac parcalarinda TEORIDE farkli olabilir - bu yuzden ('Talep_ID','Slot')
    # ikilisiyle grupluyoruz, tek bir satirin butun Talep_ID'yi temsil ettigini
    # varsaymiyoruz (bkz. slot_analizi_yap'taki ayni gerekce).
    for (talep_id, g_slot_kesin), grp in opt_bugun_ilk_bacak.groupby(['Talep_ID', 'Slot']):
        row_mantiksal_slot = mantiksal_slota_esle(g_slot_kesin)
        talep_tarihi = grp['Talep_Tarihi'].iloc[0]
        talep_slotu = grp['Talep_Slotu'].iloc[0]
        desi = grp['Bu_Talebin_Desisi'].sum()

        if row_mantiksal_slot == slot_saati:
            if talep_tarihi < secilen_tarih or (talep_tarihi == secilen_tarih and talep_slotu != slot_saati):
                gelen_yuk += desi
                if talep_tarihi < secilen_tarih:
                    gelen_mesaj = "Önceki günden devreden yük"
                elif talep_slotu == '09:00' and slot_saati == '17:00':
                    gelen_mesaj = "Bugün 09:00'dan devreden yük"

    kesinlesen_yuk = 0.0 if iptal_edildi else (zamaninda_desi + gelen_yuk)

    return {
        "base_demand": round(base_demand, 2),
        "iptal_edildi": iptal_edildi,
        "kismi_ertelendi": kismi_ertelendi,
        "aktarilan_hedef": aktarilan_hedef,
        "zamaninda_desi": round(zamaninda_desi, 2),
        "ertelenen_desi": round(ertelenen_desi, 2),
        "gelen_yuk": round(gelen_yuk, 2),
        "gelen_mesaj": gelen_mesaj,
        "kesinlesen_yuk": round(kesinlesen_yuk, 2),
    }


def slot_kartini_ciz(data):
    """Bir slot karti (09:00 ya da 17:00) icin HTML render eder - 3 durum:
    iptal (tamamen ertelendi), kismi (bir kismi ertelendi), aktif (planlandi)."""
    if data['iptal_edildi']:
        st.markdown(f"""
        <div class="pasif-kart">
            <h4>Ana Talep: {data['base_demand']} Desi</h4>
        </div>
        """, unsafe_allow_html=True)
        st.error(f"⚠️ Bu sefer tamamen ertelendi. Yükler {data['aktarilan_hedef']} aktarıldı.")
        return

    gelen_html = f'<span class="badge-gelen">+{data["gelen_yuk"]} birim ({data["gelen_mesaj"]})</span>' if data['gelen_yuk'] > 0 else ""

    if data['kismi_ertelendi']:
        st.markdown(f"""
        <div class="kismi-kart">
            <h4>Ana Talep: {data['base_demand']} Desi {gelen_html}</h4>
            <hr>
            <h3 style='color:#b45309;'>⚠️ Kısmen Ertelendi</h3>
            <p><b>Zamanında Giden:</b> {data['zamaninda_desi']} Desi &nbsp;|&nbsp; <b>Ertelenen:</b> {data['ertelenen_desi']} Desi ({data['aktarilan_hedef']})</p>
            <p><b>Kesinleşen Toplam Sefer Yükü:</b> {data['kesinlesen_yuk']} Desi</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="aktif-kart">
            <h4>Ana Talep: {data['base_demand']} Desi {gelen_html}</h4>
            <hr>
            <h3 style='color:#166534;'>✅ Sefer Planlandı</h3>
            <p><b>Kesinleşen Toplam Sefer Yükü:</b> {data['kesinlesen_yuk']} Desi</p>
        </div>
        """, unsafe_allow_html=True)


# --- 5. ARAYÜZÜN ÇİZİLMESİ (RENDER) ---
for idx, sekme in enumerate(sekmeler):
    with sekme:
        secilen_tarih = benzersiz_gunler[idx]
        st.markdown(f"### 📅 {secilen_tarih.strftime('%d.%m.%Y')} | 🛣️ Rota: **{secilen_rota}**")

        # --- HARİTA SLOT FİLTRESİ ---
        st.markdown("##### 📍 Haritada Görüntülenecek Slotu Seçin:")
        secilen_harita_slotu = st.radio(
            "Görselleştirilecek Operasyon Ağı:",
            options=["Tümü (Günlük Ağ)", "09:00 Slotu Operasyonu", "17:00 Slotu Operasyonu"],
            horizontal=True,
            key=f"radio_slot_{idx}",
            label_visibility="collapsed"
        )

        # Haritaya gidecek veriyi filtreleme mantığı (Talep slotuna göre)
        df_gunluk_opt = df_opt[(df_opt['Rota'] == secilen_rota) & (df_opt['Talep_Tarihi'] == secilen_tarih)]

        if secilen_harita_slotu == "09:00 Slotu Operasyonu":
            df_gunluk_opt = df_gunluk_opt[df_gunluk_opt['Talep_Slotu'] == '09:00']
        elif secilen_harita_slotu == "17:00 Slotu Operasyonu":
            df_gunluk_opt = df_gunluk_opt[df_gunluk_opt['Talep_Slotu'] == '17:00']

        with st.expander(f"🗺️ {secilen_harita_slotu} Haritasını İncele", expanded=True):
            rota_haritasi = rota_haritasini_ciz(secilen_rota, df_gunluk_opt)
            st_folium(rota_haritasi, height=350, key=f"map_{idx}", use_container_width=True)

        st.write("---")

        # --- KARTLARIN ÇİZİMİ ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🕗 09:00 Slotu")
            data_09 = slot_analizi_yap(secilen_tarih, '09:00', secilen_rota, df_talep, df_opt)
            slot_kartini_ciz(data_09)

        with col2:
            st.subheader("🕔 17:00 Slotu")
            data_17 = slot_analizi_yap(secilen_tarih, '17:00', secilen_rota, df_talep, df_opt)
            slot_kartini_ciz(data_17)