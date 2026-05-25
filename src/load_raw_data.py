import os
import pandas as pd

def load_raw_data(data_folder=os.path.join("data", "raw")):
    """
    data/raw klasöründeki orijinal .xlsx Excel dosyalarını yükler,
    veri tiplerini RAM dostu olacak şekilde optimize eder
    ve temizlenmiş veri çerçeveleri (DataFrame) olarak döndürür.
    """
    
    # Tam dosya yollarını tam olarak belirttiğin isimlere göre eşliyoruz
    talep_path = os.path.join(data_folder, "Desi_talep.xlsx")
    koordinat_path = os.path.join(data_folder, "Koordinatlar.xlsx")
    kiralik_path = os.path.join(data_folder, "Kiralık_Araçlar.xlsx")
    maliyet_path = os.path.join(data_folder, "Araç_Kapasite_Maliyet.xlsx")
    
    print(f"--- Veri Yükleme Başladı (Kaynak: {data_folder}) ---")
    
    # ==========================================
    # 1. DESİ TALEP VERİSİ YÜKLEME & OPTİMİZASYON
    # ==========================================
    # Excel formatında okuyoruz ve Tarih sütununu zaman tipine çeviriyoruz
    df_talep = pd.read_excel(talep_path)
    df_talep["Tarih"] = pd.to_datetime(df_talep["Tarih"])
    
    df_talep["Çıkış Transfer Merkezi"] = df_talep["Çıkış Transfer Merkezi"].astype("category")
    df_talep["Varış Transfer Merkezi"] = df_talep["Varış Transfer Merkezi"].astype("category")
    df_talep["Toplam Desi"] = df_talep["Toplam Desi"].astype("float32")
    print("-> Desi Talep Excel verisi başarıyla yüklendi.")
    
    # ==========================================
    # 2. KOORDİNAT VERİSİ YÜKLEME & OPTİMİZASYON
    # ==========================================
    df_koordinat = pd.read_excel(koordinat_path)
    df_koordinat["Transfer Merkezi"] = df_koordinat["Transfer Merkezi"].astype("category")
    df_koordinat["Enlem"] = df_koordinat["Enlem"].astype("float32")
    df_koordinat["Boylam"] = df_koordinat["Boylam"].astype("float32")
    print("-> Koordinat Excel verisi başarıyla yüklendi.")
    
    # ==========================================
    # 3. KİRALIK ARAÇLAR VERİSİ YÜKLEME & OPTİMİZASYON
    # ==========================================
    df_kiralik = pd.read_excel(kiralik_path)
    df_kiralik["Çıkış Transfer Merkezi"] = df_kiralik["Çıkış Transfer Merkezi"].astype("category")
    df_kiralik["Varış Transfer Merkezi"] = df_kiralik["Varış Transfer Merkezi"].astype("category")
    df_kiralik["Araç Türü"] = df_kiralik["Araç Türü"].astype("category")
    df_kiralik["Araç sayısı"] = df_kiralik["Araç sayısı"].astype("int16")
    print("-> Kiralık Araçlar Excel verisi başarıyla yüklendi.")
    
    # ==========================================
    # 4. KAPASİTE VE MALİYET VERİSİ YÜKLEME & OPTİMİZASYON
    # ==========================================
    df_maliyet = pd.read_excel(maliyet_path)
    df_maliyet["Araç Adı"] = df_maliyet["Araç Adı"].astype("category")
    df_maliyet["Kapasite (desi)"] = df_maliyet["Kapasite (desi)"].astype("int32")
    df_maliyet["Kiralık Araç Günlük Kira (TL)"] = df_maliyet["Kiralık Araç Günlük Kira (TL)"].astype("int32")
    df_maliyet["Kiralık Araç Kilometre Başına Maliyet (TL)"] = df_maliyet["Kiralık Araç Kilometre Başına Maliyet (TL)"].astype("int32")
    df_maliyet["Spot Araç Sabit Günlük Maliyet (TL)"] = df_maliyet["Spot Araç Sabit Günlük Maliyet (TL)"].astype("int32")
    df_maliyet["Spot Kilometre Başına Maliyet (TL)"] = df_maliyet["Spot Kilometre Başına Maliyet (TL)"].astype("int32")
    print("-> Araç Kapasite ve Maliyet Excel verisi başarıyla yüklendi.")
    
    print("--- Tüm Excel Verileri Belleğe Alındı ve Tipler Optimize Edildi ---\n")
    
    return df_talep, df_koordinat, df_kiralik, df_maliyet

# Test Etme Bölümü
if __name__ == "__main__":
    # Terminalde projede bulunduğun ana dizine göre klasörü denetler
    try:
        talep, koordinat, kiralik, maliyet = load_raw_data(data_folder="data/raw")
        print("Sistem Testi Başarılı: Tüm Excel dosyaları sorunsuz okundu!")
    except FileNotFoundError:
        # Eğer test src içinden çağrılırsa bir üst klasöre bakmayı dener
        talep, koordinat, kiralik, maliyet = load_raw_data(data_folder="../data/raw")
        print("Sistem Testi Başarılı: Tüm Excel dosyaları sorunsuz okundu!")