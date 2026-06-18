import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RENTED_STOKS_CSV = BASE_DIR / "data" / "static_datas" / "rented_stoks.csv"
SOLUTION_CSV = BASE_DIR / "results" / "optimization_results.csv"


def kiralik_arac_limit_haritasi_uret(df_kiralik):
    """
    rented_stoks.csv'den hızlı arama sözlüğü (Limit Haritası) üretir.
    Anahtar: (route, vehicle_type) -> Değer: Maksimum Araç Sayısı
    """
    limit_haritasi = {}

    for _, row in df_kiralik.iterrows():
        route = str(row['route']).strip()
        arac_turu = str(row['vehicle_type']).strip()
        arac_sayisi = int(row['quantity'])

        limit_haritasi[(route, arac_turu)] = arac_sayisi

    return limit_haritasi


def emniyet_kilidi_kontrol_gunluk(df_solution, limit_haritasi):
    """
    Optimizasyon çıktısını GÜN GÜN gruplayarak kiralık araç limitlerini denetler.
    Kiralık araç sınırları her yeni gün için baştan sıfırlanır.
    """
    df_kiralik = df_solution[df_solution['Araç_Tipi'].str.startswith('Kiralık')].copy()

    if df_kiralik.empty:
        print("ℹ️  Çıktıda kiralık araç kaydı bulunamadı.")
        return True

    df_kiralik['Saf_Araç_Türü'] = df_kiralik['Araç_Tipi'].str.replace('Kiralık ', '', n=1)
    df_kiralik['Hat'] = df_kiralik['Çıkış_TM'].str.strip() + '-' + df_kiralik['Varış_TM'].str.strip()

    gruplanmis_kararlar = df_kiralik.groupby(
        ['Tarih', 'Hat', 'Saf_Araç_Türü']
    )['Araç_Sayısı'].sum().reset_index()

    for _, row in gruplanmis_kararlar.iterrows():
        tarih = str(row['Tarih']).strip()
        hat = row['Hat']
        arac_turu = row['Saf_Araç_Türü']
        atanan_sayi = int(row['Araç_Sayısı'])

        anahtar = (hat, arac_turu)

        if anahtar in limit_haritasi:
            yasal_limit = limit_haritasi[anahtar]

            if atanan_sayi > yasal_limit:
                raise ValueError(
                    f"🚨 GÜNLÜK KURAL İHLALİ DETEKTÖRÜ:\n"
                    f"Tarih: {tarih}\n"
                    f"Hat: {hat} | Araç Türü: {arac_turu}\n"
                    f"Şirket Sözleşmesindeki Günlük Maksimum Sınır: {yasal_limit}\n"
                    f"Modelin O Gün Atamaya Çalıştığı: {atanan_sayi}\n"
                    f"Kritik ihlal nedeniyle sistem DURDURULDU!"
                )

    print("✅ Emniyet Kilidi Raporu: Tüm günlerin günlük kiralık araç limitleri başarıyla doğrulandı. Sınır aşımı yoktur.")
    return True


if __name__ == "__main__":
    print("🔍 Kiralık Araç Limit Kontrolü Başlatıldı...\n")

    if not RENTED_STOKS_CSV.exists():
        raise FileNotFoundError(f"❌ Kritik Hata: {RENTED_STOKS_CSV} bulunamadı!")

    if not SOLUTION_CSV.exists():
        print(f"⚠️ {SOLUTION_CSV} henüz üretilmemiş. Test atlanıyor.")
    else:
        df_kiralik = pd.read_csv(RENTED_STOKS_CSV)
        df_solution = pd.read_csv(SOLUTION_CSV)

        print(f"✓ Kiralık stok verileri yüklendi: {len(df_kiralik)} kayıt")
        print(f"✓ Optimizasyon çıktısı yüklendi: {len(df_solution)} kayıt\n")

        limit_haritasi = kiralik_arac_limit_haritasi_uret(df_kiralik)
        emniyet_kilidi_kontrol_gunluk(df_solution, limit_haritasi)
