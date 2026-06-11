import pandas as pd

# 1. Veriyi okuma
df = pd.read_excel('data/raw/Desi_talep.xlsx')
df['Tarih'] = pd.to_datetime(df['Tarih'])
print(f"[-] Ham veri satır sayısı: {df.shape[0]}")

# 2. Her iki sütunu birleştirerek tüm TM'lerin tam listesini çıkar
tum_merkezler = pd.Series(
    pd.concat([df['Çıkış Transfer Merkezi'], df['Varış Transfer Merkezi']]).unique()
)
cikis_merkezleri = pd.DataFrame({'Çıkış Transfer Merkezi': tum_merkezler})
varis_merkezleri = pd.DataFrame({'Varış Transfer Merkezi': tum_merkezler})
print(f"[-] Toplam benzersiz TM sayısı: {len(tum_merkezler)}")

# 3. Tam takvim (1 Ocak - 9 Mayıs)
tarih_araligi = pd.date_range(start="2026-01-01", end="2026-05-09", freq="D")
tarih_df = pd.DataFrame({'Tarih': tarih_araligi})

# 4. Cross join → tam zaman-rota indeksi
tam_indeks = tarih_df.merge(cikis_merkezleri, how='cross').merge(varis_merkezleri, how='cross')
print(f"[-] Gereksinime uygun oluşturulan tam matris satır sayısı: {len(tam_indeks)}")

# 5. Ham veriyle birleştirme
df_merged = tam_indeks.merge(
    df[['Tarih', 'Çıkış Transfer Merkezi', 'Varış Transfer Merkezi', 'Toplam Desi']],
    on=['Tarih', 'Çıkış Transfer Merkezi', 'Varış Transfer Merkezi'],
    how='left'
)

# 6. Karşılığı olmayan boş gün ve rotalara 0.0 ata
df_merged['Toplam Desi'] = df_merged['Toplam Desi'].fillna(0.0)

# 7. Kronolojik ve rota bazlı sıralama
df_merged = df_merged.sort_values(
    ['Çıkış Transfer Merkezi', 'Varış Transfer Merkezi', 'Tarih']
).reset_index(drop=True)

# 8. Kaydet
df_merged.to_excel("kesintisiz_matris.xlsx", index=False)
print(f"[-] Kaydedildi. Final satır: {len(df_merged)}")