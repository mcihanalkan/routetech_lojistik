# RouteTech Lojistik - Anahat Optimizasyonu

Yapay zeka destekli lojistik anahat optimizasyonu yarışması projesidir.
ML tabanli talep tahmini ve OR-Tools ile arac/rota optimizasyonu icerir.

## Toplam Maliyet Sonucu

**Toplam Maliyet: 7.160.246 TL**

| Kalem | Tutar (TL) |
|---|---|
| Kiralik arac sabit maliyeti | 802.704 |
| Kiralik ugrama ekstra km maliyeti | 0 |
| Spot arac maliyeti (direkt) | 2.638.333 |
| Spot arac maliyeti (ugrama) | 3.503.425 |
| SLA gecikme cezasi | 215.784 |

| Istatistik | Deger |
|---|---|
| Toplam arac sayisi | 446 |
| Direkt kiralik arac | 98 |
| Ugrama kiralik arac | 0 |
| Direkt spot arac | 165 |
| Ugrama spot arac | 183 |
| Toplam ertelenen yuk | 53.946 desi |
| Cozucu suresi | ~453 sn |

## Test Ortami

- **Islemci**: AMD Ryzen 7 7735HS (16 thread, ~3.2 GHz)
- **RAM**: 32 GB DDR5
- **Isletim Sistemi**: Windows 11 Home 10.0.26200
- **Cihaz**: ASUS TUF Gaming A15 FA507NV
- **OR-Tools Cozucu Suresi**: 450 sn (max_time_seconds)

## Pipeline

1. **Talep Tahmini** (`src/predict_model/run_forecast.py`): Egitilmis `.joblib` model ile haftalik desi tahmini uretir.
2. **Veri Hazirlama** (`main.py`): Excel girdilerini okur, mesafe matrisi hesaplar, optimizasyon girdisini paketler.
3. **Optimizasyon** (`src/optimization.py`): OR-Tools CP-SAT ile arac atama, rota secimi ve maliyet minimizasyonu yapar. Tum CPU cekirdeklerini kullanir.

## Klasor Yapisi

```text
data/raw/                   Yarisma Excel girdileri
data/outputs/               Pipeline ciktilari (JSON, CSV)
src/data.py                 Veri okuma ve kolon standardizasyonu
src/optimization.py         OR-Tools CP-SAT optimizasyon modeli
src/predict_model/          ML tahmin motoru ve egitilmis model (.joblib)
haversine/haversine.py      Haversine mesafe hesabi
```

## Calistirma

```bash
pip install -r requirements.txt
python main.py
```

Farkli tarih araligi:

```bash
python main.py --start 2026-05-11 --end 2026-05-17
```

Mevcut JSON'u tekrar kullanmak icin:

```bash
python main.py --skip-predict
```

## Ciktilar

- `data/outputs/optimization_input.json` — OR-Tools girdi paketi
- `src/predict_model/ortools_payload.csv` — Algoritma icin tahmin ciktisi
- `results/ortools_payload.xlsx` — Juri raporlamasi icin Excel ciktisi
- `results/optimization_results.xlsx` — Optimizasyon sonuc raporu (Excel)
- `results/optimization_results.csv` — Optimizasyon sonuc verisi
- `results/optimization_results.txt` — Detayli sonuc raporu
