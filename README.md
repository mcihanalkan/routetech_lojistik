# RouteTech Lojistik - Anahat Optimizasyonu

Yapay zeka destekli lojistik anahat optimizasyonu yarismasi projesi.
ML tabanli talep tahmini ve OR-Tools ile arac/rota optimizasyonu icerir.

## Toplam Maliyet Sonucu

**Toplam Maliyet: 7.019.811 TL**

| Kalem | Tutar (TL) |
|---|---|
| Kiralik arac sabit maliyeti | 802.704 |
| Kiralik ugrama ekstra km maliyeti | 6.566 |
| Spot arac maliyeti (direkt) | 2.643.903 |
| Spot arac maliyeti (ugrama) | 3.317.686 |
| SLA gecikme cezasi | 248.952 |

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
- `src/predict_model/ortools_payload.xlsx` — Juri raporlamasi icin Excel ciktisi
