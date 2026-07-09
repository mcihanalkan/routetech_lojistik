# RouteTech Lojistik - Anahat Optimizasyonu

Yapay zeka destekli lojistik anahat optimizasyonu yarışması projesidir.
ML tabanlı talep tahmini ve OR-Tools ile araç/rota optimizasyonu içerir.

## Toplam Maliyet Sonucu

**Toplam Maliyet: 7,176,211 TL**

| Kalem | Tutar (TL) |
|---|---|
| Kiralık araç sabit maliyeti | 802.704 |
| Spot araç maliyeti (direkt) | 2,643,666 |
| Spot araç maliyeti (uğrama) | 3,478,709 |
| SLA gecikme cezası | 251,132 |

| İstatistik | Değer |
|---|---|
| Toplam araç sayısı | 443 |
| Direkt kiralık araç | 98 |
| Direkt spot araç | 162 |
| Uğrama spot araç | 183 |
| Toplam ertelenen yük | 62,783 desi |
| Çözücü süresi | ~451 sn |

## Test Ortamı

- **İşlemci**: AMD Ryzen 7 7735HS (16 thread, ~3.2 GHz)
- **RAM**: 32 GB DDR5
- **İşletim Sistemi**: Windows 11 Home 10.0.26200
- **Cihaz**: ASUS TUF Gaming A15 FA507NV
- **OR-Tools Çözücü Süresi**: 450 sn (max_time_seconds)

## Pipeline

1. **Talep Tahmini** (`src/predict_model/run_forecast.py`): Eğitilmiş `.joblib` model ile haftalık desi tahmini üretir.
2. **Veri Hazırlama** (`main.py`): Excel girdilerini okur, mesafe matrisi hesaplar, optimizasyon girdisini paketler.
3. **Optimizasyon** (`src/optimization.py`): OR-Tools CP-SAT ile araç atama, rota seçimi ve maliyet minimizasyonu yapar. Tüm CPU çekirdeklerini kullanır.

## %10 Minimum Doluluk Kuralı ve Son Gün Yaklaşımı

Spot araçlar için %10 minimum doluluk kuralı uygulanmaktadır. Doluluk oranını karşılamayan yükler o gün taşımaya alınmaz ve bir sonraki güne ertelenir (SLA cezası uygulanır).

Ancak planlamanın son gününde (17 Mayıs) erteleme yapılacak bir sonraki gün bulunmadığı için, %10 doluluk kısıtı devre dışı bırakılır. Bu sayede kalan tüm yükler son gün teslim edilir ve karşılanmamış talep kalmaz.

## SLA Gecikme Cezası

Ertelenen her desi için 4 TL/desi ceza uygulanmaktadır. Bu ceza olmadan optimizer tüm yükleri sürekli ertelemeye bırakarak maliyeti düşürmeye çalışır. Ceza sayesinde erteleme caydırıcı olur ancak gerektiğinde erteleme yapılmasına da izin verir.

## Klasör Yapısı

```text
data/raw/                   Yarışma Excel girdileri
data/outputs/               Pipeline çıktıları (JSON, CSV)
src/data.py                 Veri okuma ve kolon standardizasyonu
src/optimization.py         OR-Tools CP-SAT optimizasyon modeli
src/predict_model/          ML tahmin motoru ve eğitilmiş model (.joblib)
haversine/haversine.py      Haversine mesafe hesabı
```

## Gereksinimler

- **Python**: 3.8 - 3.12 (önerilen: 3.10, 3.11 veya 3.12)
- Bağımlılıkların tam listesi: `requirements.txt`

## Çalıştırma

```bash
pip install -r requirements.txt
python main.py
```

Farklı tarih aralığı:

```bash
python main.py --start 2026-05-11 --end 2026-05-17
```

Mevcut JSON'u tekrar kullanmak için:

```bash
python main.py --skip-predict
```

## Çıktılar

- `data/outputs/optimization_input.json` — Optimizasyon girdisi (hatlar, mesafeler, araç bilgileri)
- `src/predict_model/ortools_payload.csv` — Optimizasyon modeli için talep tahminleme çıktısı
- `results/ortools_payload.xlsx` — Jüri için tahmin talepleme çıktısı (Excel formatında)
- `results/optimization_results.xlsx` — Jüri için araç planlama çıktısı (Excel formatında)
- `results/optimization_results.csv` — Optimizasyon sonuç verisi
- `results/optimization_results.txt` — Detaylı sonuç raporu ve maliyet özeti
