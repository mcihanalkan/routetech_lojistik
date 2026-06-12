# RouteTech Lojistik Projesi

Bu klasor, yapay zeka destekli lojistik anahat optimizasyonu yarismasi icin
veri okuma, mesafe hesaplama ve baseline talep tahmini akisini icerir.

`src/optimization.py` eski haliyle korunmustur. Ahmet bu dosyayi OR-Tools ile
iyilestirdikten sonra ana pipeline'a optimizasyon adimi baglanacaktir.
`main.py` optimizasyonu calistirmaz; optimizasyon motoru icin girdi paketini hazirlar.

## Su Anki Ana Akis

`main.py` su islemleri yapar:

1. `data/raw` altindaki Excel dosyalarini kolon semasina gore bulur.
2. Talep, koordinat, kiralik arac ve arac maliyet tablolarini standart kolonlara cevirir.
3. `haversine/haversine.py` ile rota mesafelerini hesaplar.
4. `src/predict_model/run_forecast.py` motorunu calistirir ve yeni `alns_payload.json` uretir.
5. Yeni JSON'daki rotalara gore mesafe ve ugrama adaylarini uretir.
6. Arac maliyetleri ve kiralik limitlerini standart formatta yazar.
7. OR-Tools icin `optimization_input.json` paketini olusturur.

Tahmin JSON'u ana girdidir. Baseline forecast ana akista kullanilmaz.

Su anda arac optimizasyonu ana akista kapali.

## Klasor Yapisi

```text
data/raw/                   Yarışma Excel girdileri
data/outputs/               main.py çıktıları
src/data.py                 Veri okuma ve kolon standardizasyonu
haversine/haversine.py      Haversine matris ve rota mesafesi hesabi
src/forecast.py             Baseline talep tahmini
src/submission.py           CSV cikti yazimi
src/forecast_payload.py     alns_payload.json -> forecast DataFrame donusumu
src/optimization.py         Korunan eski OR-Tools deneme modeli
src/predict_model/          Korunan gelismis ML tahmin calismasi
scripts/haversine_matrix.py Haversine kontrol scripti
scripts/build_complete_matrix.py Kesintisiz matris kontrol scripti
```

## Calistirma

```bash
python main.py
```

Farkli tarih araligi:

```bash
python main.py --start 2026-05-11 --end 2026-05-17
```

Calistirma:

```bash
python main.py
```

Bu komut once predict_model motorunu calistirir, sonra olusan JSON'u optimizasyon
girdilerine cevirir.

Mevcut bir JSON'u tekrar kullanmak istersen:

```bash
python main.py --forecast-json src/predict_model/alns_payload.json
```

Predict motorunu calistirmadan son JSON'u tekrar kullanmak istersen:

```bash
python main.py --skip-predict
```

Ciktilar:

- `data/outputs/forecast_for_optimization.csv`
- `data/outputs/route_distances.csv`
- `data/outputs/stopover_candidates.csv`
- `data/outputs/vehicle_costs.csv`
- `data/outputs/rental_limits.csv`
- `data/outputs/optimization_input.json`


## Ugrama ve Konsolidasyon Ayrimi

`stopover_candidates.csv` sadece geometrik olarak makul ugrama adaylarini verir.
Bir satir `source -> stopover -> destination` rotasinin direkt rotaya gore tolerans
icinde kaldigini gosterir. Bu, farkli araclarin yuklerini stopover noktasinda
birlestirme izni degildir. OR-Tools tarafinda bu tablo yalnizca ayni aracin
multi-stop guzergah secenegi olarak kullanilmalidir.
- `data/outputs/route_distances.csv`

Yardimci kontrol scriptleri:

```bash
python scripts/haversine_matrix.py
python scripts/build_complete_matrix.py
```

Bu scriptler varsayilan olarak sadece kontrol ciktisi basar. Dosya uretmek icin:

```bash
python scripts/haversine_matrix.py --save
python scripts/build_complete_matrix.py --save
```

## Bagimliliklar

Temel akis:

```bash
pip install -r requirements.txt
```

Opsiyonel ML paketleri:

```bash
pip install -r requirements-ml.txt
```
