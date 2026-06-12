# RouteTech Lojistik Projesi

Bu klasor, yapay zeka destekli lojistik anahat optimizasyonu yarismasi icin
veri okuma, mesafe hesaplama ve baseline talep tahmini akisini icerir.

`src/optimization.py` eski haliyle korunmustur. Ahmet bu dosyayi OR-Tools ile
iyilestirdikten sonra ana pipeline'a optimizasyon adimi baglanacaktir.

## Su Anki Ana Akis

`main.py` su islemleri yapar:

1. `data/raw` altindaki Excel dosyalarini kolon semasina gore bulur.
2. Talep, koordinat, kiralik arac ve arac maliyet tablolarini standart kolonlara cevirir.
3. `haversine/haversine.py` ile rota mesafelerini hesaplar.
4. Baseline rota-gun talep tahmini uretir.
5. `data/outputs` altina tahmin ve rota mesafesi CSV'lerini yazar.

Su anda arac optimizasyonu ana akista kapali.

## Klasor Yapisi

```text
data/raw/                   Yarışma Excel girdileri
data/outputs/               main.py çıktıları
src/data.py                 Veri okuma ve kolon standardizasyonu
haversine/haversine.py      Haversine matris ve rota mesafesi hesabi
src/forecast.py             Baseline talep tahmini
src/submission.py           CSV cikti yazimi
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

Ciktilar:

- `data/outputs/forecast_baseline.csv`
- `data/outputs/route_distances.csv`
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
