# RouteTech Lojistik — Anahat Optimizasyonu (Faz 2: Gelişmiş Çözüm)

RouteTech, TEKNOFEST 2026 yapay zekâ destekli lojistik anahat optimizasyonu yarışma projesidir.
Talep tahmini (ML) + araç/rota/konsolidasyon optimizasyonu (ALNS / OR-Tools CP-SAT) uçtan uca bir pipelineda birleştirilir.

Bu proje **Faz 2 (Gelişmiş Çözüm)** aşamasındadır: saat bazlı zamanlama, elleçleme + tır kapasitesi kısıtları, serbest çok bacaklı konsolidasyon (aktarma/uğrama) ve saatlik SLA cezası destekler.

---

## 1. Mimari / Boru Hattı

```
                 ┌─────────────────────────┐
                 │  src/predict_model/      │  ML tabanlı haftalık desi tahmini
                 │  run_forecast.py         │  (--skip-predict ile atlanabilir)
                 └────────────┬─────────────┘
                              │ alns_payload.json
                 ┌────────────▼─────────────┐
                 │  main.py                 │  Ham Excel verilerini okur (src/data.py),
                 │  src/submission.py       │  rota matrisi + araç/kapasite parametrelerini
                 │                          │  optimizasyon girdisine paketler
                 └────────────┬─────────────┘
                              │ data/outputs/optimization_input.json
                 ┌────────────▼─────────────┐
                 │  src/alns/alns_optimize.py│  VARSAYILAN MOTOR: ALNS (Adaptive Large
                 │  (--engine alns)          │  Neighborhood Search) — çok bacaklı serbest
                 │                           │  konsolidasyon, saat bazlı simülasyon
                 ├───────────────────────────┤
                              │
                 ┌────────────▼─────────────┐
                 │  results/*.xlsx, *.csv   │  Jüri formatında TALEP TAHMİNİ ve TAŞIMA 
                 │                          │   PLANI dosyaları üretiliyor
                 └────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │  dashboard.py (Streamlit) │  Seçilen gün/slotun haritası: rota ve
                 │"                           │  konsolidasyon durumu: (Renk Kodlu)       
                 └───────────────────────────┘
```

### 2. ALNS motoru nasıl çalışır (özet)

`src/alns/alns_engine.py`, her talep parçasını (hat, gün, slot, desi) direkt / uğramalı  (milk-run) / konsolidasyonlu yollara serbestçe atayabilen bir destroy-repair (yıkıcı/onarıcı) arama motorudur:

1. **`dummy_initial_builder`** — Optimizasyonun üzerine inşa edileceği, kapasite/zaman kısıtlarına uyan ilk geçerli başlangıç çözümünü (baseline) hızlıca kurar.
2. **Destroy (Yıkım) Operatörleri** — Mevcut çözümü geliştirilebilmesi için stratejik olarak bozar:
   * **`random_removal:`** Arama uzayında çeşitlilik (diversification) sağlamak için talepleri rastgele söker.
   * **`low_occupancy_removal:`** Çok boş giden, verimsiz araçlardaki yükleri sökerek konsolidasyona (birleştirmeye) zorlar.
   * **`shaw_related_removal:`** Birbirine benzeyen (rota, hacim, zaman) kargoları grup halinde sökerek toplu yerleşim fırsatları arar.
   * **`worst_removal:`** Gecikme (SLA) veya araç maliyeti en yüksek olan "en pahalı" atamaları hedef alıp temizler.
   * **`tm_overload_removal`** Transfer Merkezlerinde (TM) kapasitenin taştığı günlerdeki kargoları sökerek darboğazları açar.
3. **Repair (Onarım) Operatörleri** — Havuza düşen kargoları, maliyeti en aza indirecek şekilde yeniden yerleştirir:
   * **`greedy_repair:`** Kavgayı uzatmaz; kargoyu o anki en ucuz ve ilk uygun boşluğa hemen yerleştirir.
   * **`regret_repair:`** "Bu kargoyu şimdi yerleştirmezsek sonra maliyet ne kadar artar?" (pişmanlık) hesabı yaparak, yer bulması en zor kargolara öncelik verir.
   * **`cpsat_hat_repair:`** Karmaşıklaşan tek bir hat için Google OR-Tools (CP-SAT) çözücüsünü çağırarak, o hat özelinde matematiksel olarak kusursuz (optimal) onarımı yapar.
4. **Son Doğrulama Katmanları** (Arama bittikten sonra tek seferlik çalışır):
   * **`enforce_min_spot_occupancy`** — Minimum %10 spot araç doluluk kuralını dinamik uygulayarak, mikroskobik yük taşıyan araçları iptal eder ve maliyeti tıraşlar.
   * **`enforce_real_capacity_limits / dogrula_gercek_kapasite`** — Kapasite kısıtlarını kaba gün/slot mantığıyla değil, gerçekleşen fiziksel elleçleme anına göre milimetrik hesaplayıp olası son ihlalleri temizleyerek %100 kurallara uygunluğu garanti eder.

---

## 3. Tahmin Modeli Özeti

Talep tahmini birkaç dosyaya bölünmüş bir zincirdir:

* **`run_forecast.py`** — Tüm süreci baştan sona yöneten ana dosya: veriyi okur, modeli eğitir/yükler, tahmini üretir ve sonuçları (`Talep-tahmini.xlsx` vb.) kaydeder.
* **`features.py`** — Geçmiş verilerden modelin anlayacağı sayısal ipuçları çıkarır (geçmiş talep, hafta günü, tatil/kampanya günleri gibi).
* **`forecasters.py`** — Asıl tahmin modelidir. Bu ipuçlarını kullanarak her rota için gelecekteki talebi tahmin eder; kampanya/tatil sonrası ani artışları yakalamak için ayrıca bir düzeltme katmanı vardır.
* **`optimize.py`** — Modelin ayarlarını otomatik olarak en iyi sonucu verecek şekilde bulur.
* **`uncertainty.py`** — Tahmine ek olarak, bu tahminin ne kadar riskli/belirsiz olduğunu hesaplar ve buna göre bir güvenlik payı ekler; talep ID'lerini de üretir.
* **`debug_backtest.py`** — Modeli geçmiş verilerde tekrar tekrar çalıştırıp gerçek sonuçlarla karşılaştırarak test eder.
* **`metrics.py`** — Bu testte modelin ne kadar isabetli olduğunu ölçen hata oranlarını hesaplar.

---

## 4 . Klasör yapısı

```text
KISITLAR.md                     Resmi iş kuralı / kısıt dokümanı (asıl referans)
main.py                         Uçtan uca pipeline giriş noktası
dashboard.py                    Streamlit + Folium interaktif analiz paneli

src/
  config.py                    Yol sabitleri, Excel kolon eşlemeleri, formül sabitleri
  data.py                       Ham Excel girdilerini okuma / standardizasyon
  forecast_payload.py           Tahmin çıktısını ALNS/CP-SAT girdisine çeviren katman
  submission.py                 optimization_input.json üretimi
  alns/
    alns_engine.py              ALNS çekirdeği: State, destroy/repair operatörleri, kapasite doğrulama
    alns_optimize.py            ALNS çalıştırıcı + rapor/Excel üretimi (varsayılan motor)
    cost_model.py                Paylaşılan maliyet formülleri (araç + elleçleme)
    time_model.py                Paylaşılan zaman/SLA/kapasite-dağılım formülleri
  predict_model/
    src/                        Tahmin modelinin kullandığı kaynak dosyalar
    run_forecast.py             Talep tahmini üretimi (ML)
    optimize.py, debug_backtest.py  Tahmin modeli yardımcı/analiz betikleri

data/
  raw/                          Yarışma Excel girdileri (rota matrisi, araç maliyetleri, kapasiteler...)
  static_datas/                 Araç parametreleri, kiralık stok tanımları (CSV)
  outputs/                      Pipeline ara çıktıları (main.py tarafından üretilir)
  processed/                   (rezerve)

results/                        Nihai çıktılar (main.py / alns_optimize.py tarafından üretilir, git'e girmez)
tests/                          pytest test paketi
```

---

## 5. Kurulum

- **Python**: 3.10 – 3.12 önerilir (test edilen ortam: 3.11)
- Zorunlu bağımlılıklar: `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## 6. Çalıştırma

Standart çalıştırma: tahmin modelini de çalıştırır, ardından ALNS ile optimize eder:
```bash
python main.py
```

Daha önce üretilmiş tahmini (src/predict_model/alns_payload.json) tekrar kullanır:
```bash
python main.py --skip-predict
```

Sadece optimizasyon girdisini hazırlar, ALNS/CP-SAT'ı çalıştırmaz:
```bash
python main.py --skip-optimization
```

Çözücü zaman bütçesini değiştir (varsayılan 900 sn):
```bash
python main.py --max-time-seconds 600
```

İnteraktif analiz paneli (bir `main.py` çalıştırması sonrası):

```bash
streamlit run dashboard.py
```

---

## 7. Bilinen açık noktalar
- Çalıştırma sonuçları **deterministik değildir** — ALNS zaman bütçeli (varsayılan 900 sn) ve stokastik (rastgele sayı üretici) bir arama olduğundan, aynı girdiyle (tahmin verileriyle) bile farklı çalıştırmalar farklı (genelde birbirine yakın) toplam maliyet üretebilir. Aşağıdaki örnek çıktı tek bir çalıştırmanın özet istatistikleridir, "kesin sonuç" değildir.

---

## 8. Örnek çalıştırma çıktısı

```
py -3.11 main.py --skip-predict
```

```
================================================================================
OZET ISTATISTIKLER (Faz 2 - ALNS, saat bazli, konsolidasyon destekli)
================================================================================
  Kiralık Arac Maliyeti       :         477,322 TL
      -> Sabit Seyir          :         380,020 TL
      -> Ellecleme (Marjinal) :          97,302 TL
  Spot Arac Maliyeti          :      17,365,371 TL
  SLA Gecikme Cezasi          :       8,884,473 TL
      -> SLA'a ya düşen talep sayısı: 1688 
--------------------------------------------------------------------------------
  OPERASYONEL MALIYET         :      26,727,166 TL
--------------------------------------------------------------------------------
  UGRAMA / KONSOLIDASYON
      -> Cok bacakli talep sayisi (toplam)   :       2189
      -> Gercek ugrama (milk-run) talep say. :        624
      -> Konsolidasyonlu (indir+yukle) talep :       1565
      -> Gercek konsolide bacak sayisi        :       1387
--------------------------------------------------------------------------------
  KAPASITE ASIM CEZALARI (Sanal Maliyetler)
      -> Ellecleme Asim Cezasi:               0 TL
      -> TIR Park Asim Cezasi :               0 TL
--------------------------------------------------------------------------------
  YERLESTIRILEMEYEN TALEP CEZASI (Sanal Maliyet)
      -> Bekleyen satir/desi   :          0 satir /          0 desi
      -> Ceza                  :               0 TL
--------------------------------------------------------------------------------
  TOPLAM MALIYET (Objective)  :      26,727,166 TL
================================================================================
```

---

## 9. Çıktılar

| Dosya | İçerik |
|---|---|
| `data/outputs/optimization_input.json` | ALNS/CP-SAT için paketlenmiş girdi (hatlar, kapasiteler, araç parametreleri) |
| `results/Talep-tahmini.xlsx` | Jüri formatında talep tahmini |
| `results/Tasima_Plani.xlsx` | Jüri şablonuyla birebir resmi Taşıma Planı  | |
| `results/optimization_results.csv` / `.txt` | Ham karar verisi ve özet maliyet raporu |
| `results/capacity_utilization.csv` | TM × gün bazında elleçleme/tır doluluk yüzdeleri |

---
