# RouteTech Faz 2 (Gelişmiş Çözüm) — Kısıtlar ve İş Kuralları

Bu dosya, jürinin resmi Soru-Cevap dokümanından ve netleştirdiğimiz noktalardan derlenmiştir.
Test yazarken (Eray) buradaki kurallara karşı doğrulama yapılmalı.

---

## 1. Kiralık (Rental) Araçlar

- Talep yetersiz/sıfır olsa bile kiralık araçlar **her gün zorunlu olarak** sefere çıkar.
- Kiralık araçlar dönüş yapmaz. Kendi rotası üzerinde uğrama yapamaz. (Sapmasız olarak sefer yapar, direkt)
- **Ama** kiralık aracın taşıdığı yük, kendi rotası **bittikten sonra** (vardığı TM'de) tamamen normal bir yük gibi **başka bir araca (spot ya da kiralık) bindirilip konsolidasyon zincirine girebilir** — yasak olan, kiralık aracın KENDİSİNİN rotasından sapıp uğrama yapması; yükün nihai varışa ulaşana kadar aktarmalı devam etmesi serbest.
- Kiralık araçların çıkış saatini (gün içinde) biz belirleriz — sabit bir saat dayatılmıyor.
- Kiralık araçlar da tır kapasitesini azaltır, tüketir. (bu kısıt sadece spot araçlara özel değil).
- Kiralık araç maliyeti = **sabit/batık seyir maliyeti** (yol+mesafe, her gün ödenir) + **gerçek yüke bağlı dinamik elleçleme maliyeti** (yüklenen desiye göre değişir).

## 2. Tır Kapasitesi

- Günlük araç sayısı bazlı bir üst sınır var ama bu sadece tır araç türü için.
- Kamyon, Hafif Kamyon, Kamyonet için hiçbir kapasite sınırı yok.
- Kapasite = gelen + giden tır sayısının toplamı (ayrım yapılmaksızın, tek bir günlük havuz).
- Bir aracın aynı TM'de **hareket etmeden** boşaltılıp tekrar yüklenmesi → **1 birim** sayılır.
- Bir aracın TM'den ayrılıp **tekrar dönmesi** (farklı sefer) → **ayrı bir birim** daha tüketir.
- Dönüş yapan bir spot araç, döndüğü TM'de de tır kapasitesi tüketir.

## 3. Elleçleme (Handling) Kapasitesi

- TM başına günlük bir kapasite — hem gelen hem giden elleçlemenin toplamıdır.
- Süre formülü: **desi × 0,01 dakika**.
- Konsolidasyonda: bir TM'de bir yük hem indirilip hem başka bir araca yeniden yüklenirse, **her iki işlem de ayrı ayrı sayılıp toplanır** (örn. 10.000 indirme + 5.000 yükleme = 15.000 desilik kapasite kullanımı o gün).
- Gece yarısını (00:00) **kendi içinde aşan tek bir elleçleme işlemi**, süreye orantılı olarak iki güne **bölünmeli** (örn. 23:30'da başlayan 10.000 desilik işlem: 3.000 bir güne, 7.000 diğer güne). Bu kurala dikkat edelim.
- Çıkış elleçlemesi, talebin tamamlanma anından sonra **herhangi bir zamanda** yapılabilir. İstediğimiz an yapabiliriz.
- Gün-aşırı taşımada: **çıkış günü** için çıkış TM kapasitesi, **varış günü** için varış TM kapasitesi düşülür.

## 4. Zamanlama / Kalkış Saatleri

- Talep, günde **2 sabit saatte** oluşur: **09:00 ve 17:00** — bunlar "talep tamamlanma saati"dir, kalkış saati değildir.
- Araçlar günün **herhangi bir dakikasında** çıkabilir — zaman çözünürlüğü **dakikadır**, sadece 09:00/17:00'e mecbur değil.
- "Mesai saati" kavramı yok — TM'ler 7/24 çalışıyor kabul edilir, elleçleme günün her anında yapılabilir.

## 5. SLA (Teslim Süresi) Kuralları

- SLA **başlangıcı**: talebin orijinal çıkış TM'sindeki tamamlanma anı.
- SLA **bitişi**: aracın varışı değil, nihai varış TM'sindeki elleçlemenin tamamlanma anı.
- SLA süresi: **24 saat (1 gün)** ya da **48 saat (2 gün)** — tam saat cinsinden pencere.
- Gecikme cezası: **geciken desi × gecikme süresi (saat) × 0,4 TL**.
- Gecikme süresi her zaman **bir üst tam saate yuvarlanır** — 1 dakikalık gecikme bile tam 1 saatlik ceza gibi sayılır.

## 6. Konsolidasyon / Aktarma

- Konsolidasyon **zorunlu değil**, opsiyoneldir. — direkt sevkiyat da her zaman geçerli bir seçenek.
- Bir araca farklı varış noktalarına giden farklı yükler birlikte yüklenebilir.
- Spot araçlar milk-run (kısmi yük bırakıp kalanla devam etme) yapabilir.
- Kiralık araçlar kendi rotasında milk-run/uğrama yapamaz (bkz. Bölüm 1 — rotası bittikten sonra yükü aktarmaya girebilir, bu farklı bir şey).
- Bir spot aracın gün içinde yapabileceği sefer sayısında (fiziksel zaman/kapasite kısıtları dışında) üst sınır yok.
- Spot araçlar dönüş yapabilir, ama boş dönen spot araçları modellemek/hesaba katmak ZORUNLU DEĞİL — boş araç döndürmeyin (boş dönüş senaryosunu simüle etmeye gerek yok, sadece dolu sefer/dolu dönüşleri düşünün).
- Kiralık araçlarda dönüş yoktur (tekrar: Bölüm 1'le tutarlı, ayrıca burada da vurgulanıyor).

## 7. Araç Maliyeti Formülü

```
Toplam Araç Maliyeti = (Saatlik Kiralama Maliyeti × Kullanım Süresi) + (Kat Edilen Mesafe × Km Başı Maliyet)
```

- **Kullanım Süresi** = çıkış elleçleme süresi + yol (seyir) süresi + varış elleçleme süresi + (varsa) bekleme süresi — **hepsi toplanır**.
- Somut örnek (jüri referansı): 10.000 desi, 5 saatlik yol →
  - Çıkış elleçleme: 10.000 × 0,01 = 100 dk
  - Yol: 5 saat = 300 dk
  - Varış elleçleme: 100 dk
  - **Toplam kullanım süresi = 500 dk** (bekleme yoksa)
- Bekleme örneği: elleçleme kapasitesi yüzünden 1 saat beklenirse → toplam **560 dk**.

## 7b. Süre Yuvarlama Kuralı

- Araç çıkış/varış saat bilgileri **dakika (HH:MM)** cinsinden verilmeli.
- **Seyir (yol) süresi**, en yakın büyük tam dakikaya (yukarı/ceiling) yuvarlanmalı — normal yuvarlama (en yakına) DEĞİL, her zaman **yukarı**.
  - Örnek: İstanbul→Yalova, saat 10:14'te Tır çıkıyor, transfer süresi 0,92 saat = 55,2 dakika → **56 dakikaya** yuvarlanır (55'e değil).
- **Elleçleme süresi** de aynı şekilde (desi × 0,01 dk hesaplandıktan sonra) en yakın **büyük** tam dakikaya yuvarlanmalı.
- Bu, hem varış zamanı hesaplarını (SLA'yı etkiler) hem elleçleme tamamlanma anını (kapasite/SLA'yı etkiler) doğrudan etkiliyor.

## 8. %10 Minimum Doluluk Kuralı

- Temel İşlevli Çözüm (MVP/Faz-1) aşamasında vardı.
- **Gelişmiş Çözüm'de (Faz-2) KALDIRILDI** — böyle bir kural yok. Küçük/verimsiz sevkiyatlar reddedilmez, sadece gecikirse SLA cezası öder.

## 9. Talep Tahmini

- Tahmin ufku: **29 Haziran 2026 09:00 — 5 Temmuz 2026 17:00** arası.
- Her **(gün, çıkış TM, varış TM, saat)** kombinasyonu **ayrı bir satır** olmalı (09:00 ve 17:00 birleştirilmez).
- **Geçmiş veride hiç talep görülmemiş** bir TM çifti için tahmin **üretilmemeli**.
- Sıfıra yakın (örn. 0,5 altı) tahminler de dosyadan **çıkarılmamalı**, her satır sunulmalı.
- Talep ID formatı: `D00001, D00002, ...` sıralı. Bölünen taleplerde `D00001-1, D00001-2` (ikinci bölünmede `D00001-1-1`).
- Optimizasyon için zaman sınırı yok — 5 Temmuz'da tahminlenen bir talep 7 Temmuz'da teslim edilebilir, SLA cezası gerçek zamana göre hesaplanır.

## 10. Veri/Çıktı Formatı

- Taşıma Planı çıktısında saat formatı: SS:DD yeterli (saniyeli olmasına gerek yok).
- Talep-Tahmini.xlsx ile Taşıma-Planı içindeki Talep ID'ler birebir eşleşmeli.