from pathlib import Path
import pandas as pd
import numpy as np

# =============================================================================
# 1. ORTAM VE DOSYA YOLLARININ AYARLANMASI
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
PAYLOAD_CSV = BASE_DIR / "src" / "predict_model" / "ortools_payload.csv"
SOLUTION_CSV = BASE_DIR / "results" / "optimization_results.csv"

print("🔍 TEKNOFEST Lojistik Pipeline Doğrulama Motoru Başlatıldı...\n")

# =============================================================================
# 2. EBUBEKİR'İN TALEPLERİNİ (GROUND TRUTH) HAFIZAYA ALMA
# =============================================================================
if not PAYLOAD_CSV.exists():
    raise FileNotFoundError(f"❌ Kritik Hata: Ebubekir'in tahmin dosyası bulunamadı: {PAYLOAD_CSV}")

df_payload = pd.read_csv(PAYLOAD_CSV)

src_col = df_payload.columns[1] 
dest_col = df_payload.columns[2] 

toplam_talep_rehberi = {}
for (source, destination), group in df_payload.groupby([src_col, dest_col]):
    hat_key = f"{str(source).strip()}-{str(destination).strip()}"
    toplam_rec_demand = int(group['recommended_demand'].sum())
    toplam_talep_rehberi[hat_key] = toplam_rec_demand

print(f"✓ Ebubekir'in tahmin modelinden {len(toplam_talep_rehberi)} aktif anahat talebi hafızaya alındı.")

# =============================================================================
# 3. AHMET'İN ÇÖZÜM DOSYASINI (ARZ) YÜKLEME VE ANALİZ ETME
# =============================================================================
if SOLUTION_CSV.exists():
    df_solution = pd.read_csv(SOLUTION_CSV)
    print(f"✓ Ahmet'in optimizasyon çıktı dosyası yüklendi: {SOLUTION_CSV}")
else:
    print(f"⚠️ Uyarı: {SOLUTION_CSV} henüz üretilmemiş. Test amaçlı simüle ediliyor.")
    df_solution = pd.DataFrame(columns=['Tarih', 'Çıkış_TM', 'Varış_TM', 'Teslim_Edilen_Desi'])

# =============================================================================
# 4. ERAY'IN ÇEKİRDEK TEST FONKSİYONU: talep_karsilandi_mi
# =============================================================================
def talep_karsilandi_mi(cikis_tm, varis_tm, yedi_gun_toplam_talep):
    """
    Eray'ın bizzat kurguladığı doğrulama fonksiyonu.
    """
    cikis_tm = cikis_tm.strip()
    varis_tm = varis_tm.strip()
    
    filtre = (df_solution['Çıkış_TM'] == cikis_tm) & (df_solution['Varış_TM'] == varis_tm)
    df_hat_ozel = df_solution[filtre]

    yedi_gun_toplam_tasinan = int(df_hat_ozel['Teslim_Edilen_Desi'].sum())
    
    if yedi_gun_toplam_tasinan >= yedi_gun_toplam_talep:
        return True, yedi_gun_toplam_tasinan
    else:
        return False, yedi_gun_toplam_tasinan

# =============================================================================
# 5. TÜM TÜRKİYE FİLOSUNU TARAYAN HATA AVCISI DÖNGÜSÜ (The Validator)
# =============================================================================
def tum_sistemi_test_et():
    basarili_hat_sayisi = 0
    hatali_hat_sayisi = 0
    toplam_eksik_desi = 0
    
    print("\n" + "="*80)
    print("📋 BÜTÜNLÜK VE KARGONUN KORUNUMU TESTİ BAŞLATILIYOR")
    print("="*80)
    
    for hat, talep_desi in toplam_talep_rehberi.items():
        tm1, tm2 = hat.split("-", maxsplit=1)
        
        durum, tasinan_desi = talep_karsilandi_mi(tm1, tm2, talep_desi)
        
        if durum:
            basarili_hat_sayisi += 1
            
            if tasinan_desi > talep_desi:
                fazlalik_hacim = tasinan_desi - talep_desi
                
                print(f"ℹ️  BİLGİ | Hat: {tm1} -> {tm2} | Talepten FAZLA araç kapasitesi açılmış.")
                print(f"   📊 Ebubekir'in İstediği Talep : {talep_desi:>8,} desi")
                print(f"   📊 Ahmet'in Sağladığı Kapasite: {tasinan_desi:>8,} desi")
                print(f"   💡 BOŞTA KALAN GEREKSİZ HACİM : {fazlalik_hacim:>8,} desi (Bütçe israfı olabilir!)\n")
        else:
            hatali_hat_sayisi += 1
            eksik_kargo = talep_desi - tasinan_desi
            toplam_eksik_desi += eksik_kargo
            
            print(f"❌ ALARM | Hat: {tm1} -> {tm2}")
            print(f"   ⚠️ Ebubekir'in İstediği Toplam Talep : {talep_desi:>8,} desi")
            print(f"   ⚠️ Ahmet'in Gerçekte Taşıdığı Yük   : {tasinan_desi:>8,} desi")
            print(f"   🚨 KRİTİK AÇIK: {eksik_kargo:>8,} DESİ KARGO DEPOda UNUTULDU VEYA KAYIP!\n")
            
    print("="*80)
    print("📊 DOĞRULAMA MOTORU ÖZET RAPORU")
    print("="*80)
    print(f" ✅ Kusursuz Karşılanan Hat Sayısı : {basarili_hat_sayisi} hattımız jilet gibi!")
    print(f" ❌ Kuralları İhlal Eden Hat Sayısı : {hatali_hat_sayisi} hatta açık var!")
    
    if hatali_hat_sayisi > 0:
        print(f" 🚨 Toplam Sevk Edilemeyen Yük     : {toplam_eksik_desi:,} desi")
        print("\n🔥 TEST BAŞARISIZ: Ahmet'in optimizasyon kısıtlarında bir sızıntı var!")
        print("🔥 Son gün erteleme yasağı veya yük dengeleme kısıtını (Kısıt C) kontrol etmeli.")
    else:
        print("\n🏆 TEBRİKLER ERAY! Tüm hatlarda kargonun korunumu %100 doğrulandı.")
        print("🏆 Jüri test platformundan tam puan almaya hazırsınız!")
    print("="*80)

# Testi tetikliyoruz
if __name__ == "__main__":
    tum_sistemi_test_et()