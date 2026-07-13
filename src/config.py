from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

FORECAST_START = "2026-05-11"
FORECAST_END = "2026-05-17"

RENTAL_COLUMNS = {
    "source": "Çıkış Transfer Merkezi",
    "destination": "Varış Transfer Merkezi",
    "vehicle_count": "Araç sayısı",
    "vehicle_type": "Araç Türü",
}

# Faz 2: saatlik kira + km maliyeti (Araç_Kapasite_Maliyet_Saat.xlsx).
COST_COLUMNS = {
    "vehicle_type": "Araç Adı",
    "capacity": "Kapasite (desi)",
    "rental_hourly": "Kiralık Araç Saatlik Kira (TL)",
    "rental_km": "Kiralık Araç Kilometre Başına Maliyet (TL)",
    "spot_hourly": "Spot Araç Saatlik Kira (TL)",
    "spot_km": "Spot Kilometre Başına Maliyet (TL)",
}

# Faz 2: gerçek km + araç tipine göre seyir süresi + SLA hedefi (sehirler_arasi_lojistik.xlsx).
# Kuş uçuşu (haversine) hesabı artık kullanılmıyor — mesafe/süre doğrudan bu dosyadan okunur.
ROUTE_MATRIX_COLUMNS = {
    "source": "cikis",
    "destination": "varis",
    "distance_km": "mesafe_km",
    "target_delivery_days": "hedef_teslim_gun",
}

# Araç türü -> sehirler_arasi_lojistik.xlsx'teki seyir süresi sütunu (saat).
VEHICLE_DURATION_COLUMNS = {
    "Tır": "Tir_Suresi_Saat",
    "Kamyon": "Kamyon_Suresi_Saat",
    "Hafif Kamyon": "Hafif_Kamyon_Suresi_Saat",
    "Kamyonet": "Kamyonet_Suresi_Saat",
}

# Faz 2: TM başına günlük elleçleme kapasitesi (Ellecleme-kapasite.xlsx).
HANDLING_CAPACITY_COLUMNS = {
    "center": "transfer_merkezi",
    "capacity": "ellecleme_kapasite",
}

# Faz 2: TM başına günlük tır işlem kapasitesi (tir_kapasiteleri.xlsx).
TIR_CAPACITY_COLUMNS = {
    "center": "transfer_merkezi",
    "capacity": "tir_kapasitesi",
}

# Dosya yolları (PROJECT_ROOT'a göre)
PAYLOAD_CSV = PROJECT_ROOT / "src" / "predict_model" / "ortools_payload.csv"

# Model parametreleri (Faz 2 — PDF "Gelişmiş Çözüm Aşaması")
MAX_SPOT = 500
SLA_CEZA_TL_PER_DESI_PER_SAAT = 0.4  # Geciken Desi × Gecikme Süresi (saat) × 0.4 TL
ELLECLEME_DAKIKA_PER_DESI = 0.01  # 0.01 dk/desi elleçleme süresi
MAX_SOLVE_TIME = 300  # saniye

TALEP_ID_PREFIX = "D"
ARAC_ID_PREFIX = "V"