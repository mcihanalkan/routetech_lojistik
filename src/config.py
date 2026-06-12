from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

FORECAST_START = "2026-05-11"
FORECAST_END = "2026-05-17"

DEMAND_COLUMNS = {
    "source": "Çıkış Transfer Merkezi",
    "destination": "Varış Transfer Merkezi",
    "date": "Tarih",
    "demand": "Toplam Desi",
}

COORD_COLUMNS = {
    "center": "Transfer Merkezi",
    "lat": "Enlem",
    "lon": "Boylam",
}

RENTAL_COLUMNS = {
    "source": "Çıkış Transfer Merkezi",
    "destination": "Varış Transfer Merkezi",
    "vehicle_count": "Araç sayısı",
    "vehicle_type": "Araç Türü",
}

COST_COLUMNS = {
    "vehicle_type": "Araç Adı",
    "capacity": "Kapasite (desi)",
    "rental_fixed": "Kiralık Araç Günlük Kira (TL)",
    "rental_km": "Kiralık Araç Kilometre Başına Maliyet (TL)",
    "spot_fixed": "Spot Araç Sabit Günlük Maliyet (TL)",
    "spot_km": "Spot Kilometre Başına Maliyet (TL)",
}
