from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_complete_demand_grid, load_raw_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Kesintisiz talep matrisi kontrol scripti")
    parser.add_argument("--save", action="store_true", help="Sonucu data/processed altina CSV olarak yazar")
    args = parser.parse_args()

    raw = load_raw_data()
    complete = build_complete_demand_grid(raw.demand)

    print("Kesintisiz talep matrisi olusturuldu.")
    print(f"Ham talep satiri: {len(raw.demand):,}")
    print(f"Kesintisiz matris satiri: {len(complete):,}")

    if args.save:
        from src.config import PROCESSED_DATA_DIR

        output = complete.rename(
            columns={
                "date": "Tarih",
                "source": "Çıkış Transfer Merkezi",
                "destination": "Varış Transfer Merkezi",
                "demand": "Toplam Desi",
            }
        )
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DATA_DIR / "kesintisiz_talep_matrisi.csv"
        output.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Cikti: {output_path}")


if __name__ == "__main__":
    main()
