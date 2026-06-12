from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data
from haversine import build_center_distance_matrix, build_route_distance_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Haversine mesafe matrisi kontrol scripti")
    parser.add_argument("--save", action="store_true", help="Sonucu data/processed altina CSV olarak yazar")
    args = parser.parse_args()

    raw = load_raw_data()
    matrix = build_center_distance_matrix(raw.coordinates)
    center_names = raw.coordinates["center"].tolist()

    routes = raw.demand[["source", "destination"]].drop_duplicates()
    route_distances = build_route_distance_table(raw.coordinates, routes)

    ist_yalova_km = float(matrix.loc["İstanbul", "Yalova"])
    assert abs(ist_yalova_km - 46.635) <= 0.5

    print("Haversine mesafe hesaplari tamamlandi.")
    print(f"Merkez sayisi: {len(center_names)}")
    print(f"Rota sayisi: {len(route_distances)}")
    print(f"Istanbul-Yalova kontrolu: {ist_yalova_km:.2f} km")

    if args.save:
        from src.config import PROCESSED_DATA_DIR

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        matrix_path = PROCESSED_DATA_DIR / "center_distance_matrix.csv"
        route_path = PROCESSED_DATA_DIR / "route_distances.csv"
        matrix.to_csv(matrix_path, encoding="utf-8-sig")
        route_distances.to_csv(route_path, index=False, encoding="utf-8-sig")
        print(f"Merkez matrisi yazildi: {matrix_path}")
        print(f"Rota mesafeleri yazildi: {route_path}")


if __name__ == "__main__":
    main()
