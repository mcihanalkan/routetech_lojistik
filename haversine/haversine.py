from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_haversine_matrix(latitudes, longitudes):
    """
    Vectorized Haversine distance matrix in kilometers.

    This keeps the original numpy broadcasting approach:
    all center-to-center distances are calculated at once, without Python loops.
    """
    radius_km = 6371.0

    lat = np.radians(np.array(latitudes))
    lon = np.radians(np.array(longitudes))

    lat_col = lat[:, np.newaxis]
    lon_col = lon[:, np.newaxis]
    lat_row = lat[np.newaxis, :]
    lon_row = lon[np.newaxis, :]

    dlat = lat_row - lat_col
    dlon = lon_row - lon_col

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_col) * np.cos(lat_row) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return radius_km * c


def build_center_distance_matrix(coordinates: pd.DataFrame) -> pd.DataFrame:
    """
    Build a square center-to-center distance matrix.

    Expected columns:
    - center
    - lat
    - lon
    """
    matrix = calculate_haversine_matrix(coordinates["lat"], coordinates["lon"])
    centers = coordinates["center"].tolist()
    return pd.DataFrame(matrix, index=centers, columns=centers)


def build_route_distance_table(
    coordinates: pd.DataFrame,
    routes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the full center matrix into source-destination route distances.

    This is the table that forecast/optimization code should use:
    source, destination, distance_km.
    """
    center_matrix = build_center_distance_matrix(coordinates)
    rows = []

    for route in routes[["source", "destination"]].drop_duplicates().itertuples(index=False):
        if route.source not in center_matrix.index:
            raise KeyError(f"Missing coordinate for source transfer center: {route.source}")
        if route.destination not in center_matrix.columns:
            raise KeyError(f"Missing coordinate for destination transfer center: {route.destination}")

        rows.append(
            {
                "source": route.source,
                "destination": route.destination,
                "distance_km": float(center_matrix.loc[route.source, route.destination]),
            }
        )

    return pd.DataFrame(rows)


def build_distance_lookup(route_distances: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Build a fast lookup for optimization code: (source, destination) -> km."""
    return {
        (row.source, row.destination): float(row.distance_km)
        for row in route_distances.itertuples(index=False)
    }


def _load_coordinates_for_script(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "raw" / "Koordinatlar v2.xlsx"
    raw = pd.read_excel(path, sheet_name="Sheet1")
    return raw.rename(
        columns={
            "Transfer Merkezi": "center",
            "Enlem": "lat",
            "Boylam": "lon",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate Haversine distance matrix")
    parser.add_argument("--save", action="store_true", help="Write matrix to data/processed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    coordinates = _load_coordinates_for_script(project_root)
    matrix = build_center_distance_matrix(coordinates)

    ist_yalova_km = float(matrix.loc["İstanbul", "Yalova"])
    expected_km = 46.635
    tolerance_km = 0.5

    print(f"Test edilen hat: Istanbul - Yalova")
    print(f"Hesaplanan mesafe: {ist_yalova_km:.4f} km")

    assert np.isclose(ist_yalova_km, expected_km, atol=tolerance_km), (
        f"KRITIK HATA: Beklenen {expected_km} km, hesaplanan {ist_yalova_km:.2f} km"
    )

    print("Basarili: Haversine matrisi dogru calisiyor.")
    print(f"Merkez sayisi: {len(coordinates)}")
    print(f"Matris boyutu: {matrix.shape}")

    if args.save:
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "center_distance_matrix.csv"
        matrix.to_csv(output_path, encoding="utf-8-sig")
        print(f"Matris kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
