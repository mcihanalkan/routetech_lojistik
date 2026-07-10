from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    COST_COLUMNS,
    HANDLING_CAPACITY_COLUMNS,
    RAW_DATA_DIR,
    RENTAL_COLUMNS,
    ROUTE_MATRIX_COLUMNS,
    TIR_CAPACITY_COLUMNS,
    VEHICLE_DURATION_COLUMNS,
)


@dataclass(frozen=True)
class RawData:
    route_matrix: pd.DataFrame
    rentals: pd.DataFrame
    costs: pd.DataFrame
    handling_capacity: pd.DataFrame
    tir_capacity: pd.DataFrame


def _ascii_key(value: object) -> str:
    text = str(value).strip().casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def _find_excel_by_columns(data_dir: Path, required_columns: list[str]) -> Path:
    required = {_ascii_key(col) for col in required_columns}
    matches: list[Path] = []

    for path in sorted(data_dir.glob("*.xlsx")):
        try:
            columns = pd.read_excel(path, nrows=0).columns
        except Exception:
            continue
        if required.issubset({_ascii_key(col) for col in columns}):
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"Excel file with columns {required_columns} was not found in {data_dir}"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"More than one Excel file matched {required_columns}: {names}")

    return matches[0]


def _rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    by_key = {_ascii_key(col): col for col in df.columns}
    rename_map = {}
    for standard_name, original_name in mapping.items():
        key = _ascii_key(original_name)
        if key not in by_key:
            raise KeyError(f"Column '{original_name}' was not found. Columns: {list(df.columns)}")
        rename_map[by_key[key]] = standard_name
    return df.rename(columns=rename_map)


def _normalize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_raw_data(data_dir: Path = RAW_DATA_DIR) -> RawData:
    data_dir = Path(data_dir)

    route_matrix_path = _find_excel_by_columns(data_dir, list(ROUTE_MATRIX_COLUMNS.values()))
    rental_path = _find_excel_by_columns(data_dir, list(RENTAL_COLUMNS.values()))
    cost_path = _find_excel_by_columns(data_dir, list(COST_COLUMNS.values()))
    handling_path = _find_excel_by_columns(data_dir, list(HANDLING_CAPACITY_COLUMNS.values()))
    tir_path = _find_excel_by_columns(data_dir, list(TIR_CAPACITY_COLUMNS.values()))

    # _rename_columns only touches columns present in the mapping — the per-vehicle-type
    # seyir süresi columns (Tir_Suresi_Saat, ...) pass through under their original names
    # and are looked up later via config.VEHICLE_DURATION_COLUMNS.
    route_matrix = _rename_columns(pd.read_excel(route_matrix_path), ROUTE_MATRIX_COLUMNS)
    rentals = _rename_columns(pd.read_excel(rental_path), RENTAL_COLUMNS)
    costs = _rename_columns(pd.read_excel(cost_path), COST_COLUMNS)
    handling_capacity = _rename_columns(pd.read_excel(handling_path), HANDLING_CAPACITY_COLUMNS)
    tir_capacity = _rename_columns(pd.read_excel(tir_path), TIR_CAPACITY_COLUMNS)

    route_matrix = _normalize_text_columns(route_matrix, ["source", "destination"])
    rentals = _normalize_text_columns(rentals, ["source", "destination", "vehicle_type"])
    costs = _normalize_text_columns(costs, ["vehicle_type"])
    handling_capacity = _normalize_text_columns(handling_capacity, ["center"])
    tir_capacity = _normalize_text_columns(tir_capacity, ["center"])

    route_matrix["distance_km"] = pd.to_numeric(route_matrix["distance_km"], errors="raise")
    route_matrix["target_delivery_days"] = pd.to_numeric(
        route_matrix["target_delivery_days"], errors="raise"
    )
    for duration_col in VEHICLE_DURATION_COLUMNS.values():
        route_matrix[duration_col] = pd.to_numeric(route_matrix[duration_col], errors="raise")

    rentals["vehicle_count"] = pd.to_numeric(
        rentals["vehicle_count"], errors="coerce"
    ).fillna(0).astype(int)

    numeric_cost_cols = ["capacity", "rental_hourly", "rental_km", "spot_hourly", "spot_km"]
    for col in numeric_cost_cols:
        costs[col] = pd.to_numeric(costs[col], errors="raise")

    handling_capacity["capacity"] = pd.to_numeric(handling_capacity["capacity"], errors="raise")
    tir_capacity["capacity"] = pd.to_numeric(tir_capacity["capacity"], errors="raise")

    return RawData(
        route_matrix=route_matrix.sort_values(["source", "destination"]).reset_index(drop=True),
        rentals=rentals.reset_index(drop=True),
        costs=costs.reset_index(drop=True),
        handling_capacity=handling_capacity.sort_values("center").reset_index(drop=True),
        tir_capacity=tir_capacity.sort_values("center").reset_index(drop=True),
    )
