from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    COST_COLUMNS,
    COORD_COLUMNS,
    DEMAND_COLUMNS,
    RAW_DATA_DIR,
    RENTAL_COLUMNS,
)


@dataclass(frozen=True)
class RawData:
    demand: pd.DataFrame
    coordinates: pd.DataFrame
    rentals: pd.DataFrame
    costs: pd.DataFrame


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

    demand_path = _find_excel_by_columns(data_dir, list(DEMAND_COLUMNS.values()))
    coord_path = _find_excel_by_columns(data_dir, list(COORD_COLUMNS.values()))
    rental_path = _find_excel_by_columns(data_dir, list(RENTAL_COLUMNS.values()))
    cost_path = _find_excel_by_columns(data_dir, list(COST_COLUMNS.values()))

    demand = _rename_columns(pd.read_excel(demand_path), DEMAND_COLUMNS)
    coordinates = _rename_columns(pd.read_excel(coord_path), COORD_COLUMNS)
    rentals = _rename_columns(pd.read_excel(rental_path), RENTAL_COLUMNS)
    costs = _rename_columns(pd.read_excel(cost_path), COST_COLUMNS)

    demand = _normalize_text_columns(demand, ["source", "destination"])
    coordinates = _normalize_text_columns(coordinates, ["center"])
    rentals = _normalize_text_columns(rentals, ["source", "destination", "vehicle_type"])
    costs = _normalize_text_columns(costs, ["vehicle_type"])

    demand["date"] = pd.to_datetime(demand["date"])
    demand["demand"] = pd.to_numeric(demand["demand"], errors="coerce").fillna(0.0)
    demand["demand"] = demand["demand"].clip(lower=0.0)

    coordinates["lat"] = pd.to_numeric(coordinates["lat"], errors="raise")
    coordinates["lon"] = pd.to_numeric(coordinates["lon"], errors="raise")

    rentals["vehicle_count"] = pd.to_numeric(
        rentals["vehicle_count"], errors="coerce"
    ).fillna(0).astype(int)

    numeric_cost_cols = ["capacity", "rental_fixed", "rental_km", "spot_fixed", "spot_km"]
    for col in numeric_cost_cols:
        costs[col] = pd.to_numeric(costs[col], errors="raise")

    return RawData(
        demand=demand.sort_values(["source", "destination", "date"]).reset_index(drop=True),
        coordinates=coordinates.sort_values("center").reset_index(drop=True),
        rentals=rentals.reset_index(drop=True),
        costs=costs.reset_index(drop=True),
    )


def build_complete_demand_grid(
    demand: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    start_date = pd.Timestamp(start) if start else demand["date"].min()
    end_date = pd.Timestamp(end) if end else demand["date"].max()

    routes = demand[["source", "destination"]].drop_duplicates()
    dates = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    grid = routes.merge(dates, how="cross")

    complete = grid.merge(
        demand[["source", "destination", "date", "demand"]],
        on=["source", "destination", "date"],
        how="left",
    )
    complete["demand"] = complete["demand"].fillna(0.0)
    return complete.sort_values(["source", "destination", "date"]).reset_index(drop=True)
