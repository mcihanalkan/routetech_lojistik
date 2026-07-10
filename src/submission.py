from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_optimization_inputs(
    forecast: pd.DataFrame,
    route_matrix: pd.DataFrame,
    vehicle_costs: pd.DataFrame,
    rental_limits: pd.DataFrame,
    handling_capacity: pd.DataFrame,
    tir_capacity: pd.DataFrame,
    output_dir: Path,
    forecast_source: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = output_dir / "forecast_for_optimization.csv"
    route_matrix_path = output_dir / "route_matrix.csv"
    vehicle_costs_path = output_dir / "vehicle_costs.csv"
    rental_limits_path = output_dir / "rental_limits.csv"
    handling_capacity_path = output_dir / "handling_capacity.csv"
    tir_capacity_path = output_dir / "tir_capacity.csv"
    optimization_json_path = output_dir / "optimization_input.json"

    forecast.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    route_matrix.to_csv(route_matrix_path, index=False, encoding="utf-8-sig")
    vehicle_costs.to_csv(vehicle_costs_path, index=False, encoding="utf-8-sig")
    rental_limits.to_csv(rental_limits_path, index=False, encoding="utf-8-sig")
    handling_capacity.to_csv(handling_capacity_path, index=False, encoding="utf-8-sig")
    tir_capacity.to_csv(tir_capacity_path, index=False, encoding="utf-8-sig")

    payload = {
        "metadata": {
            "forecast_source": forecast_source,
            "forecast_rows": len(forecast),
            "route_matrix_rows": len(route_matrix),
            "vehicle_type_count": int(vehicle_costs["vehicle_type"].nunique()) if not vehicle_costs.empty else 0,
        },
        "forecast": _records(forecast),
        "route_matrix": _records(route_matrix),
        "vehicle_costs": _records(vehicle_costs),
        "rental_limits": _records(rental_limits),
        "handling_capacity": _records(handling_capacity),
        "tir_capacity": _records(tir_capacity),
    }
    optimization_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "forecast": forecast_path,
        "route_matrix": route_matrix_path,
        "vehicle_costs": vehicle_costs_path,
        "rental_limits": rental_limits_path,
        "handling_capacity": handling_capacity_path,
        "tir_capacity": tir_capacity_path,
        "optimization_input_json": optimization_json_path,
    }


def _records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")
