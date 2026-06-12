from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_optimization_inputs(
    forecast: pd.DataFrame,
    route_distances: pd.DataFrame,
    stopover_candidates: pd.DataFrame,
    vehicle_costs: pd.DataFrame,
    rental_limits: pd.DataFrame,
    output_dir: Path,
    forecast_source: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = output_dir / "forecast_for_optimization.csv"
    route_distances_path = output_dir / "route_distances.csv"
    stopover_candidates_path = output_dir / "stopover_candidates.csv"
    vehicle_costs_path = output_dir / "vehicle_costs.csv"
    rental_limits_path = output_dir / "rental_limits.csv"
    optimization_json_path = output_dir / "optimization_input.json"

    forecast.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    route_distances.to_csv(route_distances_path, index=False, encoding="utf-8-sig")
    stopover_candidates.to_csv(stopover_candidates_path, index=False, encoding="utf-8-sig")
    vehicle_costs.to_csv(vehicle_costs_path, index=False, encoding="utf-8-sig")
    rental_limits.to_csv(rental_limits_path, index=False, encoding="utf-8-sig")

    payload = {
        "metadata": {
            "forecast_source": forecast_source,
            "forecast_rows": len(forecast),
            "route_distance_rows": len(route_distances),
            "stopover_candidate_rows": len(stopover_candidates),
            "vehicle_type_count": int(vehicle_costs["vehicle_type"].nunique()) if not vehicle_costs.empty else 0,
        },
        "forecast": _records(forecast),
        "route_distances": _records(route_distances),
        "stopover_candidates": _records(stopover_candidates),
        "vehicle_costs": _records(vehicle_costs),
        "rental_limits": _records(rental_limits),
    }
    optimization_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "forecast": forecast_path,
        "route_distances": route_distances_path,
        "stopover_candidates": stopover_candidates_path,
        "vehicle_costs": vehicle_costs_path,
        "rental_limits": rental_limits_path,
        "optimization_input_json": optimization_json_path,
    }


def _records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")
