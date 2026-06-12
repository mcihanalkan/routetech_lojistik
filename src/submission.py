from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_forecast_outputs(
    forecast: pd.DataFrame,
    route_distances: pd.DataFrame,
    stopover_candidates: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_path = output_dir / "forecast_baseline.csv"
    route_distances_path = output_dir / "route_distances.csv"
    stopover_candidates_path = output_dir / "stopover_candidates.csv"

    forecast.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    route_distances.to_csv(route_distances_path, index=False, encoding="utf-8-sig")
    stopover_candidates.to_csv(stopover_candidates_path, index=False, encoding="utf-8-sig")

    return {
        "forecast": forecast_path,
        "route_distances": route_distances_path,
        "stopover_candidates": stopover_candidates_path,
    }
