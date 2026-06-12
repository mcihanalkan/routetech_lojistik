from __future__ import annotations

import argparse

from haversine import build_route_distance_table, build_stopover_candidate_table
from src.config import FORECAST_END, FORECAST_START, OUTPUT_DIR
from src.data import load_raw_data
from src.forecast import baseline_forecast
from src.submission import write_forecast_outputs


def run(forecast_start: str = FORECAST_START, forecast_end: str = FORECAST_END) -> dict:
    raw = load_raw_data()
    routes = raw.demand[["source", "destination"]].drop_duplicates()

    forecast = baseline_forecast(raw.demand, forecast_start, forecast_end)
    route_distances = build_route_distance_table(raw.coordinates, routes)
    stopover_candidates = build_stopover_candidate_table(
        raw.coordinates,
        routes,
        tolerance=0.15,
    )

    paths = write_forecast_outputs(
        forecast,
        route_distances,
        stopover_candidates,
        OUTPUT_DIR,
    )

    total_demand = forecast["forecast_demand"].sum() if not forecast.empty else 0.0

    print("RouteTech forecast pipeline completed.")
    print(f"Forecast range: {forecast_start} -> {forecast_end}")
    print(f"Forecast rows: {len(forecast):,}")
    print(f"Route distance rows: {len(route_distances):,}")
    print(f"Stopover candidate rows: {len(stopover_candidates):,}")
    print(f"Total forecast demand desi: {total_demand:,.2f}")
    print("Haversine route distances and stopover candidates are ready for optimization.")
    print("OR-Tools optimization is kept in src/optimization.py for Ahmet to improve.")
    print("Outputs:")
    for label, path in paths.items():
        print(f"  - {label}: {path}")

    return {
        "forecast": forecast,
        "route_distances": route_distances,
        "stopover_candidates": stopover_candidates,
        "paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RouteTech logistics forecast pipeline")
    parser.add_argument("--start", default=FORECAST_START, help="Forecast start date")
    parser.add_argument("--end", default=FORECAST_END, help="Forecast end date")
    args = parser.parse_args()
    run(args.start, args.end)


if __name__ == "__main__":
    main()
