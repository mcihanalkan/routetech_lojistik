from __future__ import annotations

import argparse
from pathlib import Path

from haversine import build_route_distance_table, build_stopover_candidate_table
from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.data import load_raw_data
from src.forecast_payload import load_alns_payload_forecast, run_predict_model
from src.submission import write_optimization_inputs


def run(forecast_json: Path | None = None, skip_predict: bool = False) -> dict:
    raw = load_raw_data()

    if forecast_json is not None:
        payload_path = Path(forecast_json)
        forecast_source = str(payload_path)
    elif skip_predict:
        payload_path = PROJECT_ROOT / "src" / "predict_model" / "alns_payload.json"
        forecast_source = str(payload_path)
    else:
        payload_path = run_predict_model(PROJECT_ROOT)
        forecast_source = str(payload_path)

    forecast = load_alns_payload_forecast(payload_path)
    routes = forecast[["source", "destination"]].drop_duplicates()
    route_distances = build_route_distance_table(raw.coordinates, routes)
    stopover_candidates = build_stopover_candidate_table(raw.coordinates, routes)

    paths = write_optimization_inputs(
        forecast=forecast,
        route_distances=route_distances,
        stopover_candidates=stopover_candidates,
        vehicle_costs=raw.costs,
        rental_limits=raw.rentals,
        output_dir=OUTPUT_DIR,
        forecast_source=forecast_source,
    )

    total_demand = forecast["recommended_demand"].sum() if not forecast.empty else 0.0

    print("RouteTech optimization input pipeline completed.")
    print(f"Forecast source: {forecast_source}")
    print(f"Forecast rows: {len(forecast):,}")
    print(f"Forecast routes: {len(routes):,}")
    print(f"Route distance rows: {len(route_distances):,}")
    print(f"Stopover candidate rows: {len(stopover_candidates):,}")
    print(f"Total recommended demand desi: {total_demand:,.2f}")
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
    parser = argparse.ArgumentParser(description="RouteTech optimization input pipeline")
    parser.add_argument("--forecast-json", type=Path, default=None, help="Use an existing alns_payload.json")
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Do not run predict_model; reuse src/predict_model/alns_payload.json",
    )
    args = parser.parse_args()
    run(args.forecast_json, args.skip_predict)


if __name__ == "__main__":
    main()
