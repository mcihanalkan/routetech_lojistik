from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.data import load_raw_data
from src.forecast_payload import load_alns_payload_forecast, run_predict_model
from src.submission import write_optimization_inputs


def run(
    forecast_json: Path | None = None,
    skip_predict: bool = False,
    skip_optimization: bool = False,
    max_time_seconds: float = 300.0,
    engine: str = "alns",
) -> dict:
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

    # Faz 2: mesafe/süre/SLA artık kuş uçuşu (haversine) değil, sehirler_arasi_lojistik.xlsx'teki
    # gerçek route_matrix'ten geliyor — koordinat tabanlı hesap tamamen kaldırıldı.
    paths = write_optimization_inputs(
        forecast=forecast,
        route_matrix=raw.route_matrix,
        vehicle_costs=raw.costs,
        rental_limits=raw.rentals,
        handling_capacity=raw.handling_capacity,
        tir_capacity=raw.tir_capacity,
        output_dir=OUTPUT_DIR,
        forecast_source=forecast_source,
    )

    total_demand = forecast["recommended_demand"].sum() if not forecast.empty else 0.0

    print("RouteTech optimization input pipeline completed.")
    print(f"Forecast source: {forecast_source}")
    print(f"Forecast rows: {len(forecast):,}")
    print(f"Route matrix rows: {len(raw.route_matrix):,}")
    print(f"Total recommended demand desi: {total_demand:,.2f}")
    print("Outputs:")
    for label, path in paths.items():
        print(f"  - {label}: {path}")

    optimization_result = None
    if not skip_optimization:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["ROUTETECH_OPTIMIZATION_INPUT"] = str(paths["optimization_input_json"])
        env["ROUTETECH_MAX_TIME_SECONDS"] = str(max_time_seconds)
        env["ROUTETECH_LOG_SEARCH_PROGRESS"] = "1"
        engine_script = "alns_optimize.py" if engine == "alns" else "optimization.py"
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / engine_script)],
            cwd=str(PROJECT_ROOT),
            check=True,
            env=env,
        )
        optimization_result = {
            "results_txt": PROJECT_ROOT / "results" / "optimization_results.txt",
            "decisions_csv": PROJECT_ROOT / "results" / "optimization_results.csv",
        }

        # NOT: tests/*.py doğrulama betikleri henüz Faz-2 kısıtlarına (elleçleme/tır
        # kapasitesi, saatlik SLA) göre güncellenmedi — bilinçli olarak burada
        # çağrılmıyorlar (bkz. plan: Stage E). Eski Faz-1 mantığıyla şu anki çıktı
        # şeması uyumsuz olduğu için çağırmak yanıltıcı sonuç verirdi.

    return {
        "forecast": forecast,
        "paths": paths,
        "optimization": optimization_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RouteTech optimization input pipeline")
    parser.add_argument("--forecast-json", type=Path, default=None, help="Use an existing alns_payload.json")
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Do not run predict_model; reuse src/predict_model/alns_payload.json",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Only prepare optimization inputs; do not run the OR-Tools solver",
    )
    parser.add_argument(
        "--max-time-seconds",
        type=float,
        default=450.0,
        help="Maximum OR-Tools solve time in seconds",
    )
    parser.add_argument(
        "--engine",
        choices=["alns", "cpsat"],
        default="alns",
        help="Optimization engine: alns (ana motor, konsolidasyon destekli) veya cpsat (Faz-2 direkt-hat modeli)",
    )
    args = parser.parse_args()
    run(args.forecast_json, args.skip_predict, args.skip_optimization, args.max_time_seconds, args.engine)


if __name__ == "__main__":
    main()
