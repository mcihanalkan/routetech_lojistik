from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ARROW_SEPARATORS = ["\u2192", "->", "=>"]


def run_predict_model(project_root: Path) -> Path:
    predict_dir = project_root / "src" / "predict_model"
    script_path = predict_dir / "run_forecast.py"
    output_path = predict_dir / "alns_payload.json"

    if not script_path.exists():
        raise FileNotFoundError(f"Predict model runner was not found: {script_path}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(predict_dir),
        check=True,
        env=env,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Predict model finished but did not create expected JSON: {output_path}"
        )

    return output_path


def split_route_label(route_label: str) -> tuple[str, str]:
    for separator in ARROW_SEPARATORS:
        if separator in route_label:
            source, destination = route_label.split(separator, 1)
            return source.strip(), destination.strip()
    raise ValueError(f"Route label cannot be split into source/destination: {route_label}")


def load_alns_payload_forecast(payload_path: Path) -> pd.DataFrame:
    payload_path = Path(payload_path)
    with payload_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    demands = payload.get("demands", [])
    if not isinstance(demands, list) or not demands:
        raise ValueError(f"Payload has no demand records: {payload_path}")

    rows = []
    for record in demands:
        route_label = record.get("TM_ID") or record.get("rota") or record.get("route")
        if not route_label:
            raise KeyError(f"Demand record has no route key: {record}")

        source, destination = split_route_label(str(route_label))
        rows.append(
            {
                "date": pd.to_datetime(record["tarih"]),
                "source": source,
                "destination": destination,
                "q10": float(record.get("demand_low", record.get("q10", 0.0))),
                "q50": float(record.get("demand_base", record.get("q50", 0.0))),
                "q90": float(record.get("demand_high", record.get("q90", 0.0))),
                "recommended_demand": float(
                    record.get(
                        "recommended_qty",
                        record.get("recommended_demand", record.get("demand_base", 0.0)),
                    )
                ),
                "safety_buffer": float(record.get("safety_buffer", 0.0)),
                "risk_class": str(record.get("risk_class", "UNKNOWN")),
                "route_label": str(route_label),
            }
        )

    return pd.DataFrame(rows).sort_values(["date", "source", "destination"]).reset_index(drop=True)
