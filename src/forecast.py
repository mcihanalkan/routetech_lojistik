from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import build_complete_demand_grid


def baseline_forecast(
    demand: pd.DataFrame,
    forecast_start: str,
    forecast_end: str,
    lookback_days: int = 28,
) -> pd.DataFrame:
    """Simple route-day forecast that works without optional ML dependencies."""
    history = build_complete_demand_grid(demand)
    history["dow"] = history["date"].dt.dayofweek
    forecast_dates = pd.date_range(forecast_start, forecast_end, freq="D")
    routes = history[["source", "destination"]].drop_duplicates()

    rows = []
    for route in routes.itertuples(index=False):
        route_history = history[
            (history["source"] == route.source)
            & (history["destination"] == route.destination)
        ].sort_values("date")

        recent = route_history.tail(lookback_days)
        nonzero_recent = recent[recent["demand"] > 0]
        fallback = float(nonzero_recent["demand"].median()) if len(nonzero_recent) else 0.0
        residual_scale = float(recent["demand"].std(ddof=0)) if len(recent) else 0.0
        residual_scale = 0.0 if np.isnan(residual_scale) else residual_scale

        for date in forecast_dates:
            same_dow = recent[recent["dow"] == date.dayofweek]
            base = float(same_dow["demand"].median()) if len(same_dow) else fallback
            if base == 0.0:
                base = float(recent["demand"].mean()) if len(recent) else 0.0

            q50 = max(base, 0.0)
            q10 = max(q50 - 0.35 * residual_scale, 0.0)
            q90 = max(q50 + 0.75 * residual_scale, q50)

            rows.append(
                {
                    "date": date,
                    "source": route.source,
                    "destination": route.destination,
                    "q10": round(q10, 2),
                    "q50": round(q50, 2),
                    "q90": round(q90, 2),
                    "forecast_demand": round(q90, 2),
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "source", "destination"]).reset_index(drop=True)
