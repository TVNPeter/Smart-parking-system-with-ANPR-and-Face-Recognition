from __future__ import annotations

from math import ceil
from datetime import datetime


def compute_fee(time_in: datetime, time_out: datetime, price_per_hour: float) -> float:
    """Compute parking fee based on elapsed minutes and a unit price.

    Note: "price_per_hour" here acts as a per-minute unit for demos,
    multiplying per-minute duration. Adjust naming/logic if charging truly
    per hour instead.
    """
    delta_seconds = (time_out - time_in).total_seconds()
    duration_minutes = max(0, int(ceil(delta_seconds / 60.0)))
    return float(duration_minutes) * float(price_per_hour)
