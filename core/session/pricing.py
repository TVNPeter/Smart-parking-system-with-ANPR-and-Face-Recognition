from __future__ import annotations

from math import ceil
from datetime import datetime


def compute_fee(time_in: datetime, time_out: datetime, price_per_hour: float = None) -> float:
    """Compute parking fee based on elapsed hours with tiered pricing:
    - First 5 hours: 4000 VND
    - Each additional hour: 2000 VND
    
    Args:
        time_in: Check-in time
        time_out: Check-out time (or current time for real-time calculation)
        price_per_hour: Deprecated parameter (kept for compatibility, not used)
    
    Returns:
        Total fee in VND
    """
    delta_seconds = (time_out - time_in).total_seconds()
    duration_hours = max(0.0, delta_seconds / 3600.0)  # Convert to hours (float)
    
    # Tiered pricing:
    # - First 5 hours: 4000 VND total
    # - Each additional hour after 5 hours: 2000 VND per hour
    if duration_hours <= 5.0:
        return 4000.0
    else:
        additional_hours = duration_hours - 5.0
        return 4000.0 + (ceil(additional_hours) * 2000.0)
