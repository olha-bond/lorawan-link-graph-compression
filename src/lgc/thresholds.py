"""Exact conversion of percentage requirements to integer packet counts."""

from decimal import Decimal, ROUND_CEILING


def minimum_required_count(total: int, percentage: float) -> int:
    """Return the minimum integer count satisfying the percentage threshold."""
    if total < 0:
        raise ValueError("total must be non-negative")

    pct = Decimal(str(percentage))
    if pct < 0 or pct > 100:
        raise ValueError("percentage must be between 0 and 100")
    if total == 0:
        return 0

    required = (Decimal(total) * pct / Decimal(100)).to_integral_value(
        rounding=ROUND_CEILING
    )
    return int(required)
