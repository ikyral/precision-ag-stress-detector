"""Transparent, unit-aware comparison of uniform and variable-rate application."""
from __future__ import annotations

import math
from collections.abc import Mapping


ZONE_NAMES = {1: "Острый стресс", 2: "Умеренный стресс", 3: "Норма"}
DEFAULT_RATES = {1: 150.0, 2: 80.0, 3: 0.0}


def _nonnegative(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _zone_mapping(values, name: str, *, require_all: bool) -> dict[int, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must map zone classes 1, 2, 3 to numbers")
    result = {}
    for key, value in values.items():
        if isinstance(key, bool) or key not in (1, 2, 3, "1", "2", "3"):
            raise ValueError(f"{name}: unsupported zone class {key!r}")
        zone = int(key)
        if zone in result:
            raise ValueError(f"{name}: duplicate zone class {zone}")
        result[zone] = _nonnegative(value, f"{name}[{zone}]")
    if require_all and set(result) != {1, 2, 3}:
        raise ValueError(f"{name} must define all three classes: 1, 2, 3")
    return {zone: result.get(zone, 0.0) for zone in ZONE_NAMES}


def validate_parameters(base_rate_kg_ha, rates_by_zone, price_per_kg, unit):
    if unit not in ("kg", "l"):
        raise ValueError("unit must be 'kg' or 'l'")
    baseline = _nonnegative(base_rate_kg_ha, "base_rate_kg_ha")
    price = _nonnegative(price_per_kg, "price_per_kg")
    rates = _zone_mapping(DEFAULT_RATES if rates_by_zone is None else rates_by_zone,
                          "rates_by_zone", require_all=True)
    return baseline, rates, price


def _finite_sum(values):
    try:
        result = math.fsum(values)
    except (ValueError, OverflowError) as exc:
        raise ValueError("The input values overflow the economic calculation") from exc
    if not math.isfinite(result):
        raise ValueError("The input values overflow the economic calculation")
    return result

def calculate_economics(
    area_by_zone,
    base_rate_kg_ha: float = 150.0,
    rates_by_zone=None,
    price_per_kg: float = 45.0,
    *,
    unit: str = "kg",
) -> dict:
    """Calculate consumption/costs from hectares; retain full precision.

    Historical *_kg parameter names are preserved for the assignment API.
    With unit='l' they mean liters/hectare and currency/liter instead.
    Savings may be negative when the prescription exceeds the baseline.
    saved_percent is None when baseline consumption is zero (undefined).
    """
    base, rates, price = validate_parameters(base_rate_kg_ha, rates_by_zone,
                                             price_per_kg, unit)
    areas = _zone_mapping(area_by_zone, "area_by_zone", require_all=False)
    total_area = _finite_sum(areas.values())
    stressed = _finite_sum(areas[z] for z in (1, 2))
    uniform = total_area * base
    smart = _finite_sum(areas[z] * rates[z] for z in ZONE_NAMES)
    saved = uniform - smart
    report = {
        "total_field_area_ha": total_area,
        "stressed_area_ha": stressed,
        "stressed_area_percent": stressed / total_area * 100 if total_area else 0.0,
        "treated_area_ha": _finite_sum(areas[z] for z in ZONE_NAMES if rates[z] > 0),
        "base_rate": base,
        "price_per_unit": price,
        "currency": "RUB",
        "quantity_unit": unit,
        "rate_unit": f"{unit}/ha",
        "uniform_application_quantity": uniform,
        "smart_application_quantity": smart,
        "saved_quantity": saved,
        f"uniform_application_{unit}": uniform,
        f"smart_application_{unit}": smart,
        f"saved_fertilizer_{unit}": saved,
        "saved_percent": saved / uniform * 100 if uniform else None,
        "uniform_cost_rub": uniform * price,
        "smart_cost_rub": smart * price,
        "saved_money_rub": saved * price,
        "zone_breakdown": [],
    }
    for zone in ZONE_NAMES:
        quantity = areas[zone] * rates[zone]
        report["zone_breakdown"].append({
            "zone_class": zone,
            "zone_name": ZONE_NAMES[zone],
            "area_ha": areas[zone],
            "area_percent": areas[zone] / total_area * 100 if total_area else 0.0,
            "fertilizer_rate": rates[zone],
            f"rate_{unit}_ha": rates[zone],
            "rate_unit": f"{unit}/ha",
            "total_quantity": quantity,
            f"total_{unit}": quantity,
            "quantity_unit": unit,
            "cost_rub": quantity * price,
        })
    if not all(math.isfinite(v) for v in (total_area, uniform, smart, saved,
                                         uniform * price, smart * price, saved * price)):
        raise ValueError("The input values overflow the economic calculation")
    return report
