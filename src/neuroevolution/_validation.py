"""Shared validation for finite simulation coordinates and configuration."""

import math
from collections.abc import Sequence
from numbers import Integral, Real

Point = tuple[float, float]


def finite_float(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def positive_float(value: float, name: str) -> float:
    result = finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def nonnegative_float(value: float, name: str) -> float:
    result = finite_float(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def point(value: Sequence[float], name: str = "Position") -> Point:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two coordinates.")
    return finite_float(value[0], name), finite_float(value[1], name)
