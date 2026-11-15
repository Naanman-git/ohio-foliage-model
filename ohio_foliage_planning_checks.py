"""Optional reference-pack formulas for inspecting the original tier design."""

import numpy as np
from scipy.special import expit


def stress_to_canopy_percent(cumulative_stress, threshold, steepness=0.35):
    """Map stress to the prototype's monotone canopy percentage."""
    if not np.isfinite(steepness) or steepness <= 0:
        raise ValueError("The logistic steepness must be positive and finite.")
    return 100.0 * expit(steepness * (np.asarray(cumulative_stress) - threshold))


def first_order_warming_delay(temperatures, weights, base_temperature, crossing_day,
                              warming_c):
    """Estimate a local timing shift with the set of cold days held fixed.

    Weights combine the photoperiod factor with the optional rainfall factor.
    Crossing day is the first integer crossing index, measured from September 1.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if temperatures.ndim != 1 or temperatures.shape != weights.shape:
        raise ValueError("Expected matching one-dimensional temperatures and weights.")
    if not 0 <= crossing_day < len(temperatures):
        raise ValueError("The approximation requires a crossing within the season.")
    rate = (base_temperature - temperatures[crossing_day]) * weights[crossing_day]
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("The crossing day must have positive daily accumulation.")
    days = slice(0, crossing_day + 1)
    lost_stress = warming_c * weights[days][temperatures[days] < base_temperature].sum()
    return float(lost_stress / rate)
