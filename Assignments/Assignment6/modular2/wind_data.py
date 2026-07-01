#!/usr/bin/env python
"""Wind rose creation and management."""

import numpy as np
from scipy.stats import weibull_min
from floris_cupy.wind_data import WindRose  # pyright: ignore[reportMissingImports]

from config import (
    WD_BINS, WB_SCALE, WB_SHAPE, FREQ_WD, WS_BINS, BINSIZE, D_TI, D_WS
)


def create_wind_rose():
    """Build joint frequency table P(WD_i, WS_j) from per-sector Weibull distributions."""
    freq_table = np.zeros((len(WD_BINS), len(WS_BINS)))
    for i, (k, lam) in enumerate(zip(WB_SHAPE, WB_SCALE)):
        p_ws = weibull_min.pdf(WS_BINS, c=k, scale=lam) * BINSIZE
        p_ws /= p_ws.sum()
        freq_table[i, :] = FREQ_WD[i] * p_ws

    wind_rose = WindRose(
        wind_directions=WD_BINS,
        wind_speeds=WS_BINS,
        ti_table=D_TI,
        freq_table=freq_table,
    )
    return wind_rose