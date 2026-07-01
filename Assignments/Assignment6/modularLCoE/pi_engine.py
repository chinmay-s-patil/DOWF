#!/usr/bin/env python
"""Fast Profitability Index (PI) engine for GPU batched evaluation."""

import cupy as cp
import numpy as np

from config import (
    TURB_COST_PER_MW, SPOT_PRICE_AVG, O_AND_M_PER_MWH,
    RENT_PER_MW, DISCOUNT_RATE, PROJECT_LIFETIME,
    BOP_FIXED_PER_TURBINE, BOP_PER_KM_CABLE, PI_WEIGHT
)


class PIEngine:
    """Pre-computes financial constants for fast batched PI evaluation."""

    def __init__(self, turbine_type_info, n_turbines):
        self.P_rated_MW = turbine_type_info['P_rated_MW']
        self.n_turbines = n_turbines
        self.install_capacity = n_turbines * self.P_rated_MW

        # Pre-compute annuity factor for discounting
        years = np.arange(1, PROJECT_LIFETIME + 1)
        discount_factors = (1 + DISCOUNT_RATE) ** years
        self.annuity_factor = float(np.sum(1.0 / discount_factors))

    def compute_pi(self, aeps_gwh, total_cable_km):
        """
        Vectorized PI computation on GPU.

        Parameters
        ----------
        aeps_gwh : cupy.ndarray (P,)
            Annual Energy Production per particle in GWh.
        total_cable_km : cupy.ndarray (P,)
            Total cable distance per particle in km.

        Returns
        -------
        pi : cupy.ndarray (P,)
            Profitability Index per particle.
        """
        aep_mwh = aeps_gwh * 1e3  # GWh → MWh

        # Investment = turbines + BoP (fixed + cable-variable)
        turbine_cost = TURB_COST_PER_MW * self.install_capacity
        bop_cost = self.n_turbines * BOP_FIXED_PER_TURBINE + BOP_PER_KM_CABLE * total_cable_km
        I = turbine_cost + bop_cost

        # Annual net cash flow
        annual_revenue = aep_mwh * SPOT_PRICE_AVG
        annual_om = aep_mwh * O_AND_M_PER_MWH
        annual_rent = self.install_capacity * RENT_PER_MW
        annual_net = annual_revenue - annual_om - annual_rent

        # Present value of net cash flows
        pv_net = annual_net * self.annuity_factor

        # Profitability Index
        pi = pv_net / I
        return pi

    def compute_pi_single(self, aep_gwh, total_cable_km):
        """Convenience wrapper for a single layout (CPU scalars)."""
        aep_mwh = aep_gwh * 1e3
        turbine_cost = TURB_COST_PER_MW * self.install_capacity
        bop_cost = self.n_turbines * BOP_FIXED_PER_TURBINE + BOP_PER_KM_CABLE * total_cable_km
        I = turbine_cost + bop_cost
        annual_revenue = aep_mwh * SPOT_PRICE_AVG
        annual_om = aep_mwh * O_AND_M_PER_MWH
        annual_rent = self.install_capacity * RENT_PER_MW
        annual_net = annual_revenue - annual_om - annual_rent
        pv_net = annual_net * self.annuity_factor
        return pv_net / I