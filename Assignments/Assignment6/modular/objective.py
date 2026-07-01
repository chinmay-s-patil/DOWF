#!/usr/bin/env python
"""Batched objective function combining FLORIS AEP, GPU penalties, and cable cost reward."""

import numpy as np
import cupy as cp

from config import AEP_WEIGHT, CABLE_COST_WEIGHT
from constraints import GPUPenaltyEngine


def make_batched_objective(n_turbines, fmodel_local, wind_rose, site_geom, hh, rotor_diameter, min_spacing):
    """Create a batched objective function for the PSO optimizer.

    Particle layout: (P, T+1, 2) where last entry is substation position.
    """
    # Compute road exclusion distances dynamically based on rotor diameter
    road_excl_normal = rotor_diameter / 2.0 + 10.0
    road_excl_toft = rotor_diameter / 2.0 + 15.0

    penalty_engine = GPUPenaltyEngine(
        site_geom, hh, rotor_diameter, min_spacing,
        road_excl_normal=road_excl_normal,
        road_excl_toft=road_excl_toft
    )

    def _batched_objective(batch_pos):
        # batch_pos: (P, T+1, 2) cupy array
        # Last entry (index -1) is the substation position
        n_particles = batch_pos.shape[0]

        turbine_pos = batch_pos[:, :-1, :]   # (P, T, 2)
        substation_pos = batch_pos[:, -1:, :]  # (P, 1, 2)

        # 1. FLORIS AEP (CPU, sequential)
        # Only turbines go into FLORIS, not substation
        aeps = cp.zeros(n_particles)
        for i in range(n_particles):
            layout = turbine_pos[i]
            x_cpu = [float(layout[j, 0]) for j in range(n_turbines)]
            y_cpu = [float(layout[j, 1]) for j in range(n_turbines)]
            fmodel_local.set(
                layout_x=x_cpu, layout_y=y_cpu,
                wind_data=wind_rose,
                wind_shear=0.17  # D_WS
            )
            fmodel_local.run()
            aeps[i] = fmodel_local.get_farm_AEP() / 1e9  # GWh

        values = aeps * AEP_WEIGHT

        # 2. GPU Penalties
        penalty, pen_components = penalty_engine.evaluate(batch_pos, n_turbines)

        # 3. Cable Cost Reward (POSITIVE - subtracted from score)
        # Compute sum of distances from substation to each turbine
        # Smaller total distance = better (lower cable costs)
        diff = turbine_pos - substation_pos  # (P, T, 2)
        dists = cp.sqrt(cp.sum(diff ** 2, axis=2))  # (P, T)
        total_cable_dist = cp.sum(dists, axis=1)  # (P,)

        # Reward: we want to MINIMIZE cable distance, so subtract it from score
        # Score = -AEP + penalties - cable_reward
        # Lower score = better
        cable_reward = CABLE_COST_WEIGHT * total_cable_dist

        score = -values + penalty - cable_reward

        # For logging: return cable distance as a "negative penalty" component
        return score, aeps, penalty, total_cable_dist

    return _batched_objective