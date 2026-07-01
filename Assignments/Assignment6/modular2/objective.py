#!/usr/bin/env python
"""Batched objective function combining FLORIS AEP, GPU penalties, and cable cost reward."""

import cupy as cp

from config import (
    AEP_WEIGHT, CABLE_COST_WEIGHT, D_WS,
    ROAD_EXCL_NORMAL_BUFFER, ROAD_EXCL_TOFT_BUFFER,
)
from constraints import GPUPenaltyEngine


def make_batched_objective(n_turbines, fmodel_local, wind_rose, site_geom, hh, rotor_diameter, min_spacing):
    """Create a batched objective function for the PSO optimizer."""
    road_excl_normal = rotor_diameter / 2.0 + ROAD_EXCL_NORMAL_BUFFER
    road_excl_toft = rotor_diameter / 2.0 + ROAD_EXCL_TOFT_BUFFER

    penalty_engine = GPUPenaltyEngine(
        site_geom, hh, rotor_diameter, min_spacing,
        road_excl_normal=road_excl_normal,
        road_excl_toft=road_excl_toft
    )

    # Detect if floris_cupy has been patched with the fast layout setter
    _has_fast_layout = hasattr(fmodel_local, 'set_layout_fast')

    def _batched_objective(batch_pos):
        n_particles = batch_pos.shape[0]
        turbine_pos = batch_pos[:, :-1, :]
        substation_pos = batch_pos[:, -1:, :]

        # Sequential AEP evaluation — FLORIS is NOT thread-safe
        aeps = cp.zeros(n_particles)
        for i in range(n_particles):
            layout = turbine_pos[i]
            layout_np = cp.asnumpy(layout)
            x_cpu = layout_np[:, 0].tolist()
            y_cpu = layout_np[:, 1].tolist()

            if _has_fast_layout:
                fmodel_local.set_layout_fast(x_cpu, y_cpu)
            else:
                fmodel_local.set(
                    layout_x=x_cpu, layout_y=y_cpu,
                    wind_data=wind_rose,
                    wind_shear=D_WS
                )
            fmodel_local.run()
            aeps[i] = fmodel_local.get_farm_AEP() / 1e9

        values = aeps * AEP_WEIGHT
        penalty, pen_components = penalty_engine.evaluate(batch_pos, n_turbines)

        diff = turbine_pos - substation_pos
        dists = cp.sqrt(cp.sum(diff ** 2, axis=2))
        total_cable_dist = cp.sum(dists, axis=1)
        cable_reward = CABLE_COST_WEIGHT * total_cable_dist

        score = -values + penalty - cable_reward
        return score, aeps, penalty, total_cable_dist

    return _batched_objective