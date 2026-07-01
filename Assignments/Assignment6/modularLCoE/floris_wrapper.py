#!/usr/bin/env python
"""FLORIS model wrapper for AEP evaluation."""

import os
import numpy as np
import cupy as cp
from floris_cupy import FlorisModel  # pyright: ignore[reportMissingImports]
from floris_cupy.optimization.yaw_optimization.yaw_optimizer_pr import YawOptimizationPR  # pyright: ignore[reportMissingImports]

from config import FLORIS_CONFIG, D_WS, WAKE_STEERING, PEN_THRESHOLD


def create_floris_model(n_turbines, turbine_type_path, wind_rose):
    """Create and initialize a FLORIS model."""
    fmodel = FlorisModel(FLORIS_CONFIG)
    abs_turb_path = os.path.abspath(turbine_type_path)
    fmodel.set(
        layout_x=[0.0] * n_turbines,
        layout_y=[0.0] * n_turbines,
        wind_data=wind_rose,
        wind_shear=D_WS,
        turbine_type=[abs_turb_path] * n_turbines
    )
    # Fix the warning: update reference wind height to match turbine hub height
    fmodel.assign_hub_height_to_ref_height()
    return fmodel


def evaluate_aep(fmodel, positions, wind_rose):
    """Evaluate AEP for a single layout. Returns AEP in Wh."""
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    fmodel.set(layout_x=x, layout_y=y, wind_data=wind_rose, wind_shear=D_WS)
    fmodel.run()
    return fmodel.get_farm_AEP()


def evaluate_aep_no_yaw(fmodel, positions, wind_rose):
    """Evaluate AEP without wake steering."""
    return evaluate_aep(fmodel, positions, wind_rose)


def evaluate_aep_with_yaw(fmodel, positions, wind_rose):
    """Evaluate AEP with optimal wake steering (yaw optimization)."""
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    fmodel.set(layout_x=x, layout_y=y, wind_data=wind_rose, wind_shear=D_WS)
    fmodel.run()

    try:
        yaw_opt = YawOptimizationPR(fmodel, display=False)
        df_opt = yaw_opt.optimize()
        yaw_angles_list = df_opt["yaw_angles_opt"].values
        yaw_angles_list = [y.get() if hasattr(y, 'get') else y for y in yaw_angles_list]
        yaw_optimal = np.stack(yaw_angles_list)
        fmodel.set(yaw_angles=yaw_optimal)
        fmodel.run()
        aep_yaw = fmodel.get_farm_AEP()
        return aep_yaw, yaw_optimal
    except Exception as e:
        print(f"  [Wake Steering] Yaw optimization failed: {e}")
        return None, None