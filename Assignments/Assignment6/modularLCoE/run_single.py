#!/usr/bin/env python
"""Run a single optimization (useful for testing/debugging without multiprocessing)."""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAX_ITER, N_PARTICLES, SAVE_PLOTS, CAPACITY_MIN_MW, CAPACITY_MAX_MW
from kml_parser import SiteGeometry
from wind_data import create_wind_rose
from turbine_data import valid_turbine_counts, load_turbine_type
from main import run_optimization, save_result_files

warnings.filterwarnings("ignore")


def main():
    from config import TURBINES

    turbine_name = TURBINES[0]  # IEA_3_4MW
    n_turbines = 17

    print(f"Running single optimization: {n_turbines} x {turbine_name}")
    print(f"Note: Substation position is now an optimization variable!")

    turb_info = load_turbine_type(turbine_name)
    result = run_optimization(n_turbines, turb_info, maxiter=MAX_ITER, disp=True)

    # Save results
    save_result_files(result, {})

    total_cable = sum(((x - result['substation_pos'][0])**2 + (y - result['substation_pos'][1])**2)**0.5 
                      for x, y in result['positions'])

    print(f"\nDone!")
    print(f"AEP (no yaw): {result['aep_no_yaw']/1e9:.3f} GWh")
    if result['aep_yaw'] is not None:
        print(f"AEP (yaw): {result['aep_yaw']/1e9:.3f} GWh")
    print(f"PI: {result['pi']:.3f}")
    print(f"Penalty: {result['total_pen']:.6f}")
    print(f"Optimized Substation: ({result['substation_pos'][1]:.1f}, {result['substation_pos'][0]:.1f}) m")
    print(f"Total Cable: {total_cable/1000:.2f} km")


if __name__ == '__main__':
    main()