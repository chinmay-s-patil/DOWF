#!/usr/bin/env python
"""Main execution script for wind farm layout optimization."""

import os
import sys
import time
import json
import shutil
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MAX_ITER, N_PARTICLES, MAX_WORKERS, SAVE_PLOTS,
    CAPACITY_MIN_MW, CAPACITY_MAX_MW, WAKE_STEERING, PEN_THRESHOLD,
    PLOTS_DIR, HIST_LOGS_DIR, HIST_PLOTS_DIR, OPTIMIZED_LAYOUT_DIR,
    SUBSTATION_EXCLUSION_FACTOR, CABLE_COST_WEIGHT
)
from kml_parser import SiteGeometry
from wind_data import create_wind_rose
from turbine_data import valid_turbine_counts, load_turbine_type
from floris_wrapper import create_floris_model, evaluate_aep_no_yaw, evaluate_aep_with_yaw
from constraints import total_penalty
from objective import make_batched_objective
from plotting import save_layout_plot, save_final_layout_plot, save_history_plot
from pso_optimizer import BatchedGPUParticleSwarm

warnings.filterwarnings("ignore")

SITE_GEOM = None
WIND_ROSE = None


def _format_time(t):
    hrs = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = t % 60
    if hrs > 0:
        return f"{hrs} hr {mins} min {secs:.2f} secs"
    return f"{mins} min {secs:.2f} secs"


def run_optimization(n_turbines, turbine_type_info, seed=42, maxiter=50, disp=True):
    """Run PSO optimization for a given turbine count and type.

    The substation position is the last variable in the particle.
    Particle shape: (n_turbines + 1, 2) where last entry is substation (x, y).
    """
    global SITE_GEOM, WIND_ROSE

    if SITE_GEOM is None:
        SITE_GEOM = SiteGeometry()
    if WIND_ROSE is None:
        WIND_ROSE = create_wind_rose()

    hh = turbine_type_info['hub_height']
    rotor_diameter = turbine_type_info['rotor_diameter']
    min_spacing = turbine_type_info['min_spacing']
    turbine_yaml_path = turbine_type_info['yaml_path']
    turbine_type_name = turbine_type_info['name']

    # Compute road exclusion distances dynamically
    road_excl_normal = rotor_diameter / 2.0 + 10.0
    road_excl_toft = rotor_diameter / 2.0 + 15.0

    # Bounds for (n_turbines + 1) positions: last one is substation
    bounds = [(SITE_GEOM.x_min, SITE_GEOM.x_max) if i % 2 == 0 else (SITE_GEOM.y_min, SITE_GEOM.y_max)
              for i in range(2 * (n_turbines + 1))]

    fmodel_local = create_floris_model(n_turbines, turbine_yaml_path, WIND_ROSE)

    pso = BatchedGPUParticleSwarm(
        n_particles=N_PARTICLES,
        n_turbines=n_turbines,
        bounds=bounds,
        maxiter=maxiter,
        seed=seed,
        disp=disp,
        turbine_type_name=turbine_type_name
    )

    objective_fn = make_batched_objective(n_turbines, fmodel_local, WIND_ROSE,
                                         SITE_GEOM, hh, rotor_diameter, min_spacing)

    # Setup output directories
    out_dir = os.path.join(PLOTS_DIR, turbine_type_name, str(n_turbines))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(HIST_LOGS_DIR, exist_ok=True)
    os.makedirs(HIST_PLOTS_DIR, exist_ok=True)

    pen_file = os.path.join(HIST_LOGS_DIR, f"{turbine_type_name}_{n_turbines}_penalty_hist.txt")
    aep_file = os.path.join(HIST_LOGS_DIR, f"{turbine_type_name}_{n_turbines}_AEP_hist.txt")
    cable_file = os.path.join(HIST_LOGS_DIR, f"{turbine_type_name}_{n_turbines}_cable_hist.txt")
    hist_plot_file = os.path.join(HIST_PLOTS_DIR, f"history_{turbine_type_name}_{n_turbines}.png")

    for fpath in [pen_file, aep_file, cable_file]:
        with open(fpath, "w") as f:
            pass

    local_iters = []
    local_pens = []
    local_aeps = []
    local_cables = []

    substation_excl_radius = rotor_diameter * SUBSTATION_EXCLUSION_FACTOR

    def on_new_best(flat_pos, score, aep, penalty, cable_dist, it,
                    cur_pos, cur_score, cur_aep, cur_penalty, cur_cable):
        # cur_pos is flattened (n_turbines+1)*2 array
        # Extract turbine positions and substation
        n_vars = n_turbines + 1
        pos_list = [(cur_pos[2 * i], cur_pos[2 * i + 1]) for i in range(n_turbines)]
        sub_x = cur_pos[2 * n_turbines]
        sub_y = cur_pos[2 * n_turbines + 1]
        substation_pos = (sub_x, sub_y)

        title = (f"Iter {it} | Score: {cur_score:.1f} | "
                 f"AEP: {cur_aep:.3f} GWh | Pen: {cur_penalty:.3f} | "
                 f"Cable: {cur_cable/1000:.2f} km")
        filename = os.path.join(out_dir, f"iter_{it:04d}.png")
        save_layout_plot(
            pos_list, substation_pos, n_turbines, SITE_GEOM, hh, min_spacing,
            filename, title,
            rotor_diameter=rotor_diameter,
            substation_excl_radius=substation_excl_radius
        )

        with open(pen_file, "a") as f:
            f.write(f"{cur_penalty}\n")
        with open(aep_file, "a") as f:
            f.write(f"{cur_aep}\n")
        with open(cable_file, "a") as f:
            f.write(f"{cur_cable}\n")

        local_iters.append(it)
        local_pens.append(cur_penalty)
        local_aeps.append(cur_aep)
        local_cables.append(cur_cable)

        save_history_plot(
            local_iters, local_pens, local_aeps, local_cables, hist_plot_file,
            f"Live History: {n_turbines} Turbines ({turbine_type_name})"
        )

    print(f"Starting PSO for n={n_turbines}...")
    start_pso_t = time.time()
    flat_best_pos, start_pos_cpu, best_score, history = pso.optimize(objective_fn, callback=on_new_best)
    pso_time = time.time() - start_pso_t

    # Extract turbine positions and substation from flattened result
    best_positions = [(flat_best_pos[2 * i], flat_best_pos[2 * i + 1]) for i in range(n_turbines)]
    best_substation = (flat_best_pos[2 * n_turbines], flat_best_pos[2 * n_turbines + 1])
    start_positions = [(start_pos_cpu[2 * i], start_pos_cpu[2 * i + 1]) for i in range(n_turbines)]
    start_substation = (start_pos_cpu[2 * n_turbines], start_pos_cpu[2 * n_turbines + 1])

    # Evaluate WITHOUT yaw steering
    exact_aep_no_yaw = evaluate_aep_no_yaw(fmodel_local, best_positions, WIND_ROSE)

    # Evaluate WITH yaw steering
    exact_aep_yaw = None
    yaw_optimal = None

    best_pen, pen_breakdown = total_penalty(
        best_positions, best_substation, SITE_GEOM, hh, rotor_diameter, min_spacing,
        road_excl_normal=road_excl_normal, road_excl_toft=road_excl_toft
    )

    if WAKE_STEERING and best_pen < PEN_THRESHOLD:
        exact_aep_yaw, yaw_optimal = evaluate_aep_with_yaw(
            fmodel_local, best_positions, WIND_ROSE
        )
    elif WAKE_STEERING:
        print(f"  [Wake Steering] Skipped — penalty too high ({best_pen:.6f} >= {PEN_THRESHOLD})")

    return {
        'n': n_turbines,
        'turbine_type_name': turbine_type_name,
        'turbine_type_info': turbine_type_info,
        'x0': start_positions,
        'substation_x0': start_substation,
        'positions': best_positions,
        'substation_pos': best_substation,
        'aep_no_yaw': float(exact_aep_no_yaw) if exact_aep_no_yaw is not None else None,
        'aep_yaw': float(exact_aep_yaw) if exact_aep_yaw is not None else None,
        'total_pen': best_pen,
        'pen_breakdown': pen_breakdown,
        'time': pso_time,
        'history': history,
        'yaw_optimal': yaw_optimal,
        'substation_excl_radius': substation_excl_radius
    }


def _run_one(args):
    """Worker function for parallel execution."""
    n, turbine_name = args
    turb_info = load_turbine_type(turbine_name)
    return run_optimization(n, turb_info, maxiter=MAX_ITER)


def save_result_files(result, all_results_dict):
    """Save all output files for a single optimization result."""
    t_name = result['turbine_type_name']
    n_curr = result['n']
    pos = result['positions']
    x0 = result['x0']
    sub_pos = result['substation_pos']
    sub_x0 = result['substation_x0']
    sub_excl = result['substation_excl_radius']
    sx, sy = sub_pos

    # Compute total cable distance
    total_cable = sum(np.sqrt((x - sx)**2 + (y - sy)**2) for x, y in pos)

    # --- Layout text file ---
    layout_file = os.path.join(OPTIMIZED_LAYOUT_DIR, f"{t_name}_{n_curr}_layout.txt")
    with open(layout_file, "w") as f:
        f.write(f"=== Best layout: {n_curr} turbines, type {t_name} ===\n")
        f.write(f"Optimization Time: {_format_time(result['time'])}\n")
        f.write(f"AEP (no yaw): {result['aep_no_yaw']/1e9:.3f} GWh/yr\n")
        if result['aep_yaw'] is not None:
            f.write(f"AEP (yaw)   : {result['aep_yaw']/1e9:.3f} GWh/yr\n")
            f.write(f"Yaw gain    : +{((result['aep_yaw']/result['aep_no_yaw']-1)*100):.1f}%\n")
        f.write(f"Penalty     : {result['total_pen']:.6f}\n")
        f.write(f"Total Cable : {total_cable/1000:.2f} km\n")
        f.write(f"\n=== Optimized Substation ===\n")
        f.write(f"  Position  : x={sy:.4f} m, y={sx:.4f} m\n")
        f.write(f"  Exclusion : {sub_excl:.2f} m\n")
        f.write(f"\n=== Turbine Positions (meters) ===\n")
        for i, (x, y) in enumerate(pos):
            f.write(f"  Turbine {i}: x={x:.4f} m, y={y:.4f} m\n")
        f.write(f"\n=== Start Substation ===\n")
        f.write(f"  Position  : x={sub_x0[1]:.4f} m, y={sub_x0[0]:.4f} m\n")
        f.write(f"\n=== Start Positions (meters) ===\n")
        for i, (x, y) in enumerate(x0):
            f.write(f"  Turbine {i}: x={x:.4f} m, y={y:.4f} m\n")

        f.write(f"\n=== Penalty Breakdown ===\n")
        for key, val in result['pen_breakdown'].items():
            f.write(f"  {key}: {val:.6f}\n")
        if result['pen_breakdown'].get('substation_boundary', 0) > 0:
            f.write(f"  WARNING: Substation is outside the site boundary!\n")

    # Save yaw angles
    if result.get("yaw_optimal") is not None:
        yaw_file = os.path.join(OPTIMIZED_LAYOUT_DIR, f"{t_name}_{n_curr}_yaw_angles.npy")
        np.save(yaw_file, result["yaw_optimal"])

    # --- History plot ---
    hist = result["history"]
    iters = [h["iter"] for h in hist]
    pens = [h["penalty"] for h in hist]
    aeps = [h["aep"] for h in hist]
    cables = [h["cable_dist"] for h in hist]

    save_history_plot(
        iters, pens, aeps, cables,
        os.path.join(HIST_PLOTS_DIR, f"history_{t_name}_{n_curr}.png"),
        f"History: {n_curr} Turbines ({t_name})"
    )

    # --- Raw history data ---
    with open(os.path.join(HIST_LOGS_DIR, f"{t_name}_{n_curr}_penalty_hist.txt"), "w") as f:
        f.write("\n".join(str(p) for p in pens) + "\nEND\n")
    with open(os.path.join(HIST_LOGS_DIR, f"{t_name}_{n_curr}_AEP_hist.txt"), "w") as f:
        f.write("\n".join(str(a) for a in aeps) + "\nEND\n")
    with open(os.path.join(HIST_LOGS_DIR, f"{t_name}_{n_curr}_cable_hist.txt"), "w") as f:
        f.write("\n".join(str(c) for c in cables) + "\nEND\n")

    # --- Times JSON ---
    times_dict = {}
    if os.path.exists("times_taken.json"):
        with open("times_taken.json", "r") as f:
            times_dict = json.load(f)
    if t_name not in times_dict:
        times_dict[t_name] = {}
    times_dict[t_name][n_curr] = result["time"]
    with open("times_taken.json", "w") as f:
        json.dump(times_dict, f, indent=4)


def main():
    """Main entry point."""
    global SITE_GEOM, WIND_ROSE

    SITE_GEOM = SiteGeometry()
    WIND_ROSE = create_wind_rose()

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    if SAVE_PLOTS:
        if os.path.exists(PLOTS_DIR):
            shutil.rmtree(PLOTS_DIR)
        os.makedirs(PLOTS_DIR, exist_ok=True)

    os.makedirs(HIST_PLOTS_DIR, exist_ok=True)
    os.makedirs(HIST_LOGS_DIR, exist_ok=True)
    os.makedirs(OPTIMIZED_LAYOUT_DIR, exist_ok=True)

    print("Running MULTIPROCESS parallel sweep with Spawn...")
    all_results = {}
    start_time = time.time()

    from config import TURBINES

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for t_idx, t_name in enumerate(TURBINES):
            n_min_t, n_max_t, p_mw = valid_turbine_counts(t_name, CAPACITY_MIN_MW, CAPACITY_MAX_MW)
            print(f"  {t_name}: rated {p_mw:.3f} MW -> valid n = [{n_min_t}, {n_max_t}]")
            for n in range(n_min_t, n_max_t + 1):
                futures[ex.submit(_run_one, (n, t_name))] = (n, t_name)

        for fut in as_completed(futures):
            r = fut.result()
            key = f"{r['n']}_{r['turbine_type_name']}"
            all_results[key] = r

            t = r['time']
            time_str = _format_time(t)
            total_cable = sum(np.sqrt((x - r['substation_pos'][0])**2 + 
                                       (y - r['substation_pos'][1])**2) 
                              for x, y in r['positions'])

            aep_base = r['aep_no_yaw'] / 1e9
            aep_yaw_str = ""
            if r['aep_yaw'] is not None:
                aep_yaw_val = r['aep_yaw'] / 1e9
                gain = (r['aep_yaw'] / r['aep_no_yaw'] - 1) * 100
                aep_yaw_str = f" | AEP+yaw={aep_yaw_val:.3f} GWh (+{gain:.1f}%)"

            print(f"  n={r['n']} | type={r['turbine_type_name']} | "
                  f"AEP={aep_base:.3f} GWh{aep_yaw_str} | "
                  f"penalty={r['total_pen']:.6f} | "
                  f"cable={total_cable/1000:.2f} km")
            print(f"  Substation: ({r['substation_pos'][1]:.1f}, {r['substation_pos'][0]:.1f}) m")
            print(f"  Time: {time_str}\n")

            save_result_files(r, all_results)

    total_time = time.time() - start_time
    print(f"--- Complete in {total_time:.2f} s ({(total_time/60):.2f} min) ---")

    # Find best overall
    satisfying = {k: r for k, r in all_results.items() if r["total_pen"] < PEN_THRESHOLD}
    candidates = satisfying if satisfying else all_results
    best_overall = max(candidates.values(), key=lambda r: r["aep_no_yaw"])

    if not satisfying:
        print(f"\nWARNING: No run met penalty < {PEN_THRESHOLD}. Showing best available.")

    # Print summary
    print(f"\n=== BEST: {best_overall['n']} turbines, {best_overall['turbine_type_name']} ===")
    print(f"AEP (no yaw) : {best_overall['aep_no_yaw']/1e9:.3f} GWh/yr")
    if best_overall['aep_yaw'] is not None:
        print(f"AEP (yaw)    : {best_overall['aep_yaw']/1e9:.3f} GWh/yr")
        print(f"Yaw gain     : +{((best_overall['aep_yaw']/best_overall['aep_no_yaw']-1)*100):.1f}%")
    print(f"Penalty      : {best_overall['total_pen']:.6f}")

    sx, sy = best_overall['substation_pos']
    total_cable = sum(np.sqrt((x - sx)**2 + (y - sy)**2) for x, y in best_overall['positions'])
    print(f"\n=== Optimized Substation ===")
    print(f"  Position  : x={sy:.4f} m, y={sx:.4f} m")
    print(f"  Exclusion : {best_overall['substation_excl_radius']:.2f} m")
    print(f"  Total Cable: {total_cable/1000:.2f} km")

    print(f"\n=== Turbine Positions (meters) ===")
    for i, (x, y) in enumerate(best_overall["positions"]):
        print(f"  Turbine {i}: x={x:.4f} m, y={y:.4f} m")

    print(f"\n=== Start Substation ===")
    ssx, ssy = best_overall['substation_x0']
    print(f"  Position  : x={ssy:.4f} m, y={ssx:.4f} m")

    print(f"\n=== Start Positions (meters) ===")
    for i, (x, y) in enumerate(best_overall["x0"]):
        print(f"  Turbine {i}: x={x:.4f} m, y={y:.4f} m")

    # Save best overall file
    best_file = os.path.join(OPTIMIZED_LAYOUT_DIR,
                              f"best_layout_n{best_overall['n']}_"
                              f"{best_overall['turbine_type_name']}.txt")
    with open(best_file, "w") as f:
        f.write(f"=== BEST OVERALL LAYOUT ===\n")
        f.write(f"Turbines: {best_overall['n']}\n")
        f.write(f"Type: {best_overall['turbine_type_name']}\n")
        f.write(f"AEP (no yaw): {best_overall['aep_no_yaw']/1e9:.3f} GWh/yr\n")
        if best_overall['aep_yaw'] is not None:
            f.write(f"AEP (yaw)   : {best_overall['aep_yaw']/1e9:.3f} GWh/yr\n")
            f.write(f"Yaw gain    : +{((best_overall['aep_yaw']/best_overall['aep_no_yaw']-1)*100):.1f}%\n")
        f.write(f"Penalty     : {best_overall['total_pen']:.6f}\n")
        f.write(f"Total Cable : {total_cable/1000:.2f} km\n")
        f.write(f"\n=== Optimized Substation ===\n")
        f.write(f"  Position  : x={sy:.4f} m, y={sx:.4f} m\n")
        f.write(f"  Exclusion : {best_overall['substation_excl_radius']:.2f} m\n")
        f.write(f"\n=== Turbine Positions (meters) ===\n")
        for i, (x, y) in enumerate(best_overall["positions"]):
            f.write(f"  Turbine {i}: x={x:.4f} m, y={y:.4f} m\n")
        f.write(f"\n=== Start Substation ===\n")
        f.write(f"  Position  : x={ssy:.4f} m, y={ssx:.4f} m\n")
        f.write(f"\n=== Start Positions (meters) ===\n")
        for i, (x, y) in enumerate(best_overall["x0"]):
            f.write(f"  Turbine {i}: x={x:.4f} m, y={y:.4f} m\n")

    # Save final plot
    best_turb_info = best_overall['turbine_type_info']
    final_plot_file = os.path.join(OPTIMIZED_LAYOUT_DIR,
                                    f"best_layout_n{best_overall['n']}_"
                                    f"{best_overall['turbine_type_name']}.png")
    title = (f"Best: {best_overall['n']} Turbines ({best_overall['turbine_type_name']}) | "
             f"AEP: {best_overall['aep_no_yaw']/1e9:.2f} GWh | "
             f"Cable: {total_cable/1000:.2f} km")
    if best_overall['aep_yaw'] is not None:
        title += f" | AEP+yaw: {best_overall['aep_yaw']/1e9:.2f} GWh"

    save_final_layout_plot(
        best_overall["positions"], best_overall["x0"],
        best_overall["substation_pos"], best_overall["substation_x0"],
        SITE_GEOM, best_turb_info['hub_height'], best_turb_info['min_spacing'],
        final_plot_file, title,
        rotor_diameter=best_turb_info['rotor_diameter'],
        substation_excl_radius=best_overall['substation_excl_radius']
    )

    print(f"\nSaved plot to {final_plot_file}")
    print(f"Saved data to {best_file}")

    if best_overall.get("yaw_optimal") is not None:
        yaw_npy = os.path.join(OPTIMIZED_LAYOUT_DIR,
                                f"best_layout_n{best_overall['n']}_"
                                f"{best_overall['turbine_type_name']}_yaw.npy")
        np.save(yaw_npy, best_overall["yaw_optimal"])
        print(f"Saved yaw angles to {yaw_npy}")

    print(f"\nBest AEP (no yaw): {best_overall['aep_no_yaw'] / 1e9:.2f} GWh/yr")
    if best_overall['aep_yaw'] is not None:
        print(f"Best AEP (yaw)   : {best_overall['aep_yaw'] / 1e9:.2f} GWh/yr")


if __name__ == '__main__':
    main()