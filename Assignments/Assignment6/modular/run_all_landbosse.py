#!/usr/bin/env python
"""
Batch LandBOSSE + FLORIS analysis for all layout files in optimizedLayout/.

For each layout .txt file:
  1) Print the filename
  2) Parse turbine & substation positions
  3) Configure & run LandBOSSE  (BoP costs)
  4) Run FLORIS                 (AEP, wake losses, capacity factor)
  5) Compute financials         (BoP, LCOE, NPV, PI, IRR)
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import weibull_min
from scipy.optimize import newton

# ── Suppress noisy warnings ───────────────────────────────────────────
warnings.filterwarnings("ignore")

# ── Working directory ─────────────────────────────────────────────────
BASE_DIR = r"/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/modular"
os.chdir(BASE_DIR)

# ── Imports that need cwd set first ───────────────────────────────────
from landbosse.main_function import run_landbosse          # type: ignore
from floris import FlorisModel
from floris.wind_data import WindRose

# ══════════════════════════════════════════════════════════════════════
#  FIXED PARAMETERS (from the assignment / randTry.ipynb)
# ══════════════════════════════════════════════════════════════════════

d_ws               = 0.17
d_ti               = 12 / 100
d_fuel_cost         = 9.5
d_construction_time = 12        # months
d_rent              = 15_000    # $/MW-yr
d_o_and_m           = 0.012     # $/kWh  →  $12/MWh
d_discount          = 3.6 / 100
d_life_time         = 20
TURB_COST_PER_MW    = 1.3e6

rho = 1.225  # kg/m^3

# Interconnect – same for all layouts (from notebook)
lattitude_degree_to_km    = 111
longitude_degree_to_km_55 = 63

def get_ref_WGS84(latt, long):
    lattLB = 55 + 14/60 + 39.2/3600
    longLB = 9  +  0/60 + 41.3/3600
    latt = (latt - lattLB) * lattitude_degree_to_km * 1000
    long = (long - longLB) * longitude_degree_to_km_55 * 1000
    return latt, long

def point_to_line_distance_meters(pl1, pl2, point):
    p1, p2, p = np.array(pl1), np.array(pl2), np.array(point)
    lv = p2 - p1
    pv = p  - p1
    cross = abs(lv[0]*pv[1] - lv[1]*pv[0])
    return cross / np.linalg.norm(lv)

pl1 = get_ref_WGS84(55.26958668425792, 9.194729595538924)
pl2 = get_ref_WGS84(55.24081570901905, 9.208746443354379)
METERS_TO_MILES = 0.000621371

# ── Wind Rose ─────────────────────────────────────────────────────────
binsize = 3
WD_BINS  = np.array([0,30,60,90,120,150,180,210,240,270,300,330], dtype=float)
WB_SCALE = np.array([9.785,8.284,8.721,9.633,10.114,8.340,
                      8.936,10.759,11.710,11.363,10.682,8.965])
WB_SHAPE = np.array([2.306,2.089,1.888,1.935,1.945,1.902,
                      1.909,1.910,1.968,2.049,2.064,1.928])
FREQ_WD  = np.array([14.71,6.09,6.16,8.17,9.58,6.05,
                      5.34,7.27,8.00,14.60,7.78,6.25]) / 100.0
WS_BINS  = np.arange(3.0, 26.0, 1.0)

freq_table = np.zeros((len(WD_BINS), len(WS_BINS)))
for i, (k, lam) in enumerate(zip(WB_SHAPE, WB_SCALE)):
    p_ws = weibull_min.pdf(WS_BINS, c=k, scale=lam) * binsize
    p_ws /= p_ws.sum()
    freq_table[i, :] = FREQ_WD[i] * p_ws

wind_rose = WindRose(
    wind_directions=WD_BINS,
    wind_speeds=WS_BINS,
    ti_table=d_ti,
    freq_table=freq_table,
)

# Spot prices
years = np.arange(2021, 2041)
A_denmark = np.array([
    100.5, 249.8, 154.3, 74.3, 57.7,
    53.6,  63.1,  51.2,  38.9, 35.9,
] + [32.6] * 10)
SPOT_C = 1.235
SPOT_K = 0.04295

# ══════════════════════════════════════════════════════════════════════
#  TURBINE TYPE → PARAMETER LOOKUP (from YAML files)
# ══════════════════════════════════════════════════════════════════════

TURBINE_PARAMS = {}

def _load_turbine_params():
    """Read each turbine YAML to get rated power, rotor D, hub H, V_rated, Ct_rated."""
    turbine_dir = os.path.join(BASE_DIR, "turbineData")
    for yf in glob.glob(os.path.join(turbine_dir, "*.yaml")):
        with open(yf) as f:
            data = yaml.safe_load(f)
        ttype = data["turbine_type"]
        D     = data["rotor_diameter"]
        H     = data["hub_height"]

        ws_list  = data["power_thrust_table"]["wind_speed"]
        pwr_list = data["power_thrust_table"]["power"]
        ct_list  = data["power_thrust_table"]["thrust_coefficient"]

        # Rated power = max of power curve (kW)
        P_rated = max(pwr_list)

        # V_rated = first wind speed where power reaches P_rated
        V_rated = None
        for ws, pw in zip(ws_list, pwr_list):
            if pw >= P_rated * 0.999:
                V_rated = ws
                break

        # Ct at rated
        Ct_rated = None
        for ws, ct in zip(ws_list, ct_list):
            if ws >= V_rated - 0.01:
                Ct_rated = ct
                break

        TURBINE_PARAMS[ttype] = {
            "P_rated_kW": P_rated,
            "P_rated_MW": P_rated / 1000,
            "D": D,
            "H": H,
            "V_rated": V_rated,
            "Ct_rated": Ct_rated,
        }

_load_turbine_params()

# ══════════════════════════════════════════════════════════════════════
#  FILE PARSING
# ══════════════════════════════════════════════════════════════════════

def parse_layout(filepath):
    """Parse a layout .txt file → (turbine_lines, substation_lines, turbine_type)."""
    def linetoxy(s):
        s = s.replace("x=", "").replace("y=", "").replace(" m", "")
        return [float(i) for i in s.replace(",", "").split(":")[1].strip().split()]

    substation_lines = []
    turbine_lines    = []
    turbine_type     = None

    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            # Extract turbine type from header
            if line.startswith("=== Best layout:") and "type" in line:
                turbine_type = line.split("type")[-1].strip().rstrip("=").strip()
            if "Substation" in line and "Start" not in line and "Position" in "".join([f.readline() for _ in range(0)]):
                pass  # handled below

    # Re-read for positions
    substation_lines = []
    turbine_lines    = []
    with open(filepath) as f:
        line = f.readline().strip()
        # Get turbine_type from first line
        if "type" in line:
            turbine_type = line.split("type")[-1].strip().rstrip("=").strip()
        while True:
            if "Optimized Substation" in line or (line.startswith("=== Optimized Substation")):
                line = f.readline().strip()  # Position line
                substation_lines.append(linetoxy(line))
            if "Turbine" in line and "Positions" not in line and "Start" not in line and "===" not in line:
                turbine_lines.append(linetoxy(line))
            line = f.readline().strip()
            if not line or "Start" in line:
                break

    return turbine_lines, substation_lines, turbine_type


def parse_layout_v2(filepath):
    """Robust parser for layout files."""
    def linetoxy(s):
        s = s.replace("x=", "").replace("y=", "").replace(" m", "")
        return [float(i) for i in s.replace(",", "").split(":")[1].strip().split()]

    turbine_type = None
    substation_lines = []
    turbine_lines = []

    with open(filepath) as f:
        lines = f.readlines()

    in_optimized_turbines = False
    in_optimized_sub      = False
    past_start            = False

    for i, raw in enumerate(lines):
        line = raw.strip()

        # Get turbine type from header
        if "=== Best layout:" in line and "type" in line:
            turbine_type = line.split("type")[-1].strip().rstrip(" =")

        # Detect section boundaries
        if "=== Optimized Substation ===" in line:
            in_optimized_sub = True
            continue
        if "=== Turbine Positions" in line:
            in_optimized_sub = False
            in_optimized_turbines = True
            continue
        if "=== Start" in line:
            past_start = True
            in_optimized_turbines = False
            in_optimized_sub = False
            continue

        if past_start:
            continue

        # Parse substation position
        if in_optimized_sub and "Position" in line:
            substation_lines.append(linetoxy(line))
            in_optimized_sub = False
            continue

        # Parse turbine positions
        if in_optimized_turbines and line.startswith("Turbine"):
            turbine_lines.append(linetoxy(line))

    return turbine_lines, substation_lines, turbine_type


# ══════════════════════════════════════════════════════════════════════
#  LandBOSSE PROJECT LIST CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

project_list_path = Path("/home/lavender/Studies/Design of Wind Farms/modules/LandBOSSE/input/project_list.xlsx")
old_projects = pd.read_excel(project_list_path)
old_projects = pd.DataFrame(old_projects.iloc[[0]])


# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

layout_dir = os.path.join(BASE_DIR, "optimizedLayout")
layout_files = sorted(glob.glob(os.path.join(layout_dir, "*_layout.txt")))

# Skip the 'best_layout_*' duplicate if it exists
layout_files = [f for f in layout_files if not os.path.basename(f).startswith("best_layout")]

print(f"\n{'='*80}")
print(f"  BATCH LandBOSSE + FLORIS ANALYSIS")
print(f"  Found {len(layout_files)} layout files")
print(f"{'='*80}\n")

results_summary = []

for idx, lf in enumerate(layout_files):
    fname = os.path.basename(lf)
    print(f"\n{'#'*80}")
    print(f"  [{idx+1}/{len(layout_files)}]  FILE: {fname}")
    print(f"{'#'*80}")

    # ── 1) Parse layout ───────────────────────────────────────────────
    turbine_lines, substation_lines, turbine_type = parse_layout_v2(lf)
    n_turb = len(turbine_lines)

    if turbine_type is None:
        # Try to infer from filename
        # e.g. BAR_BAU_IEA_3.3MW_18_layout.txt → BAR_BAU_IEA_3.3MW
        parts = fname.replace("_layout.txt", "").rsplit("_", 1)  # split off the count
        turbine_type = parts[0]

    print(f"  Turbine type : {turbine_type}")
    print(f"  # turbines   : {n_turb}")
    print(f"  # substations: {len(substation_lines)}")

    if turbine_type not in TURBINE_PARAMS:
        print(f"  ⚠ WARNING: Unknown turbine type '{turbine_type}', skipping.")
        continue

    tp = TURBINE_PARAMS[turbine_type]
    P_rated_MW = tp["P_rated_MW"]
    D          = tp["D"]
    H          = tp["H"]
    V_rated    = tp["V_rated"]
    Ct_rated   = tp["Ct_rated"]

    A_rotor  = np.pi * D**2 / 4
    T_rated  = 0.5 * rho * A_rotor * Ct_rated * V_rated**2

    print(f"  Rated power  : {P_rated_MW:.3f} MW")
    print(f"  Rotor D      : {D:.0f} m,  Hub H: {H:.0f} m")
    print(f"  V_rated      : {V_rated} m/s,  Ct_rated: {Ct_rated:.6f}")

    # ── 2) Interconnect distance ──────────────────────────────────────
    d_interconnect_distance = 6.34  # default
    for pt in substation_lines:
        d_m = point_to_line_distance_meters(pl1, pl2, pt)
        d_interconnect_distance = d_m * METERS_TO_MILES

    # ── 3) Configure & run LandBOSSE ─────────────────────────────────
    projects = old_projects.copy()
    projects["Project ID"]       = [fname.replace("_layout.txt", "")]
    projects["Project data file"] = "iea36_120_project_data"
    projects["Total project construction time (months)"] = d_construction_time
    projects["Hub height m"]     = H
    projects["Rotor diameter m"] = D
    projects["Turbine spacing (times rotor diameter)"] = 5
    projects["Line Frequency (Hz)"] = 50
    projects["Number of turbines"] = n_turb
    projects["Fuel cost USD per gal"] = d_fuel_cost
    projects["Wind shear exponent"] = d_ws
    projects["Turbine rating MW"] = P_rated_MW
    projects["Row spacing (times rotor diameter)"] = 0
    projects["Rated Thrust (N)"] = T_rated
    projects["Distance to interconnect (miles)"] = d_interconnect_distance
    projects["Interconnect Voltage (kV)"] = 132

    projects.to_excel(project_list_path, index=False)

    print(f"\n  Running LandBOSSE...")
    try:
        landbosse_results = run_landbosse(
            np.array(turbine_lines),
            np.array(substation_lines),
            45,
            False,   # WriteExcel
            False    # Display
        )
    except Exception as e:
        print(f"  ⚠ LandBOSSE FAILED: {e}")
        continue

    bop_cost = landbosse_results['Cost per project'].sum()

    # BoP breakdown
    bop_by_module = (landbosse_results
                     .groupby('Module')['Cost per project']
                     .sum()
                     .sort_values(ascending=False))

    print(f"\n  BoP Cost: ${bop_cost:,.0f}")
    print(f"  BoP breakdown by module:")
    for mod, cost in bop_by_module.items():
        print(f"    {mod:25s}  ${cost:>14,.0f}")

    # ── 4) Run FLORIS ─────────────────────────────────────────────────
    x = [t[0] for t in turbine_lines]
    y = [t[1] for t in turbine_lines]

    print(f"\n  Running FLORIS...")
    fmodel = FlorisModel(r"inputs/gch.yaml")
    fmodel.set(layout_x=x, layout_y=y, wind_data=wind_rose, wind_shear=d_ws)
    fmodel.run()
    AEP_GWH_NO_YAW = fmodel.get_farm_AEP() / 1e9

    yaw_file = lf.replace("_layout.txt", "_yaw_angles.npy")
    if os.path.exists(yaw_file):
        print(f"  Loading existing yaw angles from {os.path.basename(yaw_file)}")
        yaw_optimal = np.load(yaw_file)
        fmodel.set(yaw_angles=yaw_optimal)
        fmodel.run()
    else:
        print(f"  Optimizing yaw angles...")
        try:
            from floris_cupy.optimization.yaw_optimization.yaw_optimizer_pr import YawOptimizationPR
            yaw_opt = YawOptimizationPR(fmodel, display=False)
            df_opt  = yaw_opt.optimize()
            yaw_optimal = np.stack(df_opt["yaw_angles_opt"].values)
            np.save(yaw_file, yaw_optimal)
            fmodel.set(yaw_angles=yaw_optimal)
            fmodel.run()
        except Exception as e:
            print(f"  ⚠ Yaw optimization failed: {e}. Running without yaw.")
            fmodel.run()

    fmodel_nw = FlorisModel(r"inputs/gch.yaml")
    fmodel_nw.set(layout_x=x, layout_y=y, wind_data=wind_rose, wind_shear=d_ws)
    fmodel_nw.run_no_wake()

    AEP_GWH    = fmodel.get_farm_AEP()    / 1e9
    AEP_GWH_NW = fmodel_nw.get_farm_AEP() / 1e9

    # Handle cupy arrays
    try:
        import cupy as cp
        AEP_GWH        = float(cp.asnumpy(AEP_GWH)) if hasattr(AEP_GWH, '__cuda_array_interface__') else float(AEP_GWH)
        AEP_GWH_NO_YAW = float(cp.asnumpy(AEP_GWH_NO_YAW)) if hasattr(AEP_GWH_NO_YAW, '__cuda_array_interface__') else float(AEP_GWH_NO_YAW)
        AEP_GWH_NW     = float(cp.asnumpy(AEP_GWH_NW)) if hasattr(AEP_GWH_NW, '__cuda_array_interface__') else float(AEP_GWH_NW)
    except ImportError:
        AEP_GWH        = float(AEP_GWH)
        AEP_GWH_NO_YAW = float(AEP_GWH_NO_YAW)
        AEP_GWH_NW     = float(AEP_GWH_NW)

    WAKE_LOSSES     = 100 * (AEP_GWH_NW - AEP_GWH) / AEP_GWH_NW
    FARM_EFFICIENCY = 100 * AEP_GWH / AEP_GWH_NW
    YAW_IMPROVEMENT = 100 * (AEP_GWH - AEP_GWH_NO_YAW) / AEP_GWH_NO_YAW if AEP_GWH_NO_YAW > 0 else 0.0

    INSTALL_CAPACITY = n_turb * P_rated_MW
    MAX_CAP_YEAR     = INSTALL_CAPACITY * 8760 / 1e3   # GWh
    CAPACITY_FACTOR  = 100 * AEP_GWH / MAX_CAP_YEAR

    print(f"\n  AEP (no yaw)  : {AEP_GWH_NO_YAW:.3f} GWh")
    print(f"  AEP (with yaw): {AEP_GWH:.3f} GWh")
    print(f"  Yaw Improvem. : +{YAW_IMPROVEMENT:.3f} %")
    print(f"  AEP (no wake) : {AEP_GWH_NW:.3f} GWh")
    print(f"  Wake Losses   : {WAKE_LOSSES:.3f} %")
    print(f"  Farm Efficiency: {FARM_EFFICIENCY:.3f} %")
    print(f"  Capacity Factor: {CAPACITY_FACTOR:.3f} %")
    print(f"  Installed Cap  : {INSTALL_CAPACITY:.2f} MW")

    # ── 5) Financials ─────────────────────────────────────────────────
    AEP_MWh = AEP_GWH * 1e3

    turbine_cost = TURB_COST_PER_MW * INSTALL_CAPACITY
    I = bop_cost + turbine_cost

    # Energy-weighted mean wind speed
    try:
        farm_power = fmodel.get_farm_power()
        try:
            farm_power = cp.asnumpy(farm_power).flatten()
        except:
            farm_power = np.array(farm_power).flatten()
    except:
        farm_power = np.zeros(len(WD_BINS) * len(WS_BINS))

    wd_mesh, ws_mesh = np.meshgrid(WD_BINS, WS_BINS, indexing='ij')
    freq_flat = freq_table.flatten()
    ws_flat   = ws_mesh.flatten()

    energy_per_cond = farm_power * freq_flat * 8760.0
    total_energy = np.sum(energy_per_cond)
    if total_energy > 0:
        ws_energy_weighted = np.sum(energy_per_cond * ws_flat) / total_energy
    else:
        ws_energy_weighted = 10.0

    effective_spot = A_denmark * (SPOT_C - SPOT_K * ws_energy_weighted)
    annual_revenue = AEP_MWh * effective_spot
    annual_om      = AEP_MWh * 12.0
    annual_rental  = INSTALL_CAPACITY * d_rent
    annual_net     = annual_revenue - annual_om - annual_rental

    # PV, NPV, PI
    pv_net = np.sum(annual_net / (1 + d_discount)**np.arange(1, d_life_time + 1))
    PI  = pv_net / I
    NPV = pv_net - I

    # IRR
    def npv_of_rate(r):
        return -I + np.sum(annual_net / (1 + r)**np.arange(1, d_life_time + 1))

    try:
        IRR = newton(npv_of_rate, x0=d_discount, tol=1e-12, maxiter=200) * 100
    except:
        IRR = float('nan')

    # LCOE
    annuity = ((1 + d_discount)**d_life_time - 1) / (d_discount * (1 + d_discount)**d_life_time)
    annual_fixed_cost = annual_om + annual_rental
    TLCC = I + annual_fixed_cost * annuity
    LCOE = TLCC / (AEP_MWh * annuity)

    print(f"\n  --- FINANCIALS ---")
    print(f"  Turbine Cost     : ${turbine_cost:>14,.0f}")
    print(f"  BoP Cost         : ${bop_cost:>14,.0f}")
    print(f"  Total Investment : ${I:>14,.0f}")
    print(f"  PV (net CF)      : ${pv_net:>14,.0f}")
    print(f"  NPV              : ${NPV:>14,.0f}")
    print(f"  PI               : {PI:.3f}")
    print(f"  IRR              : {IRR:.3f} %")
    print(f"  LCOE             : ${LCOE:.2f}/MWh")

    # ── Collect summary ───────────────────────────────────────────────
    results_summary.append({
        "File": fname,
        "Type": turbine_type,
        "#Turb": n_turb,
        "Cap(MW)": INSTALL_CAPACITY,
        "AEP(GWh)": round(AEP_GWH, 3),
        "YawGain%": round(YAW_IMPROVEMENT, 2),
        "WakeLoss%": round(WAKE_LOSSES, 2),
        "CF%": round(CAPACITY_FACTOR, 2),
        "BoP($M)": round(bop_cost / 1e6, 3),
        "I($M)": round(I / 1e6, 3),
        "NPV($M)": round(NPV / 1e6, 3),
        "PI": round(PI, 3),
        "IRR%": round(IRR, 2),
        "LCOE($/MWh)": round(LCOE, 2),
    })


# ══════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*120}")
print("  SUMMARY TABLE")
print(f"{'='*120}")

if results_summary:
    df_summary = pd.DataFrame(results_summary)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_summary.to_string(index=False))
    
    # Save to CSV
    csv_path = os.path.join(BASE_DIR, "landbosse_batch_results.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")
else:
    print("  No results to display.")

print(f"\n{'='*80}")
print("  DONE")
print(f"{'='*80}\n")
