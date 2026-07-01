#!/usr/bin/env python
"""Configuration module for wind farm layout optimization."""

import os
import numpy as np


# PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TURBINE_LIB = os.path.join(BASE_DIR, "turbineData")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
KML_PATH = os.path.join(INPUTS_DIR, "Denmark Site - Chin.kml")
FLORIS_CONFIG = os.path.join(INPUTS_DIR, "gch.yaml")

# Output directories
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
HIST_LOGS_DIR = os.path.join(BASE_DIR, "history_logs")
HIST_PLOTS_DIR = os.path.join(BASE_DIR, "history_plots")
OPTIMIZED_LAYOUT_DIR = os.path.join(BASE_DIR, "optimizedLayout")


# OPTIMIZATION PARAMETERS

MAX_ITER = 1000
N_PARTICLES = 75
N_MIN, N_MAX = 17, 17
SAVE_PLOTS = True
CAPACITY_MIN_MW, CAPACITY_MAX_MW = 45, 60
WAKE_STEERING = True
PEN_THRESHOLD = 1e-3


# PARALLELIZATION

MAX_WORKERS = 4
N_EVAL_THREADS = 4  # Tune this: 4-8 depending on your GPU VRAM

# PSO PARAMETERS

PSO_W = 0.5
PSO_C1 = 1.5
PSO_C2 = 1.5


# PENALTY WEIGHTS

LINE_PENALTY_WEIGHT          = 1e9
BOUNDARY_PENALTY_WEIGHT      = 1e8
EXCLUSION_PENALTY_WEIGHT     = 1e7
AEP_WEIGHT                   = 1e4
INTER_TURBINE_PENALTY_WEIGHT = 1e7
SUBSTATION_EXCL_PENALTY_WEIGHT = 1e13
SUBSTATION_BOUNDARY_PENALTY_WEIGHT = 1e12  # Hard wall: substation must stay inside site


# CABLE COST REWARD (positive - subtracted from score)

# The substation position is optimized to minimize total cable distance.
# Cable reward = CABLE_COST_WEIGHT * sum(1 / dist(substation, turbine))
# This is SUBTRACTED from the objective score, so shorter cables = better.
# Using inverse distance so closer turbines contribute more.
CABLE_COST_WEIGHT = 5e3   # Tune this to balance AEP vs cable savings


# TURBINE SELECTION

TURBINES = ["IEA_3_4MW", "BAR_BAU_IEA_3.3MW", "BAR_BAU_LSP_3.25MW"]


# DEFAULT TURBINE PARAMETERS

DEFAULT_HUB_HEIGHT = 120.0  # m


# TURBINE SPACING

MIN_SPACING_FACTOR = 2.0  # min_spacing = MIN_SPACING_FACTOR * rotor_diameter


# SITE PARAMETERS - DENMARK (Site 1)

D_WS = 0.17          # Wind shear exponent
D_TI = 12 / 100      # Turbulence intensity
D_FUEL_COST = 9.5    # USD/gal
D_LINE_FREQ = 50     # Hz
D_STANDARD_V = 220   # V
D_INTERCONNECT_V = 100  # kV
D_RENT = 15000       # $/MW-yr
D_O_AND_M = 0.012    # $/kWh
D_DISCOUNT = 3.6     # %
D_LIFE_TIME = 20     # years
D_CONSTRUCTION_TIME = 12  # months


# CONSTRAINT PARAMETERS

ROAD_BUFFER_M = 15.0          # Visual buffer for plotting
FIELD_BORDER_MAX_M = 50.0     # Max distance from field border
TOWN_RADIUS = 1000.0          # Town exclusion radius
BUILD_RADIUS_FACTOR = 4       # Building exclusion = HH * this
WT_RADIUS_FACTOR = 4          # Existing WT exclusion = HH * this

# Road exclusion distances (rotor_radius + buffer)
ROAD_EXCL_NORMAL_M = None   # Set dynamically: rotor_radius + ROAD_EXCL_NORMAL_BUFFER
ROAD_EXCL_TOFT_M = None     # Set dynamically: rotor_radius + ROAD_EXCL_TOFT_BUFFER
ROAD_EXCL_NORMAL_BUFFER = 10.0
ROAD_EXCL_TOFT_BUFFER = 15.0

# Substation parameters
SUBSTATION_EXCLUSION_FACTOR = 1.5  # 1.5 * rotor_diameter for safety


# COORDINATE CONVERSION

LAT_TO_M = 111000.0
LON_TO_M_AT_55 = 63000.0

# Reference point for DMS conversion
REF_LAT = 55 + (14 / 60) + (39.2 / 3600)
REF_LON = 9 + (00 / 60) + (41.3 / 3600)


# KML PARSING - Placemark name patterns

KML_SITE_AREA_NAME = "Site Area"
KML_ROAD_KREIS_PATTERN = "Kreisstra"
KML_ROAD_STAAT_PATTERN = "Staatsstra"
KML_SUBSTATION_PATTERN = "substation"
KML_BUILDING_PREFIXES = ["House", "Untitled placemark"]
KML_WTG_PREFIX = "WTG"
KML_FIELD_BORDERS_NAME = "Borders Green Fields"
KML_UNTITLED_PATH_NAME = "Untitled path"
KML_TOWN_COORDS = "55°15'46.7\"N 9°04'46.7\"E"


# WIND ROSE BINS

WD_BINS = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], dtype=float)
WB_SCALE = np.array([9.785, 8.284, 8.721, 9.633, 10.114, 8.340,
                      8.936, 10.759, 11.710, 11.363, 10.682, 8.965])
WB_SHAPE = np.array([2.306, 2.089, 1.888, 1.935, 1.945, 1.902,
                      1.909, 1.910, 1.968, 2.049, 2.064, 1.928])
FREQ_WD = np.array([14.71, 6.09, 6.16, 8.17, 9.58, 6.05,
                     5.34, 7.27, 8.00, 14.60, 7.78, 6.25]) / 100.0
WS_BINS = np.arange(3.0, 26.0, 1.0)
BINSIZE = 3  # m/s


# MONITOR PARAMETERS

MONITOR_POLL_INTERVAL = 20.0   # seconds between refreshes
MONITOR_ACTIVE_TIMEOUT = 60.0  # seconds; file older than this = stale


# NUMERICAL STABILITY

EPSILON = 1e-10


# PLOTTING LIMITS

PLOT_XLIM = (-500, 4500)
PLOT_YLIM = (-500, 2500)