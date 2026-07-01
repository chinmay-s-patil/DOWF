#!/usr/bin/env python

# Rotor Radi + Road Distance

# There are online websites to see what the LCoE was for EU countries.

# For calculating Spot Price, we need WS, WS is the Wind Speed.

#
# # Import Packages


import numpy as np
import pathlib
import yaml
import time
import shutil
import json
import os
import cupy as cp
import multiprocessing as mp
import matplotlib.pyplot as plt

from scipy.stats import weibull_min
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from floris_cupy import FlorisModel  # pyright: ignore[reportMissingImports]
from floris_cupy.wind_data import WindRose  # pyright: ignore[reportMissingImports]
import floris_cupy as _floris_pkg  # pyright: ignore[reportMissingImports]
from xml.etree import ElementTree as _ET

from concurrent.futures import ProcessPoolExecutor, as_completed

# Wake-steering optimizer
from floris_cupy.optimization.yaw_optimization.yaw_optimizer_pr import YawOptimizationPR


import warnings
warnings.filterwarnings("ignore")

os.chdir(r"/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/LoCE Opti")

# # Variables

# ## Optimization Variables

MAX_ITER = 250
N_PARTICLES = 50
MAX_WORKERS = 5
N_MIN, N_MAX = 17, 17
SAVE_PLOTS = True
CAPACITY_MIN, CAPACITY_MAX = 45, 60		# MW

WAKE_STEERING = True

LINE_PENALTY_WEIGHT          = 1e9
BOUNDARY_PENALTY_WEIGHT      = 1e8
EXCLUSION_PENALTY_WEIGHT     = 1e7
INTER_TURBINE_PENALTY_WEIGHT = 1e7
AEP_WEIGHT                   = 1e4

# ## Turbine Properties

tub_lib = r"./turbineData/"
turbines = ["IEA_3_4MW", "BAR_BAU_IEA_3.3MW", "BAR_BAU_LSP_3.25MW"]
# turbines = ["IEA_3_4MW"]

turbine_yaml_path = os.path.join(tub_lib, turbines[0] + ".yaml")
with open(turbine_yaml_path, 'r') as f:
	turb_data = yaml.safe_load(f)

HH = turb_data['hub_height']
ROTOR_DIAMETER_M = turb_data['rotor_diameter']

MIN_TURBINE_SPACING = 2 * ROTOR_DIAMETER_M  # Min spacing derived from yaml

# ## Contraints

ROAD_BUFFER_M = 15.0  # 15 m visual buffer for plotting
FIELD_BORDER_MAX_M = 50.0
ROTOR_RADIUS_M = ROTOR_DIAMETER_M / 2.0

TOWN_RADIUS = 1000.0
BUILD_RADIUS = HH * 4
WT_RADIUS = HH * 4

# Roads as exclusion zones: turbine must stay >= rotor_radius + buffer away
# Normal roads (Kreisstraße level): rotor_radius + 10m
ROAD_EXCL_NORMAL_M = ROTOR_RADIUS_M + 10.0
# Toftlundvej (Bundesstraße level): rotor_radius + 15m
ROAD_EXCL_TOFT_M   = ROTOR_RADIUS_M + 15.0

FIELD_BORDER_MAX_M = 50.0   # turbine must be within 50m of a field border

# ## Thresholds

PEN_THRESHOLD = 1e-3   # relaxed — tighten once optimizer is converging well

# ## Constraint Functions

# --- Capacity-aware turbine count limits ---
def _get_turbine_power_mw(turbine_name: str) -> float:
	"""Extract rated power in MW from the turbine YAML."""
	yaml_path = os.path.join(tub_lib, turbine_name + ".yaml")
	with open(yaml_path, 'r') as f:
		data = yaml.safe_load(f)

	rated_kw = None
	# Standard FLORIS key
	if 'rated_power' in data:
		rated_kw = float(data['rated_power'])
	# Fallback: max of the power curve table
	elif 'power_thrust_table' in data and 'power' in data['power_thrust_table']:
		rated_kw = max(float(p) for p in data['power_thrust_table']['power'])
	else:
		# Hard-coded fallback from assignment Table 3
		fallback = {
			"IEA_3_4MW": 3370.0,
			"BAR_BAU_IEA_3.3MW": 3300.0,
			"BAR_BAU_LSP_3.25MW": 3250.0,
		}
		rated_kw = fallback.get(turbine_name, 3370.0)

	return rated_kw / 1000.0  # kW → MW


def valid_turbine_counts(turbine_name: str, cap_min_mw: float, cap_max_mw: float):
	"""Return (n_min, n_max) for this turbine type to stay within capacity bounds."""
	p_mw = _get_turbine_power_mw(turbine_name)
	n_min = int(np.ceil(cap_min_mw / p_mw))
	n_max = int(np.floor(cap_max_mw / p_mw))
	# Safety clamp: at least 1 turbine, and n_max >= n_min
	n_min = max(1, n_min)
	n_max = max(n_min, n_max)
	return n_min, n_max, p_mw


# # Denmark Site

# ## Denmark Vars


d_ws = 0.17
d_ti = 12/100
d_fuel_cost = 9.5
d_line_freq = 50
d_standard_V = 220
d_interconnect_V = 100
d_rent = 15000
d_o_and_m = 0.012
d_discount = 3.6
d_life_time = 20
d_construction_time = 12		# Month


# ## Plotting Map

# ### Conversion to KM


lattitude_degree_to_m = 111000
longitude_degree_to_m_at_55 = 63000


# ### Helper Function


def get_ref(stri):
	# This is using the DMS format - Degree Minutes Seconds

	lattLB = 55 + (14/60) + (39.2/(3600))
	longLB = 9 + (00/60) + (41.3/(3600))

	latt_deg = float(stri.strip().split("°")[0])
	latt_min = float(stri.strip().split("'")[0].split("°")[1])
	latt_sec = float(stri.strip().split("\"")[0].split("°")[1].split("'")[1].replace("\\", ""))

	long_deg = float(stri.strip().split("N")[1].split("°")[0])
	long_min = float(stri.strip().split("N")[1].split("'")[0].split("°")[1])
	long_sec = float(stri.strip().split("N")[1].split("\"")[0].split("°")[1].split("'")[1].replace("\\", ""))

	latt = latt_deg + (latt_min/60) + (latt_sec/(3600))
	long = long_deg + (long_min/60) + (long_sec/(3600))

	latt = (latt - lattLB) * lattitude_degree_to_m

	long = (long - longLB) * longitude_degree_to_m_at_55

	return latt, long



def get_ref_WGS84(latt, long):
	# This is using the WGS84 lattitude/longitude form.

	lattLB = 55 + (14/60) + (39.2/(3600))
	longLB = 9 + (00/60) + (41.3/(3600))

	latt = (latt - lattLB) * lattitude_degree_to_m

	long = (long - longLB) * longitude_degree_to_m_at_55

	return latt, long


# ### Boundaries — parsed from KML
_KML_PATH = os.path.join(os.path.dirname(__file__), "inputs/Denmark Site - Chin.kml")
_kml_ns = {'kml': 'http://www.opengis.net/kml/2.2'}
_kml_tree = _ET.parse(_KML_PATH)
_kml_root = _kml_tree.getroot()


def _kml_placemarks():
	"""Yield (name, placemark_element) for every Placemark in the KML."""
	for pm in _kml_root.iter('{http://www.opengis.net/kml/2.2}Placemark'):
		n = pm.find('kml:name', _kml_ns)
		yield (n.text if n is not None else ''), pm


def _parse_coords(text):
	"""Parse a KML coordinates string → list of (lon, lat) floats."""
	out = []
	for tok in text.strip().split():
		parts = tok.split(',')
		out.append((float(parts[0]), float(parts[1])))  # lon, lat
	return out


#  Site boundary
for _name, _pm in _kml_placemarks():
	if _name == 'Site Area':
		_poly_coords = _parse_coords(
			_pm.find('.//kml:coordinates', _kml_ns).text)
		break

# The polygon is closed (last == first); drop the duplicate
if _poly_coords[-1] == _poly_coords[0]:
	_poly_coords = _poly_coords[:-1]

# Convert to local metres  (get_ref_WGS84 returns (lat_m, lon_m))
_boundary_m = [get_ref_WGS84(lat, lon) for lon, lat in _poly_coords]

# boundary_x = list of x (lon_m),  boundary_y = list of y (lat_m)
boundary_x = [lon_m for (_lat_m, lon_m) in _boundary_m]
boundary_y = [lat_m for (lat_m, _lon_m) in _boundary_m]

# The optimiser bounds use lattLB/RB/LT/RT and longLB/RB/LT/RT
# Assign them from the bounding box of the site polygon
lattLB = min(boundary_y); lattLT = max(boundary_y)
lattRB = lattLB;           lattRT = lattLT
longLB = min(boundary_x); longRB = max(boundary_x)
longLT = longLB;           longRT = longRB


#  Roads
# Kreisstraße  → normal roads
# Staatsstraße → Toftlundvej (higher-class road, wider buffer)
road_segments_normal = []
road_segments_toft   = []

for _name, _pm in _kml_placemarks():
	_ls = _pm.find('kml:LineString', _kml_ns)
	if _ls is None:
		continue
	if 'Kreisstra' in _name or 'kreisstra' in _name:
		_pts = _parse_coords(_ls.find('kml:coordinates', _kml_ns).text)
		_pts_m = [get_ref_WGS84(lat, lon) for lon, lat in _pts]
		for _i in range(len(_pts_m) - 1):
			road_segments_normal.append(
				((_pts_m[_i][1], _pts_m[_i][0]),
				 (_pts_m[_i+1][1], _pts_m[_i+1][0])))
	elif 'Staatsstra' in _name or 'staatsstra' in _name:
		_pts = _parse_coords(_ls.find('kml:coordinates', _kml_ns).text)
		_pts_m = [get_ref_WGS84(lat, lon) for lon, lat in _pts]
		for _i in range(len(_pts_m) - 1):
			road_segments_toft.append(
				((_pts_m[_i][1], _pts_m[_i][0]),
				 (_pts_m[_i+1][1], _pts_m[_i+1][0])))

road_segments = road_segments_normal + road_segments_toft


def road_buffer_polygon(x0, y0, x1, y1, width):
	"""Returns the 4 corners of a rectangle buffering a line segment."""
	dx, dy = x1 - x0, y1 - y0
	length = np.hypot(dx, dy)
	px, py = -dy / length, dx / length
	offset = width / 2
	return [
		(x0 + px * offset, y0 + py * offset),
		(x1 + px * offset, y1 + py * offset),
		(x1 - px * offset, y1 - py * offset),
		(x0 - px * offset, y0 - py * offset),
	]


#  Town (Kastrup — not in KML, kept as hardcoded)
lat_k_town, lon_k_town = get_ref(r"55°15'46.7\"N 9°04'46.7\"E")


#  Buildings / Houses
builds = []
for _name, _pm in _kml_placemarks():
	if _name.startswith('House') or _name == 'Untitled placemark':
		_pt = _pm.find('kml:Point', _kml_ns)
		if _pt is None:
			continue
		_c = _parse_coords(_pt.find('kml:coordinates', _kml_ns).text)[0]
		builds.append(get_ref_WGS84(_c[1], _c[0]))  # (lat, lon) → (lat_m, lon_m)


#  Existing Wind Turbines
wts = []
for _name, _pm in _kml_placemarks():
	if _name.startswith('WTG'):
		_pt = _pm.find('kml:Point', _kml_ns)
		if _pt is None:
			continue
		_c = _parse_coords(_pt.find('kml:coordinates', _kml_ns).text)[0]
		wts.append(get_ref_WGS84(_c[1], _c[0]))


#  Field Borders
# The "Borders Green Fields" polygon outer ring traces all field borders;
# each consecutive pair of vertices is one border segment.
# Additional "Untitled path" LineStrings add extra segments.

# 1) Extract polygon vertices as (lat_m, lon_m)
fb = []
for _name, _pm in _kml_placemarks():
	if _name == 'Borders Green Fields':
		_coords_text = _pm.find('.//kml:Polygon//kml:outerBoundaryIs'
								'//kml:coordinates', _kml_ns).text
		_raw = _parse_coords(_coords_text)
		# drop closing duplicate
		if _raw[-1] == _raw[0]:
			_raw = _raw[:-1]
		fb = [get_ref_WGS84(lat, lon) for lon, lat in _raw]
		break

# Build connectivity from consecutive polygon vertices
lines = [list(range(len(fb)))]  # single chain through all vertices

# Build LINE_SEGMENTS from the polygon edges
LINE_SEGMENTS = []
for _i in range(len(fb) - 1):
	LINE_SEGMENTS.append([(fb[_i][1], fb[_i][0]),
						  (fb[_i+1][1], fb[_i+1][0])])
# close the ring
if len(fb) > 2:
	LINE_SEGMENTS.append([(fb[-1][1], fb[-1][0]),
						  (fb[0][1], fb[0][0])])

# 2) Add "Untitled path" line segments
for _name, _pm in _kml_placemarks():
	if _name == 'Untitled path':
		_ls = _pm.find('kml:LineString', _kml_ns)
		if _ls is None:
			continue
		_pts = _parse_coords(_ls.find('kml:coordinates', _kml_ns).text)
		_pts_m = [get_ref_WGS84(lat, lon) for lon, lat in _pts]
		for _i in range(len(_pts_m) - 1):
			LINE_SEGMENTS.append([(_pts_m[_i][1], _pts_m[_i][0]),
								  (_pts_m[_i+1][1], _pts_m[_i+1][0])])

print(f"KML loaded: {len(boundary_x)} boundary pts, "
	  f"{len(road_segments)} road segs, {len(builds)} buildings, "
	  f"{len(wts)} WTGs, {len(fb)} field border pts, "
	  f"{len(LINE_SEGMENTS)} line segments")

# #### Actual Plotting


fig, ax = plt.subplots(figsize=(20,12))

ax.add_patch(plt.Polygon(list(zip(boundary_x, boundary_y)),closed=True, facecolor='green', edgecolor='blue', alpha=0.5, label="Valid Area"))

# Draw road buffers + road lines
first_road = True
for (x0, y0), (x1, y1) in road_segments:
	buf = road_buffer_polygon(x0, y0, x1, y1, ROAD_BUFFER_M)
	ax.add_patch(plt.Polygon(buf, closed=True, facecolor='blue', edgecolor='blue', alpha=0.4, label="Invalid Area - Road" if first_road else "_nolegend_"))
	ax.plot([x0, x1], [y0, y1], color='black', label="Road" if first_road else "_nolegend_")
	first_road = False

# Buildings
first_build = True
for lat, lon in builds:
	ax.plot(lon, lat, 'x', color='darkgreen')
	ax.add_patch(plt.Circle((lon, lat), radius=(HH * 4), facecolor='red', edgecolor='red', alpha=0.5, label="Invalid Area - Building" if first_build else "_nolegend_"))
	first_build = False

# Towns
ax.plot(lon_k_town, lat_k_town, 'x', color='purple', label="Town")
ax.add_patch(plt.Circle((lon_k_town, lat_k_town), radius=(1000), facecolor='purple', edgecolor='purple', alpha=0.5, label="Invalid Area - Town"))

# Wind Turbines
first_wt = True
for lat, lon in wts:
	ax.plot(lon, lat, '>', color='navy', label="Wind Turbines" if first_wt else "_nolegend_")
	ax.add_patch(plt.Circle((lon, lat), radius=(HH * 4), facecolor='firebrick', edgecolor='firebrick', alpha=0.5, label="Invalid Area - Wind Turbine" if first_wt else "_nolegend_"))
	first_wt = False

first_fb = True
for (x1, y1), (x2, y2) in LINE_SEGMENTS:
	ax.plot([x1, x2], [y1, y2], color='yellow', label="Field Border" if first_fb else "_nolegend_")
	first_fb = False

ax.set_aspect('equal')
ax.autoscale()
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_xlim(-500, 4500)
ax.set_ylim(-500, 2500)
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
# plt.show()

# ## Setting Up Floris

# ### Wind Rose (Kastrup)

binsize = 3  # m/s

WD_BINS  = np.array([0,30,60,90,120,150,180,210,240,270,300,330], dtype=float)
WB_SCALE = np.array([9.785,8.284,8.721,9.633,10.114,8.340,
					  8.936,10.759,11.710,11.363,10.682,8.965])  # lambda [m/s]
WB_SHAPE = np.array([2.306,2.089,1.888,1.935,1.945,1.902,
					  1.909,1.910,1.968,2.049,2.064,1.928])       # k [-]
FREQ_WD  = np.array([14.71,6.09,6.16,8.17,9.58,6.05,
					  5.34,7.27,8.00,14.60,7.78,6.25]) / 100.0

# Wind-speed bins: cut-in to cut-out at 1 m/s resolution
WS_BINS   = np.arange(3.0, 26.0, 1.0)   # 23 bins
SITE1_TI  = d_ti      # turbulence intensity (Table 1)
WIND_SHEAR= d_ws      # shear exponent alpha (Table 1)

# Build joint freq table P(WD_i, WS_j) from per-sector Weibull distributions
freq_table = np.zeros((len(WD_BINS), len(WS_BINS)))
for i, (k, lam) in enumerate(zip(WB_SHAPE, WB_SCALE)):
	p_ws = weibull_min.pdf(WS_BINS, c=k, scale=lam) * binsize
	p_ws /= p_ws.sum()   # renormalise (remove truncated tail)
	freq_table[i, :] = FREQ_WD[i] * p_ws

wind_rose = WindRose(
	wind_directions=WD_BINS,
	wind_speeds=WS_BINS,
	ti_table=SITE1_TI,
	freq_table=freq_table,
)

# ### Show WindRose

wind_rose.plot()

# ## Actual Optimization
#
# ### Line Segment Forcing

# ### Variables

# #### Line Segments (already built from KML above)




# ### Constraints

# #### Line Constraint Penalty


def _point_to_segment_dist_sq(px, py, x1, y1, x2, y2):
	"""Squared distance from point (px, py) to segment (x1,y1)-(x2,y2)."""
	dx, dy = x2 - x1, y2 - y1
	seg_len_sq = dx * dx + dy * dy
	if seg_len_sq == 0:
		return (px - x1) ** 2 + (py - y1) ** 2
	t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
	cx, cy = x1 + t * dx, y1 + t * dy
	return (px - cx) ** 2 + (py - cy) ** 2


def min_dist_to_any_line(x, y):
	"""Field border penalty: penalise if further than 50m from any field border segment"""
	best_field = np.inf
	for (x1, y1), (x2, y2) in LINE_SEGMENTS:
		d = _point_to_segment_dist_sq(x, y, x1, y1, x2, y2)
		if d < best_field: best_field = d
	dist_field = np.sqrt(best_field) if best_field != np.inf else np.inf
	return max(0.0, dist_field - FIELD_BORDER_MAX_M) ** 2

def line_penalty(positions):
	"""
	Total line-constraint penalty for all turbines.
	positions: list of (x, y) tuples
	"""
	return sum(min_dist_to_any_line(x, y) for x, y in positions)


# #### Boundary Constraint


# Build the actual boundary polygon
boundary_polygon = Polygon(zip(boundary_x, boundary_y))

def boundary_penalty(positions):
	total = 0.0
	for x, y in positions:
		p = Point(x, y)
		if not boundary_polygon.contains(p):
			total += boundary_polygon.exterior.distance(p) ** 2
	return total


# #### Exclusion Penalty




EXCLUSION_ZONES = []
for lat, lon in builds:
	EXCLUSION_ZONES.append((lon, lat, BUILD_RADIUS))

EXCLUSION_ZONES.append((lon_k_town, lat_k_town, TOWN_RADIUS))

for lat, lon in wts:
	EXCLUSION_ZONES.append((lon, lat, WT_RADIUS))


# Roads are line segments, not points — handled separately in GPU penalty below
# Store as flat arrays for GPU use
ROAD_NORMAL_SEGS = road_segments_normal  # already defined
ROAD_TOFT_SEGS   = road_segments_toft

def exclusion_penalty(positions):
	total = 0.0
	for tx, ty in positions:
		# Point exclusions
		for cx, cy, r in EXCLUSION_ZONES:
			dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)
			if dist < r:
				total += (r - dist) ** 2

		# Road normal exclusion
		for (x1, y1), (x2, y2) in ROAD_NORMAL_SEGS:
			d = np.sqrt(_point_to_segment_dist_sq(tx, ty, x1, y1, x2, y2))
			if d < ROAD_EXCL_NORMAL_M:
				total += (ROAD_EXCL_NORMAL_M - d) ** 2

		# Road Toft exclusion
		for (x1, y1), (x2, y2) in ROAD_TOFT_SEGS:
			d = np.sqrt(_point_to_segment_dist_sq(tx, ty, x1, y1, x2, y2))
			if d < ROAD_EXCL_TOFT_M:
				total += (ROAD_EXCL_TOFT_M - d) ** 2
	return total


# #### Inter-Turbine Distance Penalty

def interturbine_penalty(positions):
	total = 0.0
	n = len(positions)
	for i in range(n):
		for j in range(i + 1, n):
			dx = positions[i][0] - positions[j][0]
			dy = positions[i][1] - positions[j][1]
			dist = np.sqrt(dx**2 + dy**2)
			if dist < MIN_TURBINE_SPACING:
				total += (MIN_TURBINE_SPACING - dist) ** 2
	return total


# #### Objective Function


fmodel = FlorisModel(r"inputs/gch.yaml")

def evaluate_layout(positions: list[tuple[float, float]]) -> float:
	x = []
	y = []
	for pos in positions:
		x.append(pos[0])
		y.append(pos[1])
	fmodel.set(layout_x=x,layout_y=y,wind_data=wind_rose,wind_shear=d_ws)
	fmodel.run()
	return(fmodel.get_farm_AEP()/1e9)

def _objective(flat_xy):
	positions = [(flat_xy[2 * i], flat_xy[2 * i + 1]) for i in range(N_TURBINES)]
	value = evaluate_layout(positions) * AEP_WEIGHT
	penalty  = LINE_PENALTY_WEIGHT * line_penalty(positions)
	penalty += BOUNDARY_PENALTY_WEIGHT * boundary_penalty(positions)
	penalty += EXCLUSION_PENALTY_WEIGHT * exclusion_penalty(positions)
	penalty += INTER_TURBINE_PENALTY_WEIGHT * interturbine_penalty(positions)
	return -value + penalty


# #### Boundaries


X_MIN = min(longLB, longLT)
X_MAX = max(longRB, longRT)
Y_MIN = min(lattLB, lattRB)
Y_MAX = max(lattLT, lattRT)


# #### OBJECTIVE

def _make_objective(n_turbines, fmodel_local):
	def _get_seg_data(segs):
		if not segs:
			return None, None, None, None
		p1 = cp.array([(x1, y1) for ((x1, y1), (x2, y2)) in segs])
		p2 = cp.array([(x2, y2) for ((x1, y1), (x2, y2)) in segs])
		dxdy = p2 - p1
		lensq = cp.sum(dxdy**2, axis=1)
		lensq = cp.where(lensq == 0, 1e-10, lensq)
		return p1, p2, dxdy, lensq

	l_p1,  l_p2,  l_dxdy,  l_lensq  = _get_seg_data(LINE_SEGMENTS)
	rn_p1, rn_p2, rn_dxdy, rn_lensq = _get_seg_data(ROAD_NORMAL_SEGS)
	rt_p1, rt_p2, rt_dxdy, rt_lensq = _get_seg_data(ROAD_TOFT_SEGS)

	exc_arr = cp.array(EXCLUSION_ZONES)   # (E, 3)
	exc_xy  = exc_arr[:, :2]
	exc_r   = exc_arr[:, 2]


	def _dist_to_segs_batch(px, py, p1, dxdy, lensq):
		"""
		px, py : (P, T, 1)
		p1     : (S, 2)
		returns: (P, T)  — min distance to any segment
		"""
		if p1 is None:
			return cp.full((px.shape[0], px.shape[1]), cp.inf)
		# (P, T, S)
		px_diff = px - p1[:, 0]          # broadcasts (P,T,1) - (S,) -> (P,T,S)
		py_diff = py - p1[:, 1]
		t = (px_diff * dxdy[:, 0] + py_diff * dxdy[:, 1]) / lensq   # (P,T,S)
		t = cp.clip(t, 0.0, 1.0)
		cx = p1[:, 0] + t * dxdy[:, 0]   # (P,T,S)
		cy = p1[:, 1] + t * dxdy[:, 1]
		dists = cp.sqrt((px - cx)**2 + (py - cy)**2)                 # (P,T,S)
		return cp.min(dists, axis=2)                                   # (P,T)

	def _batched_objective(batch_pos):
		# batch_pos: (P, T, 2)
		n_particles = batch_pos.shape[0]

		# --- 1. FLORIS AEP (CPU, sequential) ---
		# NOTE: aeps is kept in GWh to match the non-batched objective scale.
		# Do NOT divide by 1e9 again downstream.
		aeps = cp.zeros(n_particles)
		for i in range(n_particles):
			layout = batch_pos[i]
			x_cpu = [float(layout[j, 0]) for j in range(n_turbines)]
			y_cpu = [float(layout[j, 1]) for j in range(n_turbines)]
			fmodel_local.set(
				layout_x=x_cpu, layout_y=y_cpu,
				wind_data=wind_rose,
				wind_shear=d_ws
			)
			fmodel_local.run()
			aeps[i] = fmodel_local.get_farm_AEP() / 1e9

		values = aeps * AEP_WEIGHT

		# --- 2. GPU Penalties ---
		px = batch_pos[:, :, 0:1]   # (P, T, 1)
		py = batch_pos[:, :, 1:2]

		# Field border penalty: penalise if further than 50m from any field border segment
		dist_field = _dist_to_segs_batch(px, py, l_p1, l_dxdy, l_lensq)   # (P, T)
		pen_field  = cp.maximum(0.0, dist_field - FIELD_BORDER_MAX_M)
		line_pen   = cp.sum(pen_field ** 2, axis=1)                         # (P,)

		# Road exclusion penalty: penalise if CLOSER than the minimum allowed distance
		dist_road_n = _dist_to_segs_batch(px, py, rn_p1, rn_dxdy, rn_lensq)  # (P, T)
		pen_road_n  = cp.maximum(0.0, ROAD_EXCL_NORMAL_M - dist_road_n)
		road_pen_n  = cp.sum(pen_road_n ** 2, axis=1)

		dist_road_t = _dist_to_segs_batch(px, py, rt_p1, rt_dxdy, rt_lensq)
		pen_road_t  = cp.maximum(0.0, ROAD_EXCL_TOFT_M - dist_road_t)
		road_pen_t  = cp.sum(pen_road_t ** 2, axis=1)

		road_pen = road_pen_n + road_pen_t

		# Point exclusion zones (buildings, town, existing turbines)
		diff_exc = batch_pos[:, :, None, :] - exc_xy[None, None, :, :]  # (P,T,E,2)
		dist_exc = cp.sqrt(cp.sum(diff_exc**2, axis=3))                  # (P,T,E)
		pen_exc  = cp.maximum(0.0, exc_r[None, None, :] - dist_exc)
		exc_pen  = cp.sum(pen_exc ** 2, axis=(1, 2))                     # (P,)

		# Inter-turbine spacing
		diff_inter = batch_pos[:, :, None, :] - batch_pos[:, None, :, :]  # (P,T,T,2)
		dist_inter = cp.sqrt(cp.sum(diff_inter**2, axis=3))               # (P,T,T)
		mask = cp.triu(cp.ones((n_turbines, n_turbines), dtype=bool), k=1)
		dist_inter = cp.where(mask, dist_inter, cp.inf)
		pen_inter  = cp.maximum(0.0, MIN_TURBINE_SPACING - dist_inter)
		inter_pen  = cp.sum(pen_inter ** 2, axis=(1, 2))                  # (P,)

		# Boundary penalty (Shapely, CPU)
		bound_pen_cpu = np.zeros(n_particles)
		batch_pos_cpu = batch_pos.get() if hasattr(batch_pos, 'get') else batch_pos
		for i in range(n_particles):
			pos_list = [(batch_pos_cpu[i, j, 0], batch_pos_cpu[i, j, 1])
						for j in range(n_turbines)]
			bound_pen_cpu[i] = boundary_penalty(pos_list)
		bound_pen = cp.array(bound_pen_cpu)

		penalty = (LINE_PENALTY_WEIGHT      * line_pen   +
				   EXCLUSION_PENALTY_WEIGHT * road_pen   +   # roads use same weight as buildings
				   BOUNDARY_PENALTY_WEIGHT  * bound_pen  +
				   EXCLUSION_PENALTY_WEIGHT * exc_pen    +
				   INTER_TURBINE_PENALTY_WEIGHT * inter_pen)

		score = -values + penalty
		return score, aeps, penalty

	return _batched_objective

def save_layout_plot(positions, n_turbines, filename, title=""):
	fig, ax = plt.subplots(figsize=(20,12))
	ax.add_patch(plt.Polygon(list(zip(boundary_x, boundary_y)),closed=True, facecolor='green', edgecolor='blue', alpha=0.5, label="Valid Area"))

	first_road = True
	for (x0, y0), (x1, y1) in road_segments:
		buf = road_buffer_polygon(x0, y0, x1, y1, ROAD_BUFFER_M)
		ax.add_patch(plt.Polygon(buf, closed=True, facecolor='blue', edgecolor='blue', alpha=0.4, label="Invalid Area - Road" if first_road else "_nolegend_"))
		ax.plot([x0, x1], [y0, y1], color='black', label="Road" if first_road else "_nolegend_")
		first_road = False

	first_build = True
	for lat, lon in builds:
		ax.plot(lon, lat, 'x', color='darkgreen')
		ax.add_patch(plt.Circle((lon, lat), radius=(HH * 4), facecolor='red', edgecolor='red', alpha=0.5, label="Invalid Area - Building" if first_build else "_nolegend_"))
		first_build = False

	ax.plot(lon_k_town, lat_k_town, 'x', color='purple', label="Town")
	ax.add_patch(plt.Circle((lon_k_town, lat_k_town), radius=(1000), facecolor='purple', edgecolor='purple', alpha=0.5, label="Invalid Area - Town"))

	first_wt = True
	for lat, lon in wts:
		ax.plot(lon, lat, '>', color='navy', label="Wind Turbines" if first_wt else "_nolegend_")
		ax.add_patch(plt.Circle((lon, lat), radius=(HH * 4), facecolor='firebrick', edgecolor='firebrick', alpha=0.5, label="Invalid Area - Wind Turbine" if first_wt else "_nolegend_"))
		first_wt = False

	first_line = True
	for (x1, y1), (x2, y2) in LINE_SEGMENTS:
		ax.plot([x1, x2], [y1, y2], color='yellow', label="Field Border" if first_line else "_nolegend_")
		first_line = False

	first_best = True
	for x, y in positions:
		ax.plot(x, y, '*', color='lime', markersize=14,
				label="Optimized turbine" if first_best else "_nolegend_")
		ax.add_patch(plt.Circle((x, y), radius=MIN_TURBINE_SPACING, facecolor='none', edgecolor='magenta', linestyle='-', linewidth=1.5, label="Min Spacing" if first_best else "_nolegend_"))
		first_best = False

	if title:
		ax.set_title(title, fontsize=16)

	ax.set_aspect('equal')
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.set_xlim(-500, 4500)
	ax.set_ylim(-500, 2500)
	ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
	plt.tight_layout()
	plt.savefig(filename, dpi=300, bbox_inches='tight')
	plt.close(fig)

def run_optimization(n_turbines, seed=42, maxiter=50, disp=True, turbine_type_name=""):
	# pyrefly: ignore [missing-import]
	from pso_optimizer import BatchedGPUParticleSwarm

	bounds = [(X_MIN, X_MAX) if i % 2 == 0 else (Y_MIN, Y_MAX)
			  for i in range(2 * n_turbines)]

	fmodel_local = FlorisModel(r"inputs/gch.yaml")

	abs_turb_path = os.path.abspath(turbine_yaml_path)
	# We must initialize layout_x and layout_y first before setting turbine_type for all turbines
	# Since we evaluate batches later, we can just set a dummy layout here to initialize the types
	fmodel_local.set(
		layout_x=[0.0]*n_turbines,
		layout_y=[0.0]*n_turbines,
		wind_data=wind_rose, 
		wind_shear=d_ws,
		turbine_type=[abs_turb_path] * n_turbines
	)
	# fmodel_local.assign_hub_height_to_ref_height()

	pso = BatchedGPUParticleSwarm(
		n_particles=N_PARTICLES,
		n_turbines=n_turbines,
		bounds=bounds,
		maxiter=maxiter,
		seed=seed,
		disp=disp,
		turbine_type_name=turbine_type_name
	)

	objective_fn = _make_objective(n_turbines, fmodel_local)

	if SAVE_PLOTS:
		# Ensure directory exists for plots
		out_dir = os.path.join("plots", turbine_type_name, str(n_turbines))
		os.makedirs(out_dir, exist_ok=True)

		hist_logs_dir = "history_logs"
		hist_plots_dir = "history_plots"
		os.makedirs(hist_logs_dir, exist_ok=True)
		os.makedirs(hist_plots_dir, exist_ok=True)

		pen_file = os.path.join(hist_logs_dir, f"{turbine_type_name}_{n_turbines}_penalty_hist.txt")
		aep_file = os.path.join(hist_logs_dir, f"{turbine_type_name}_{n_turbines}_AEP_hist.txt")
		hist_plot_file = os.path.join(hist_plots_dir, f"history_{turbine_type_name}_{n_turbines}.png")

		# Clear existing logs for fresh run
		with open(pen_file, "w") as f: pass
		with open(aep_file, "w") as f: pass

		local_iters = []
		local_pens = []
		local_aeps = []

		def on_new_best(flat_pos, score, aep, penalty, it,
						cur_pos, cur_score, cur_aep, cur_penalty):
			# Plot the CURRENT iteration's best particle layout
			pos_list = [(cur_pos[2 * i], cur_pos[2 * i + 1]) for i in range(n_turbines)]
			title = f"Iter {it} | Cur Score: {cur_score:.1f} | Cur AEP: {cur_aep:.3f} GWh | Best AEP: {aep:.3f} GWh"
			filename = os.path.join(out_dir, f"iter_{it:04d}.png")
			save_layout_plot(pos_list, n_turbines, filename, title)

			# Log the current iteration's penalty and AEP
			with open(pen_file, "a") as f:
				f.write(f"{cur_penalty}\n")
			with open(aep_file, "a") as f:
				f.write(f"{cur_aep}\n")

			local_iters.append(it)
			local_pens.append(cur_penalty)
			# cur_aep from _batched_objective is already in GWh — do NOT divide by 1e9 again
			local_aeps.append(cur_aep)

			fig, ax1 = plt.subplots(figsize=(10, 6))
			ax1.set_xlabel('Iteration')
			ax1.set_ylabel('Penalty', color='tab:red')
			ax1.set_yscale('log')
			ax1.plot(local_iters, local_pens, color='tab:red', marker='o', label='Penalty')
			ax1.axhline(1e-4, color='tab:red', linestyle='--', alpha=0.5, label='Threshold (1e-4)')
			ax1.tick_params(axis='y', labelcolor='tab:red')

			ax2 = ax1.twinx()
			ax2.set_ylabel('AEP (GWh)', color='tab:blue')
			ax2.plot(local_iters, local_aeps, color='tab:blue', marker='x', label='AEP')
			ax2.tick_params(axis='y', labelcolor='tab:blue')

			plt.title(f"Live Optimization History: {n_turbines} Turbines ({turbine_type_name})")
			fig.tight_layout()
			plt.savefig(hist_plot_file)
			plt.close(fig)
	else:
		on_new_best = None

	print(f"Starting PSO for n={n_turbines}...")
	start_pso_t = time.time()
	flat_best_pos, start_pos_cpu, best_score, history = pso.optimize(objective_fn, callback=on_new_best)
	pso_time = time.time() - start_pso_t

	best_positions = [(flat_best_pos[2 * i], flat_best_pos[2 * i + 1]) for i in range(n_turbines)]
	start_positions = [(start_pos_cpu[2 * i], start_pos_cpu[2 * i + 1]) for i in range(n_turbines)]

	# --- Evaluate best layout: WITHOUT yaw steering -----------------
	x_cpu = [p[0] for p in best_positions]
	y_cpu = [p[1] for p in best_positions]
	fmodel_local.set(
		layout_x=x_cpu, layout_y=y_cpu,
		wind_data=wind_rose,
		wind_shear=d_ws
	)
	fmodel_local.run()
	exact_aep_no_yaw = fmodel_local.get_farm_AEP()

	# --- Evaluate best layout: WITH yaw steering --------------------
	exact_aep_yaw = None
	yaw_optimal = None

	# Compute penalty for the best layout to decide if yaw opt is worth it
	best_pen = (LINE_PENALTY_WEIGHT * line_penalty(best_positions) +
				BOUNDARY_PENALTY_WEIGHT * boundary_penalty(best_positions) +
				EXCLUSION_PENALTY_WEIGHT * exclusion_penalty(best_positions) +
				INTER_TURBINE_PENALTY_WEIGHT * interturbine_penalty(best_positions))

	if WAKE_STEERING and best_pen < PEN_THRESHOLD:
		try:
			yaw_opt = YawOptimizationPR(fmodel_local, display=False)
			df_opt = yaw_opt.optimize()
			yaw_angles_list = df_opt["yaw_angles_opt"].values
			yaw_angles_list = [y.get() if hasattr(y, 'get') else y for y in yaw_angles_list]
			yaw_optimal = np.stack(yaw_angles_list)
			fmodel_local.set(yaw_angles=yaw_optimal)
			fmodel_local.run()
			exact_aep_yaw = fmodel_local.get_farm_AEP()
		except Exception as e:
			print(f"  [Wake Steering] Yaw optimization failed: {e}")
	elif WAKE_STEERING:
		print(f"  [Wake Steering] Skipped — penalty too high ({best_pen:.6f} >= {PEN_THRESHOLD})")

	return (flat_best_pos, start_positions, best_positions,
			exact_aep_no_yaw, exact_aep_yaw, history, pso_time, yaw_optimal)


# ### Actual Running

# #### Parallel Sweep




def _run_one(args):
	n, t_idx = args

	global HH, ROTOR_DIAMETER_M, MIN_TURBINE_SPACING, EXCLUSION_ZONES
	global ROAD_EXCL_NORMAL_M, ROAD_EXCL_TOFT_M, turbine_yaml_path

	turbine_yaml_path = os.path.join(tub_lib, turbines[t_idx] + ".yaml")
	with open(turbine_yaml_path, 'r') as f:
		turb_data = yaml.safe_load(f)

	HH = turb_data['hub_height']
	ROTOR_DIAMETER_M = turb_data['rotor_diameter']
	MIN_TURBINE_SPACING = 2 * ROTOR_DIAMETER_M

	EXCLUSION_ZONES = []
	for lat, lon in builds:
		EXCLUSION_ZONES.append((lon, lat, BUILD_RADIUS))
	EXCLUSION_ZONES.append((lon_k_town, lat_k_town, TOWN_RADIUS))
	for lat, lon in wts:
		EXCLUSION_ZONES.append((lon, lat, WT_RADIUS))

	flat_best_pos, x0, pos, exact_aep_no_yaw, exact_aep_yaw, history, pso_time, yaw_optimal = run_optimization(
		n_turbines=n, maxiter=MAX_ITER, turbine_type_name=turbines[t_idx]
	)
	line_pen  = line_penalty(pos)
	bound_pen = boundary_penalty(pos)
	excl_pen  = exclusion_penalty(pos)
	inter_pen = interturbine_penalty(pos)
	total_pen = line_pen + bound_pen + excl_pen + inter_pen

	return {
		"n": n, "t_idx": t_idx, "x0": x0, "positions": pos,
		"aep_no_yaw": float(exact_aep_no_yaw) if exact_aep_no_yaw is not None else None,
		"aep_yaw": float(exact_aep_yaw) if exact_aep_yaw is not None else None,
		"total_pen": total_pen,
		"time": pso_time, "history": history,
		"yaw_optimal": yaw_optimal
	}

if __name__ == '__main__':
	try:
		mp.set_start_method('spawn')
	except RuntimeError:
		pass  # already set

	if SAVE_PLOTS:
		if os.path.exists("plots"):
			shutil.rmtree("plots")
		os.makedirs("plots", exist_ok=True)

	print("Running MULTIPROCESS parallel sweep with Spawn...")
	all_results = {}

	# Initialize output directories and trackers
	os.makedirs("history_plots", exist_ok=True)
	os.makedirs("history_logs", exist_ok=True)
	os.makedirs("optimizedLayout", exist_ok=True)
	times_dict = {}

	start_time = time.time()

	# Use ProcessPoolExecutor for true parallel python processes
	with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
		futures = {}
		for t_idx, t_name in enumerate(turbines):
			n_min_t, n_max_t, p_mw = valid_turbine_counts(
				t_name, CAPACITY_MIN, CAPACITY_MAX
			)
			print(f"  {t_name}: rated {p_mw:.3f} MW → valid n = [{n_min_t}, {n_max_t}]")
			for n in range(n_min_t, n_max_t + 1):
				futures[ex.submit(_run_one, (n, t_idx))] = (n, t_idx)

		for fut in as_completed(futures):
			r = fut.result()
			key = f"{r['n']}_{r['t_idx']}"
			all_results[key] = r

			t = r['time']
			hrs = int(t // 3600)
			mins = int((t % 3600) // 60)
			secs = t % 60
			time_str = f"{hrs} hr {mins} min {secs:.2f} secs" if hrs > 0 else f"{mins} min {secs:.2f} secs"

			aep_base = r['aep_no_yaw'] / 1e9
			aep_yaw_str = ""
			if r['aep_yaw'] is not None:
				aep_yaw_val = r['aep_yaw'] / 1e9
				gain = (r['aep_yaw'] / r['aep_no_yaw'] - 1) * 100
				aep_yaw_str = f" | AEP+yaw={aep_yaw_val:.3f} GWh (+{gain:.1f}%)"

			print(f"  n={r['n']} | type={turbines[r['t_idx']]} | AEP={aep_base:.3f} GWh{aep_yaw_str} | penalty={r['total_pen']:.6f}")
			print(f"  Time taken for optimization of Turbine {r['n']} (type {r['t_idx']}) is {time_str}\n")

			t_name = turbines[r["t_idx"]]
			n_curr = r["n"]

			# Save optimized layout text file dynamically
			with open(f"optimizedLayout/{t_name}_{n_curr}_layout.txt", "w") as f:
				f.write(f"=== Best layout: {n_curr} turbines, type {t_name} ===\n")
				f.write(f"Optimization Time: {time_str}\n")
				f.write(f"AEP (no yaw): {r['aep_no_yaw']/1e9:.3f} GWh/yr\n")
				if r['aep_yaw'] is not None:
					f.write(f"AEP (yaw)   : {r['aep_yaw']/1e9:.3f} GWh/yr\n")
					f.write(f"Yaw gain    : +{((r['aep_yaw']/r['aep_no_yaw']-1)*100):.1f}%\n")
				f.write(f"Penalty     : {r['total_pen']:.6f}\n")
				for i, (x, y) in enumerate(r["positions"]):
					f.write(f"  Turbine {i}: x={x:.4f} km, y={y:.4f} km\n")
				f.write("\n=== Start position (random seed) ===\n")
				for i, (x, y) in enumerate(r["x0"]):
					f.write(f"  Turbine {i}: x={x:.4f} km, y={y:.4f} km\n")
				# Save yaw angles if wake steering was used
				if r.get("yaw_optimal") is not None:
					yaw_file = f"optimizedLayout/{t_name}_{n_curr}_yaw_angles.npy"
					np.save(yaw_file, r["yaw_optimal"])

			# Save history plot dynamically
			hist = r["history"]
			iters = [h["iter"] for h in hist]
			pens = [h["penalty"] for h in hist]
			# h["aep"] from _batched_objective is already in GWh — do NOT divide by 1e9 again
			aeps = [h["aep"] for h in hist]

			fig, ax1 = plt.subplots(figsize=(10, 6))
			ax1.set_xlabel('Iteration')
			ax1.set_ylabel('Penalty', color='tab:red')
			ax1.set_yscale('log')
			ax1.plot(iters, pens, color='tab:red', label='Penalty')
			ax1.axhline(1e-4, color='tab:red', linestyle='--', alpha=0.5, label='Threshold (1e-4)')
			ax1.tick_params(axis='y', labelcolor='tab:red')

			ax2 = ax1.twinx()
			ax2.set_ylabel('AEP (GWh)', color='tab:blue')
			ax2.plot(iters, aeps, color='tab:blue', label='AEP')
			ax2.tick_params(axis='y', labelcolor='tab:blue')

			plt.title(f"Optimization History: {n_curr} Turbines ({t_name})")
			fig.tight_layout()
			plt.savefig(f"history_plots/history_{t_name}_{n_curr}.png")
			plt.close(fig)

			# Save raw history data to text files
			with open(f"history_logs/{t_name}_{n_curr}_penalty_hist.txt", "w") as f:
				f.write("\n".join(str(p) for p in pens))

			with open(f"history_logs/{t_name}_{n_curr}_AEP_hist.txt", "w") as f:
				f.write("\n".join(str(a) for a in aeps))

			with open(f"history_logs/{t_name}_{n_curr}_penalty_hist.txt", "a") as f:
				f.write("\nEND\n")
			with open(f"history_logs/{t_name}_{n_curr}_AEP_hist.txt", "a") as f:
				f.write("\nEND\n")

			# Save times dynamically
			if t_name not in times_dict:
				times_dict[t_name] = {}
			times_dict[t_name][n_curr] = r["time"]
			with open("times_taken.json", "w") as f:
				json.dump(times_dict, f, indent=4)

	total_time = time.time() - start_time
	print(f"--- Optimization complete in {total_time:.2f} seconds ({(total_time/60):.2f} minutes) ---")



	satisfying = {k: r for k, r in all_results.items() if r["total_pen"] < PEN_THRESHOLD}
	candidates = satisfying if satisfying else all_results   # fallback: take best anyway

	best_overall = max(candidates.values(), key=lambda r: r["aep_no_yaw"])

	if not satisfying:
		print(f"\nWARNING: No run met penalty < {PEN_THRESHOLD}. Showing best available.")

	print(f"\n=== Best layout: {best_overall['n']} turbines, type {turbines[best_overall['t_idx']]} ===")
	print(f"AEP (no yaw) : {best_overall['aep_no_yaw']/1e9:.3f} GWh/yr")
	if best_overall['aep_yaw'] is not None:
		print(f"AEP (yaw)    : {best_overall['aep_yaw']/1e9:.3f} GWh/yr")
		print(f"Yaw gain     : +{((best_overall['aep_yaw']/best_overall['aep_no_yaw']-1)*100):.1f}%")
	print(f"Penalty      : {best_overall['total_pen']:.6f}")
	for i, (x, y) in enumerate(best_overall["positions"]):
		print(f"  Turbine {i}: x={x:.4f} km, y={y:.4f} km")

	# Expose for plotting
	N_TURBINES = best_overall["n"]
	positions  = best_overall["positions"]
	x0_start   = best_overall["x0"]
	aep        = best_overall["aep_no_yaw"]

	print("\n=== Start position (random seed) ===")
	for i, (x, y) in enumerate(x0_start):
		print(f"  Turbine {i}: x={x:.4f} km, y={y:.4f} km")

	print("=== Best positions found ===")
	for i, (x, y) in enumerate(best_overall["positions"]):
		print(f"  Turbine {i}: x={x:.4f} km, y={y:.4f} km")

	# Save the coordinates to a file
	N_TURBINES = best_overall["n"]
	BEST_T_IDX = best_overall["t_idx"]
	with open(f"optimized_layout_n{N_TURBINES}_t{BEST_T_IDX}.txt", "w") as f:
		f.write(f"=== Best layout: {N_TURBINES} turbines, type {turbines[BEST_T_IDX]} ===\n")
		f.write(f"AEP (no yaw): {best_overall['aep_no_yaw']/1e9:.3f} GWh/yr\n")
		if best_overall['aep_yaw'] is not None:
			f.write(f"AEP (yaw)   : {best_overall['aep_yaw']/1e9:.3f} GWh/yr\n")
			f.write(f"Yaw gain    : +{((best_overall['aep_yaw']/best_overall['aep_no_yaw']-1)*100):.1f}%\n")
		f.write(f"Penalty     : {best_overall['total_pen']:.6f}\n")
		for i, (x, y) in enumerate(best_overall["positions"]):
			f.write(f"  Turbine {i}: x={x:.4f} km, y={y:.4f} km\n")
		f.write("\n=== Start position (random seed) ===\n")
		for i, (x, y) in enumerate(x0_start):
			f.write(f"  Turbine {i}: x={x:.4f} km, y={y:.4f} km\n")

	if best_overall.get("yaw_optimal") is not None:
		yaw_npy = f"optimized_layout_n{N_TURBINES}_t{BEST_T_IDX}_yaw.npy"
		np.save(yaw_npy, best_overall["yaw_optimal"])
		print(f"Saved optimal yaw angles to {yaw_npy}")

	print(f"\nBest AEP (no yaw): {best_overall['aep_no_yaw'] / 1e9:.2f} GWh/yr")
	if best_overall['aep_yaw'] is not None:
		print(f"Best AEP (yaw)   : {best_overall['aep_yaw'] / 1e9:.2f} GWh/yr")

	# We also need to restore the global state so plotting uses the correct turbine bounds
	turbine_yaml_path = os.path.join(tub_lib, turbines[BEST_T_IDX] + ".yaml")
	with open(turbine_yaml_path, 'r') as f:
		turb_data = yaml.safe_load(f)

	fig, ax = plt.subplots(figsize=(20,12))

	ax.add_patch(plt.Polygon(list(zip(boundary_x, boundary_y)),closed=True, facecolor='green', edgecolor='blue', alpha=0.5, label="Valid Area"))

	first_road = True
	for (x0, y0), (x1, y1) in road_segments:
		buf = road_buffer_polygon(x0, y0, x1, y1, ROAD_BUFFER_M)
		ax.add_patch(plt.Polygon(buf, closed=True, facecolor='blue', edgecolor='blue', alpha=0.4, label="Invalid Area - Road" if first_road else "_nolegend_"))
		ax.plot([x0, x1], [y0, y1], color='black', label="Road" if first_road else "_nolegend_")
		first_road = False

	first_build = True
	for lat, lon in builds:
		ax.plot(lon, lat, 'x', color='darkgreen')
		ax.add_patch(plt.Circle((lon, lat), radius=(HH * 4), facecolor='red', edgecolor='red', alpha=0.5, label="Invalid Area - Building" if first_build else "_nolegend_"))
		first_build = False

	ax.plot(lon_k_town, lat_k_town, 'x', color='purple', label="Town")
	ax.add_patch(plt.Circle((lon_k_town, lat_k_town), radius=(1000), facecolor='purple', edgecolor='purple', alpha=0.5, label="Invalid Area - Town"))

	first_wt = True
	for lat, lon in wts:
		ax.plot(lon, lat, '>', color='navy', label="Wind Turbines" if first_wt else "_nolegend_")
		ax.add_patch(plt.Circle((lon, lat), radius=(HH * 4), facecolor='firebrick', edgecolor='firebrick', alpha=0.5, label="Invalid Area - Wind Turbine" if first_wt else "_nolegend_"))
		first_wt = False

	first_line = True
	for (x1, y1), (x2, y2) in LINE_SEGMENTS:
		ax.plot([x1, x2], [y1, y2], color='yellow', label="Field Border" if first_line else "_nolegend_")
		first_line = False

	# --- Start positions ---
	first_start = True
	for x, y in x0_start:
		ax.plot(x, y, 's', color='orange', markersize=10,
				label="Start position" if first_start else "_nolegend_")
		first_start = False

	# --- Best positions ---
	first_best = True
	for x, y in positions:
		ax.plot(x, y, '*', color='lime', markersize=14,
				label="Optimized turbine" if first_best else "_nolegend_")
		ax.add_patch(plt.Circle((x, y), radius=MIN_TURBINE_SPACING, facecolor='none', edgecolor='magenta', linestyle='-', linewidth=1.5, label="Min Spacing" if first_best else "_nolegend_"))
		first_best = False

	ax.set_aspect('equal')
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.set_xlim(-500, 4500)
	ax.set_ylim(-500, 2500)
	ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
	title = f"Best Layout: {N_TURBINES} Turbines ({turbines[BEST_T_IDX]}) | AEP: {best_overall['aep_no_yaw']/1e9:.2f} GWh"
	if best_overall['aep_yaw'] is not None:
		title += f" | AEP+yaw: {best_overall['aep_yaw']/1e9:.2f} GWh"
	ax.set_title(title, fontsize=16)

	plt.savefig(f"optimized_layout_n{N_TURBINES}_t{BEST_T_IDX}.png", dpi=300, bbox_inches='tight')
	plt.close(fig)
	print(f"Saved layout plot to optimized_layout_n{N_TURBINES}_t{BEST_T_IDX}.png")