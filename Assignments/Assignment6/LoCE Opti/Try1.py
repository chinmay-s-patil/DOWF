#!/usr/bin/env python
"""
Assignment 6 – Wind Farm Layout Optimisation with Substation Placement + LCoE
=============================================================================
Changes vs. original:
  1. Substation position (x_sub, y_sub) is now an optimisation variable
     appended to the flat layout vector.
  2. Substation reward: total cable distance from each turbine to the
     substation is MINIMISED via a reward term in the objective
     (adds AEP_WEIGHT-scaled benefit for short connections).
  3. After optimisation the substation is drawn on the layout map.
  4. LandBOSSE-equivalent BOS cost model (self-contained, no pandas 1.0.3
     dependency) computes and prints all key economic KPIs:
       – BoS cost breakdown (foundation, roads, collection, substation,
         grid connection, erection, development, management)
       – Capital cost (CapEx = turbine + BoS + development)
       – Capacity Factor (CF) from wind-rose AEP
       – LCOE  = (CapEx · FCR + OpEx) / AEP_annual
       – Profitability Index (PI)
       – Net Present Value (NPV), Simple Payback Period (SPP)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pathlib, yaml, os, time, json, shutil
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import weibull_min
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from floris_cupy import FlorisModel           # pyright: ignore
from floris_cupy.wind_data import WindRose    # pyright: ignore
import floris_cupy as _floris_pkg            # pyright: ignore
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(r"/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/LoCE Opti")

# ---------------------------------------------------------------------------
# Optimisation hyper-parameters
# ---------------------------------------------------------------------------
MAX_ITER   = 250
N_PARTICLES = 50
MAX_WORKERS = 5
N_MIN, N_MAX = 16, 16
SAVE_PLOTS   = True

LINE_PENALTY_WEIGHT          = 1e9
BOUNDARY_PENALTY_WEIGHT      = 1e8
EXCLUSION_PENALTY_WEIGHT     = 1e7
INTER_TURBINE_PENALTY_WEIGHT = 1e7
AEP_WEIGHT                   = 1e4

# Cable reward weight: reward = CABLE_REWARD_WEIGHT / (1 + total_cable_km)
# Positive reward reduces the objective, pulling substation to centre-of-mass.
CABLE_REWARD_WEIGHT = 1e6   # tune relative to AEP_WEIGHT

# ---------------------------------------------------------------------------
# Turbine library
# ---------------------------------------------------------------------------
tub_lib  = r"./turbineData/"
turbines = ["IEA_3_4MW", "BAR_BAU_IEA_3.3MW", "BAR_BAU_LSP_3.25MW"]
turbines = ["BAR_BAU_IEA_3.3MW"]

turbine_yaml_path = os.path.join(tub_lib, turbines[0] + ".yaml")
with open(turbine_yaml_path, 'r') as f:
    turb_data = yaml.safe_load(f)

HH                = turb_data['hub_height']
ROTOR_DIAMETER_M  = turb_data['rotor_diameter']
MIN_TURBINE_SPACING = 2 * ROTOR_DIAMETER_M

# ---------------------------------------------------------------------------
# Denmark site parameters (unchanged from original)
# ---------------------------------------------------------------------------
d_ws              = 0.17
d_ti              = 12 / 100
d_fuel_cost       = 9.5
d_line_freq       = 50
d_standard_V      = 220
d_interconnect_V  = 100
d_rent            = 15000
d_o_and_m         = 0.012     # fraction of CapEx per year
d_discount        = 3.6 / 100 # 3.6 %
d_life_time       = 20
d_construction_time = 12      # months

lattitude_degree_to_m    = 111000
longitude_degree_to_m_at_55 = 63000

# ---------------------------------------------------------------------------
# Coordinate helpers (unchanged)
# ---------------------------------------------------------------------------
def get_ref(stri):
    lattLB = 55 + 14/60 + 39.2/3600
    longLB = 9  +  0/60 + 41.3/3600
    latt_deg = float(stri.strip().split("°")[0])
    latt_min = float(stri.strip().split("'")[0].split("°")[1].replace("\\", ""))
    latt_sec = float(stri.strip().split('"')[0].split("°")[1].split("'")[1].replace("\\", ""))
    long_deg = float(stri.strip().split("N")[1].split("°")[0])
    long_min = float(stri.strip().split("N")[1].split("'")[0].split("°")[1].replace("\\", ""))
    long_sec = float(stri.strip().split("N")[1].split('"')[0].split("°")[1].split("'")[1].replace("\\", ""))
    latt = latt_deg + latt_min/60 + latt_sec/3600
    long = long_deg + long_min/60 + long_sec/3600
    latt = (latt - lattLB) * lattitude_degree_to_m
    long = (long - longLB) * longitude_degree_to_m_at_55
    return latt, long

def get_ref_WGS84(latt, long):
    lattLB = 55 + 14/60 + 39.2/3600
    longLB = 9  +  0/60 + 41.3/3600
    latt = (latt - lattLB) * lattitude_degree_to_m
    long = (long - longLB) * longitude_degree_to_m_at_55
    return latt, long

# ---------------------------------------------------------------------------
# Boundaries, roads, buildings, turbines (all unchanged from original)
# ---------------------------------------------------------------------------
lattLB, longLB = get_ref(r'55°14\'39.2\"N 9°00\'41.3\"E')
lattRB, longRB = get_ref(r'55°14\'53.8\"N 9°03\'30.7\"E')
lattLT, longLT = get_ref(r'55°15\'48.1\"N 9°00\'48.7\"E')
lattRT, longRT = get_ref(r'55°15\'41.7\"N 9°04\'04.8\"E')

lat_k_start, lon_k_start = get_ref(r'55°14\'39.2\"N 9°00\'41.3\"E')
lat_k_1,     lon_k_1     = get_ref(r'55°14\'49.0\"N 9°02\'23.0\"E')
lat_k_2,     lon_k_2     = get_ref(r'55°14\'53.0\"N 9°02\'60.0\"E')
lat_k_end,   lon_k_end   = get_ref(r'55°15\'03.6\"N 9°03\'38.0\"E')
lat_t_start, lon_t_start = get_ref(r'55°15\'47.1\"N 9°02\'07.5\"E')
lat_t_end,   lon_t_end   = get_ref(r'55°14\'42.8\"N 9°02\'24.7\"E')
lat_m_start, lon_m_start = get_ref(r'55°15\'19.4\"N 9°02\'15.0\"E')
lat_m_1,     lon_m_1     = get_ref(r'55°15\'18.0\"N 9°02\'30.4\"E')
lat_m_2,     lon_m_2     = get_ref(r'55°15\'33.2\"N 9°02\'35.7\"E')
lat_m_3,     lon_m_3     = get_ref(r'55°15\'54.6\"N 9°02\'49.1\"E')
lat_m_4,     lon_m_4     = get_ref(r'55°15\'46.0\"N 9°02\'43.7\"E')
lat_m_5,     lon_m_5     = get_ref(r'55°15\'45.0\"N 9°02\'50.1\"E')
lat_m_6,     lon_m_6     = get_ref(r'55°15\'45.6\"N 9°02\'54.6\"E')
lat_m_7,     lon_m_7     = get_ref(r'55°15\'43.3\"N 9°03\'10.5\"E')
lat_m_end,   lon_m_end   = get_ref(r'55°15\'54.6\"N 9°02\'49.1\"E')
lat_u_start, lon_u_start = get_ref(r'55°15\'41.7\"N 9°04\'04.8\"E')
lat_u_1,     lon_u_1     = get_ref(r'55°15\'41.9\"N 9°03\'46.7\"E')
lat_u_2,     lon_u_2     = get_ref(r'55°15\'49.5\"N 9°03\'18.2\"E')
lat_u_end,   lon_u_end   = get_ref(r'55°15\'54.6\"N 9°02\'49.1\"E')
lat_s_start, lon_s_start = get_ref(r'55°15\'02.2\"N 9°02\'19.4\"E')
lat_s_end,   lon_s_end   = get_ref(r'55°15\'07.4\"N 9°00\'48.6\"E')
lat_e_start, lon_e_start = get_ref(r'55°15\'48.1\"N 9°00\'48.7\"E')
lat_e_1,     lon_e_1     = get_ref(r'55°15\'07.4\"N 9°00\'48.6\"E')
lat_e_end,   lon_e_end   = get_ref(r'55°14\'39.2\"N 9°00\'41.3\"E')
lat_k_town,  lon_k_town  = get_ref(r'55°15\'46.7\"N 9°04\'46.7\"E')

builds = []
builds.append(get_ref_WGS84(55.26376264248587,  9.056352828928594))
builds.append(get_ref_WGS84(55.26231259217023,  9.050049639108959))
builds.append(get_ref_WGS84(55.2646173108904,   9.053834297505002))
builds.append(get_ref_WGS84(55.26411229432016,  9.051162083724197))
builds.append(get_ref_WGS84(55.26576740553818,  9.051293190948849))
builds.append(get_ref_WGS84(55.264889460129645, 9.045758966509334))
builds.append(get_ref_WGS84(55.2671271658051,   9.044682957544456))
builds.append(get_ref_WGS84(55.26829141385979,  9.049309200476348))
builds.append(get_ref_WGS84(55.244789869056724, 9.018664063980859))
builds.append(get_ref_WGS84(55.2442058697137,   9.013678153589987))
builds.append(get_ref(r'55°15\'34.3\"N 9°03\'57.9\"E'))
builds.append(get_ref_WGS84(55.262889159277975, 9.059533065052847))
builds.append(get_ref_WGS84(55.262144456274946, 9.060026613116126))
builds.append(get_ref_WGS84(55.26772112772359,  9.014694124807043))

wts = []
wts.append(get_ref_WGS84(55.2518537387715,  9.067113095780657))
wts.append(get_ref_WGS84(55.25005608478371, 9.065860880416732))
wts.append(get_ref_WGS84(55.24838910327743, 9.064680783026962))
wts.append(get_ref_WGS84(55.24662865506457, 9.063393833038278))
wts.append(get_ref_WGS84(55.25122021158104, 9.070125865365986))
wts.append(get_ref_WGS84(55.24940653519977, 9.068889527809011))
wts.append(get_ref_WGS84(55.2476480890615,  9.067588117653926))
wts.append(get_ref_WGS84(55.24588890783806, 9.066452109190593))

ROAD_BUFFER_M = 15.0
boundary_x = [longLB, longRB, longRT, longLT]
boundary_y = [lattLB, lattRB, lattRT, lattLT]

road_segments_normal = [
    ((lon_k_start, lat_k_start), (lon_k_1,   lat_k_1)),
    ((lon_k_1,     lat_k_1),     (lon_k_2,   lat_k_2)),
    ((lon_k_2,     lat_k_2),     (lon_k_end, lat_k_end)),
    ((lon_m_start, lat_m_start), (lon_m_1,   lat_m_1)),
    ((lon_m_1,     lat_m_1),     (lon_m_2,   lat_m_2)),
    ((lon_m_2,     lat_m_2),     (lon_m_3,   lat_m_3)),
    ((lon_m_4,     lat_m_4),     (lon_m_5,   lat_m_5)),
    ((lon_m_5,     lat_m_5),     (lon_m_6,   lat_m_6)),
    ((lon_m_6,     lat_m_6),     (lon_m_7,   lat_m_7)),
    ((lon_m_4,     lat_m_4),     (lon_m_end, lat_m_end)),
    ((lon_u_start, lat_u_start), (lon_u_1,   lat_u_1)),
    ((lon_u_1,     lat_u_1),     (lon_u_2,   lat_u_2)),
    ((lon_u_2,     lat_u_2),     (lon_u_end, lat_u_end)),
    ((lon_s_start, lat_s_start), (lon_s_end, lat_s_end)),
    ((lon_e_start, lat_e_start), (lon_e_1,   lat_e_1)),
    ((lon_e_1,     lat_e_1),     (lon_e_end, lat_e_end)),
]
road_segments_toft = [((lon_t_start, lat_t_start), (lon_t_end, lat_t_end))]
road_segments = road_segments_normal + road_segments_toft

# Field border points and connectivity (unchanged)
fb = []
fb.append(get_ref_WGS84(55.263378079208664, 9.013538798791178))
fb.append(get_ref_WGS84(55.26179918912272,  9.023299488092954))
fb.append(get_ref_WGS84(55.259586009260396, 9.013611349046208))
fb.append(get_ref_WGS84(55.259193935675405, 9.020673691208579))
fb.append(get_ref_WGS84(55.2595214708722,   9.0230302748183))
fb.append(get_ref_WGS84(55.259378271002554, 9.025777960420816))
fb.append(get_ref_WGS84(55.26158955844181,  9.025215304079365))
fb.append(get_ref_WGS84(55.26133286769302,  9.025996682092389))
fb.append(get_ref_WGS84(55.26408211153519,  9.026229682092394))
fb.append(get_ref_WGS84(55.262858338573665, 9.0165206820924))
fb.append(get_ref_WGS84(55.26297458335403,  9.025491682092403))
fb.append(get_ref_WGS84(55.263673866522346, 9.016592317674311))
fb.append(get_ref_WGS84(55.25932834020308,  9.017505682092377))
fb.append(get_ref_WGS84(55.259174200214076, 9.020072631023627))
fb.append(get_ref_WGS84(55.26226868149219,  9.020374153862859))
fb.append(get_ref_WGS84(55.26264444703107,  9.01778336389916))
fb.append(get_ref_WGS84(55.25756314091412,  9.013645153862841))
fb.append(get_ref_WGS84(55.25720245038016,  9.017287363899133))
fb.append(get_ref_WGS84(55.256221450983844, 9.017067363899129))
fb.append(get_ref_WGS84(55.25663427915377,  9.013682635062942))
fb.append(get_ref_WGS84(55.25227933900355,  9.013579761751688))
fb.append(get_ref_WGS84(55.252772510595804, 9.014885728317171))
fb.append(get_ref_WGS84(55.25369862479202,  9.021289363899083))
fb.append(get_ref_WGS84(55.254410107761316, 9.023465761751694))
fb.append(get_ref_WGS84(55.25402528015716,  9.027360728317186))
fb.append(get_ref_WGS84(55.25583022731078,  9.028793728719485))
fb.append(get_ref_WGS84(55.25610728445766,  9.028820364282486))
fb.append(get_ref_WGS84(55.25651058312273,  9.024429509541378))
fb.append(get_ref_WGS84(55.26062528271976,  9.028256682092394))
fb.append(get_ref_WGS84(55.256452568469896, 9.025560476688629))
fb.append(get_ref_WGS84(55.25670462338135,  9.022741077788233))
fb.append(get_ref_WGS84(55.25918641280878,  9.021083364301436))
fb.append(get_ref_WGS84(55.256852469712854, 9.020928317655349))
fb.append(get_ref_WGS84(55.25652775471802,  9.024627238377558))
fb.append(get_ref_WGS84(55.259386169237,    9.024882635465257))
fb.append(get_ref_WGS84(55.25618541151709,  9.028140602795677))
fb.append(get_ref_WGS84(55.26030105520822,  9.029186999883349))
fb.append(get_ref_WGS84(55.256087112765265, 9.02913068209239))
fb.append(get_ref_WGS84(55.25957099829968,  9.031465682092385))
fb.append(get_ref_WGS84(55.255880055548246, 9.031405999883319))
fb.append(get_ref_WGS84(55.26413663029794,  9.028381995441729))
fb.append(get_ref_WGS84(55.259911107076,    9.030403999481049))
fb.append(get_ref_WGS84(55.26419873631289,  9.030540835089983))
fb.append(get_ref_WGS84(55.264219612989415, 9.032889039844271))
fb.append(get_ref_WGS84(55.25065491000093,  9.038712578608001))
fb.append(get_ref_WGS84(55.255770584185186, 9.032449682092357))
fb.append(get_ref_WGS84(55.25509552695409,  9.033109317655345))
fb.append(get_ref_WGS84(55.253796055708506, 9.037872317655324))
fb.append(get_ref_WGS84(55.25285207203288,  9.041681856493883))
fb.append(get_ref_WGS84(55.252617108197704, 9.04269609273523))
fb.append(get_ref_WGS84(55.25121528398915,  9.048198153862828))
fb.append(get_ref_WGS84(55.25129210840144,  9.047859728317148))
fb.append(get_ref_WGS84(55.251047050818485, 9.048852635062875))
fb.append(get_ref_WGS84(55.25053705595912,  9.050874271047139))
fb.append(get_ref_WGS84(55.25024769984094,  9.051968331631807))
fb.append(get_ref_WGS84(55.24961122412278,  9.054430999480939))
fb.append(get_ref_WGS84(55.24943110868766,  9.055107270644779))
fb.append(get_ref_WGS84(55.248740706183014, 9.05707176175165))
fb.append(get_ref_WGS84(55.24828555772259,  9.058515940225304))
fb.append(get_ref_WGS84(55.2487046465304,   9.058633231584592))
fb.append(get_ref_WGS84(55.25078105083899,  9.05981499948098))
fb.append(get_ref_WGS84(55.250269514694274, 9.057995425026625))
fb.append(get_ref_WGS84(55.24951053473027,  9.012836302097487))
fb.append(get_ref_WGS84(55.2492163284551,   9.015688339837745))
fb.append(get_ref_WGS84(55.24902019407573,  9.017395292572216))
fb.append(get_ref_WGS84(55.24892447387547,  9.0184323152763))
fb.append(get_ref_WGS84(55.24878115454767,  9.019663006773676))
fb.append(get_ref_WGS84(55.24831953029172,  9.02418918568632))
fb.append(get_ref_WGS84(55.24821788573318,  9.024989437744052))
fb.append(get_ref_WGS84(55.24806169968701,  9.026402926432059))
fb.append(get_ref_WGS84(55.247302845722054, 9.028205999885529))
fb.append(get_ref_WGS84(55.24580125104239,  9.027815656609373))
fb.append(get_ref_WGS84(55.24897347126353,  9.02874256913812))
fb.append(get_ref_WGS84(55.248965928164225, 9.030773871735189))
fb.append(get_ref_WGS84(55.248965928164225, 9.033837367236718))
fb.append(get_ref_WGS84(55.24895633622263,  9.044116605337392))
fb.append(get_ref_WGS84(55.24979178168034,  9.046247711366332))
fb.append(get_ref_WGS84(55.25049830197436,  9.046995773069371))
fb.append(get_ref_WGS84(55.25139549996574,  9.038552359965397))
fb.append(get_ref_WGS84(55.25140467274515,  9.0406337541843))
fb.append(get_ref_WGS84(55.25189728172065,  9.016409448109773))
fb.append(get_ref_WGS84(55.251811220618244, 9.018160958747437))
fb.append(get_ref_WGS84(55.25164483529184,  9.02054663702977))
fb.append(get_ref_WGS84(55.25131780003252,  9.02660646116331))
fb.append(get_ref_WGS84(55.244250737695246, 9.011482661115275))
fb.append(get_ref_WGS84(55.24477501186283,  9.01728546202535))
fb.append(get_ref_WGS84(55.24535959255163,  9.023391898093454))
fb.append(get_ref_WGS84(55.24600560678458,  9.030114777493683))
fb.append(get_ref_WGS84(55.246319880172095, 9.033192906826937))
fb.append(get_ref_WGS84(55.24652939438348,  9.035321563430532))
fb.append(get_ref_WGS84(55.24695714915413,  9.03973201728221))
fb.append(get_ref_WGS84(55.24763805540208,  9.046485524851162))
fb.append(get_ref_WGS84(55.2478260615601,   9.048451545900498))
fb.append(get_ref_WGS84(55.248286613602474, 9.050957010448316))
fb.append(get_ref_WGS84(55.25915885833599,  9.032726486977777))
fb.append(get_ref_WGS84(55.25863610853423,  9.034523567033938))
fb.append(get_ref_WGS84(55.25821380403339,  9.03587348591767))
fb.append(get_ref_WGS84(55.25797248022116,  9.03680642946182))
fb.append(get_ref_WGS84(55.257572629278556, 9.042665116894664))
fb.append(get_ref_WGS84(55.25744442411859,  9.045159582974144))
fb.append(get_ref_WGS84(55.257369009120254, 9.045986660571753))
fb.append(get_ref_WGS84(55.257210637171696, 9.046807121570005))
fb.append(get_ref_WGS84(55.25707866006572,  9.047316601383432))
fb.append(get_ref_WGS84(55.25671289264315,  9.048236311680547))
fb.append(get_ref_WGS84(55.25616612064468,  9.04962580208081))
fb.append(get_ref_WGS84(55.25615480804112,  9.049817684088465))
fb.append(get_ref_WGS84(55.25584342283734,  9.051734628344061))
fb.append(get_ref_WGS84(55.25569258663375,  9.052654338656616))
fb.append(get_ref_WGS84(55.25546256032172,  9.054209244552972))
fb.append(get_ref_WGS84(55.25540222533073,  9.054566542084467))
fb.append(get_ref_WGS84(55.254757389378774, 9.05684927628772))
fb.append(get_ref_WGS84(55.254546212744025, 9.057610187672527))
fb.append(get_ref_WGS84(55.253003835214464, 9.06122947933416))
fb.append(get_ref_WGS84(55.25186370701055,  9.063664363800962))
fb.append(get_ref_WGS84(55.25540798131461,  9.037532295481142))
fb.append(get_ref_WGS84(55.25494632862662,  9.041759456936028))
fb.append(get_ref_WGS84(55.25401590108877,  9.04368106540857))
fb.append(get_ref_WGS84(55.25370710330228,  9.044351617663736))
fb.append(get_ref_WGS84(55.25300389147301,  9.045842925887026))
fb.append(get_ref_WGS84(55.252716488269876, 9.046572486739313))
fb.append(get_ref_WGS84(55.25251775079296,  9.047361056186139))
fb.append(get_ref_WGS84(55.25262170590054,  9.048407117713682))
fb.append(get_ref_WGS84(55.25297132517922,  9.050121459284552))
fb.append(get_ref_WGS84(55.25299946123608,  9.050583379140999))
fb.append(get_ref_WGS84(55.25311218386971,  9.052104952809133))
fb.append(get_ref_WGS84(55.253182197391546, 9.053303233638285))
fb.append(get_ref_WGS84(55.2489492215643,   9.035925594736396))
fb.append(get_ref_WGS84(55.25018304982121,  9.04815845711716))
fb.append(get_ref_WGS84(55.24939306385255,  9.049460121330727))
fb.append(get_ref_WGS84(55.25020204451423,  9.051236684056777))
fb.append(get_ref_WGS84(55.25038118718926,  9.05139879127014))
fb.append(get_ref_WGS84(55.249790956165654, 9.049857118587944))
fb.append(get_ref_WGS84(55.249217688250845, 9.048365070562902))
fb.append(get_ref_WGS84(55.2555435008887,   9.033276601201356))
fb.append(get_ref_WGS84(55.264108607028334, 9.025373286768398))
fb.append(get_ref_WGS84(55.264497236481404, 9.013682351522888))
fb.append(get_ref_WGS84(55.26445934034586,  9.014768666111596))
fb.append(get_ref_WGS84(55.26222072409856,  9.020559348085323))
fb.append(get_ref_WGS84(55.26418551001019,  9.020715932355472))
fb.append(get_ref_WGS84(55.26533606796102,  9.020869460549681))
fb.append(get_ref_WGS84(55.26423758322738,  9.034172048302294))
fb.append(get_ref_WGS84(55.26408396075054,  9.035172603634564))
fb.append(get_ref_WGS84(55.262703696505014, 9.035492611146793))
fb.append(get_ref_WGS84(55.25893475007203,  9.036506636801228))
fb.append(get_ref_WGS84(55.260523466859716, 9.038730131916617))
fb.append(get_ref_WGS84(55.2598119635145,   9.043536302127572))
fb.append(get_ref_WGS84(55.26026030957294,  9.04577690105154))
fb.append(get_ref_WGS84(55.26028954935754,  9.046358430543258))
fb.append(get_ref_WGS84(55.26031878912063,  9.048239849487048))
fb.append(get_ref_WGS84(55.259963028884584, 9.051091779940204))
fb.append(get_ref_WGS84(55.26217105766619,  9.052207342482957))
fb.append(get_ref_WGS84(55.26277933705035,  9.045476508688918))
fb.append(get_ref_WGS84(55.26252746122752,  9.047171769134632))
fb.append(get_ref_WGS84(55.2626603190034,   9.048483288677163))
fb.append(get_ref_WGS84(55.26255790784042,  9.049406209836722))
fb.append(get_ref_WGS84(55.261908811753464, 9.05338078055656))
fb.append(get_ref_WGS84(55.26180555679392,  9.054201376473408))
fb.append(get_ref_WGS84(55.25892832516542,  9.033509479163001))
fb.append(get_ref_WGS84(55.2529618128421,   9.016351526687206))
fb.append(get_ref_WGS84(55.25185601311282,  9.01704327120105))
fb.append(get_ref_WGS84(55.25198074991654,  9.023146700668205))
fb.append(get_ref_WGS84(55.25152705238807,  9.023353577122455))
fb.append(get_ref_WGS84(55.2516527538692,   9.021234827489986))
fb.append(get_ref_WGS84(55.25887838703052,  9.047619181879195))
fb.append(get_ref_WGS84(55.25814163346454,  9.050137202174618))
fb.append(get_ref_WGS84(55.25843719322199,  9.051550300131455))
fb.append(get_ref_WGS84(55.26156327367187,  9.039884487753524))
fb.append(get_ref_WGS84(55.263449231476514, 9.04275850751901))
fb.append(get_ref_WGS84(55.263935003358704, 9.043528159086737))
fb.append(get_ref_WGS84(55.26545313487896,  9.045277736345772))
fb.append(get_ref_WGS84(55.261380762765164, 9.041103908244152))
fb.append(get_ref_WGS84(55.25121959443652,  9.029075578241915))
fb.append(get_ref_WGS84(55.25207752492171,  9.029886077128559))
fb.append(get_ref_WGS84(55.252226879348385, 9.03088548928953))
fb.append(get_ref_WGS84(55.25209141838044,  9.032140848467339))
fb.append(get_ref_WGS84(55.25478894761439,  9.03422053681479))
fb.append(get_ref_WGS84(55.25089458066402,  9.033831153491723))
fb.append(get_ref_WGS84(55.2510266889164,   9.0313372460178))
fb.append(get_ref_WGS84(55.25922779364273,  9.055538834641853))
fb.append(get_ref_WGS84(55.260399353156,    9.06107211474652))
fb.append(get_ref_WGS84(55.26030290457672,  9.061523456290434))
fb.append(get_ref_WGS84(55.2618621279051,   9.061932484564608))
fb.append(get_ref_WGS84(55.259009689282784, 9.056408673842139))
fb.append(get_ref_WGS84(55.2630146878491,   9.057609051933003))
fb.append(get_ref_WGS84(55.263784664718784, 9.05484126848888))
fb.append(get_ref_WGS84(55.26200860999288,  9.053027948514249))
fb.append(get_ref_WGS84(55.2640268434764,   9.053621031847511))
fb.append(get_ref_WGS84(55.25395065621873,  9.059211816588933))
fb.append(get_ref_WGS84(55.25980755068798,  9.062498282003395))
fb.append(get_ref_WGS84(55.2593051636274,   9.064018293176174))
fb.append(get_ref_WGS84(55.258993452388395, 9.064963938389333))
fb.append(get_ref_WGS84(55.25303711642365,  9.061219710178927))
fb.append(get_ref_WGS84(55.252257339242966, 9.063067389619157))
fb.append(get_ref_WGS84(55.258736260274425, 9.067214090051877))
fb.append(get_ref_WGS84(55.26157599830298,  9.064793067834298))
fb.append(get_ref_WGS84(55.26157938483438,  9.067932906664897))

lines = [
    [11,9], [142,95],
    [0,9,15,14,1,6,7,28,36,41,38,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,192],
    [2,12,13,3,31,4,34,5,7,29],
    [16,17,32,30,27,33,29,35,26,37,39,45,46,47,48,49,51,50,52,53,54,55,56,57,58],
    [62,63,64,65,66,67,68,69,72,73,74,75,76,77,51],
    [115,116,117,118,119,120,121,122,123,124,125,108,54],
    [78,79,48], [20,21,22,23,24,25,26], [14,138], [69,70,71],
    [85,65], [80,63], [64,81], [66,82], [86,67], [68,83],
    [87,73], [88,74], [89,126], [76,91], [77,127,92],
    [132,128,131,129,130], [99,117], [49,116], [101,118],
    [9,11], [15,12], [14,13], [31,32], [28,35], [36,37],
    [41,42], [6,10,134], [38,39], [7,8], [1,4], [147,152],
    [45,133,157], [133,96], [43,94], [16,17,18,19],
    [96,143,144,145,146,147,148,149,150], [159,158],
    [22,160,161], [160,162], [121,50], [52,122], [53,124],
    [123,106], [102,163,148], [163,164,165,155],
    [144,166,167,168,169], [166,170,167], [171,172,173,174,177],
    [175,174,177], [175,176], [111,179,180,181], [109,178,182,183],
    [107,156,184], [185,186], [187,188,189,190,193], [112,190],
    [103,119], [105,121],
]

# ---------------------------------------------------------------------------
# Wind Rose (Kastrup)
# ---------------------------------------------------------------------------
binsize = 3
WD_BINS  = np.array([0,30,60,90,120,150,180,210,240,270,300,330], dtype=float)
WB_SCALE = np.array([9.785,8.284,8.721,9.633,10.114,8.340,
                     8.936,10.759,11.710,11.363,10.682,8.965])
WB_SHAPE = np.array([2.306,2.089,1.888,1.935,1.945,1.902,
                     1.909,1.910,1.968,2.049,2.064,1.928])
FREQ_WD  = np.array([14.71,6.09,6.16,8.17,9.58,6.05,
                     5.34,7.27,8.00,14.60,7.78,6.25]) / 100.0
WS_BINS   = np.arange(3.0, 26.0, 1.0)
SITE1_TI  = d_ti
WIND_SHEAR= d_ws

freq_table = np.zeros((len(WD_BINS), len(WS_BINS)))
for i, (k, lam) in enumerate(zip(WB_SHAPE, WB_SCALE)):
    p_ws = weibull_min.pdf(WS_BINS, c=k, scale=lam) * binsize
    p_ws /= p_ws.sum()
    freq_table[i, :] = FREQ_WD[i] * p_ws

wind_rose = WindRose(
    wind_directions=WD_BINS,
    wind_speeds=WS_BINS,
    ti_table=SITE1_TI,
    freq_table=freq_table,
)

# ---------------------------------------------------------------------------
# Constraint helpers (unchanged)
# ---------------------------------------------------------------------------
LINE_SEGMENTS = []
for line in lines:
    for i in range(len(line) - 1):
        p1 = (fb[line[i]][1],   fb[line[i]][0])
        p2 = (fb[line[i+1]][1], fb[line[i+1]][0])
        LINE_SEGMENTS.append([p1, p2])

boundary_polygon = Polygon(zip(boundary_x, boundary_y))

def road_buffer_polygon(x0, y0, x1, y1, width):
    dx, dy = x1 - x0, y1 - y0
    length = np.hypot(dx, dy)
    px, py = -dy / length, dx / length
    offset = width / 2
    return [(x0+px*offset, y0+py*offset),(x1+px*offset, y1+py*offset),
            (x1-px*offset, y1-py*offset),(x0-px*offset, y0-py*offset)]

def _point_to_segment_dist_sq(px, py, x1, y1, x2, y2):
    dx, dy = x2-x1, y2-y1
    seg_len_sq = dx*dx + dy*dy
    if seg_len_sq == 0:
        return (px-x1)**2 + (py-y1)**2
    t = max(0.0, min(1.0, ((px-x1)*dx + (py-y1)*dy) / seg_len_sq))
    cx, cy = x1+t*dx, y1+t*dy
    return (px-cx)**2 + (py-cy)**2

def min_dist_to_any_line(x, y):
    best_field = np.inf
    for (x1,y1),(x2,y2) in LINE_SEGMENTS:
        d = _point_to_segment_dist_sq(x,y,x1,y1,x2,y2)
        if d < best_field: best_field = d
    dist_field = np.sqrt(best_field) if best_field != np.inf else np.inf
    return max(0.0, dist_field - 50.0)**2

def line_penalty(positions):
    return sum(min_dist_to_any_line(x,y) for x,y in positions)

def boundary_penalty(positions):
    total = 0.0
    for x,y in positions:
        p = Point(x,y)
        if not boundary_polygon.contains(p):
            total += boundary_polygon.exterior.distance(p)**2
    return total

ROTOR_RADIUS_M = ROTOR_DIAMETER_M / 2.0
EXCLUSION_ZONES = []
BUILD_RADIUS = HH * 4
for lat,lon in builds:
    EXCLUSION_ZONES.append((lon,lat,BUILD_RADIUS))
EXCLUSION_ZONES.append((lon_k_town,lat_k_town,1000.0))
WT_RADIUS = HH * 4
for lat,lon in wts:
    EXCLUSION_ZONES.append((lon,lat,WT_RADIUS))
ROAD_EXCL_NORMAL_M = ROTOR_RADIUS_M + 10.0
ROAD_EXCL_TOFT_M   = ROTOR_RADIUS_M + 15.0

def exclusion_penalty(positions):
    total = 0.0
    for tx,ty in positions:
        for cx,cy,r in EXCLUSION_ZONES:
            dist = np.sqrt((tx-cx)**2 + (ty-cy)**2)
            if dist < r:
                total += (r-dist)**2
        for (x1,y1),(x2,y2) in road_segments_normal:
            d = np.sqrt(_point_to_segment_dist_sq(tx,ty,x1,y1,x2,y2))
            if d < ROAD_EXCL_NORMAL_M:
                total += (ROAD_EXCL_NORMAL_M-d)**2
        for (x1,y1),(x2,y2) in road_segments_toft:
            d = np.sqrt(_point_to_segment_dist_sq(tx,ty,x1,y1,x2,y2))
            if d < ROAD_EXCL_TOFT_M:
                total += (ROAD_EXCL_TOFT_M-d)**2
    return total

def interturbine_penalty(positions):
    total = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i+1,n):
            dx = positions[i][0]-positions[j][0]
            dy = positions[i][1]-positions[j][1]
            dist = np.sqrt(dx**2+dy**2)
            if dist < MIN_TURBINE_SPACING:
                total += (MIN_TURBINE_SPACING-dist)**2
    return total

# ---------------------------------------------------------------------------
# Cable distance reward (CPU, called after extracting substation position)
# ---------------------------------------------------------------------------
def total_cable_km(turbine_positions, sub_x, sub_y):
    """Sum of straight-line distances from each turbine to the substation (km)."""
    total = 0.0
    for tx, ty in turbine_positions:
        total += np.sqrt((tx - sub_x)**2 + (ty - sub_y)**2)
    return total / 1000.0  # convert m → km

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
X_MIN = min(longLB, longLT)
X_MAX = max(longRB, longRT)
Y_MIN = min(lattLB, lattRB)
Y_MAX = max(lattLT, lattRT)

# ---------------------------------------------------------------------------
# GPU-batched objective factory
# (now receives N_TURBINES*2 + 2 variables; last 2 are substation x, y)
# ---------------------------------------------------------------------------
def _make_objective(n_turbines, fmodel_local):
    import cupy as cp

    def _get_seg_data(segs):
        if not segs:
            return None, None, None, None
        p1   = cp.array([(x1,y1) for ((x1,y1),(x2,y2)) in segs])
        p2   = cp.array([(x2,y2) for ((x1,y1),(x2,y2)) in segs])
        dxdy = p2 - p1
        lensq = cp.sum(dxdy**2, axis=1)
        lensq = cp.where(lensq==0, 1e-10, lensq)
        return p1, p2, dxdy, lensq

    l_p1,  l_p2,  l_dxdy,  l_lensq  = _get_seg_data(LINE_SEGMENTS)
    rn_p1, rn_p2, rn_dxdy, rn_lensq = _get_seg_data(road_segments_normal)
    rt_p1, rt_p2, rt_dxdy, rt_lensq = _get_seg_data(road_segments_toft)

    exc_arr = cp.array(EXCLUSION_ZONES)
    exc_xy  = exc_arr[:, :2]
    exc_r   = exc_arr[:, 2]
    FIELD_BORDER_MAX_M = 50.0

    def _dist_to_segs_batch(px, py, p1, dxdy, lensq):
        if p1 is None:
            return cp.full((px.shape[0], px.shape[1]), cp.inf)
        px_diff = px - p1[:, 0]
        py_diff = py - p1[:, 1]
        t = (px_diff*dxdy[:,0] + py_diff*dxdy[:,1]) / lensq
        t = cp.clip(t, 0.0, 1.0)
        cx = p1[:,0] + t*dxdy[:,0]
        cy = p1[:,1] + t*dxdy[:,1]
        dists = cp.sqrt((px-cx)**2 + (py-cy)**2)
        return cp.min(dists, axis=2)

    def _batched_objective(batch_pos_full):
        """
        batch_pos_full : (P, N_TURBINES+1, 2)
          first N_TURBINES rows → turbine positions
          last row              → substation position
        """
        n_particles = batch_pos_full.shape[0]
        batch_pos   = batch_pos_full[:, :n_turbines, :]    # (P, T, 2)
        sub_xy      = batch_pos_full[:, n_turbines, :]     # (P, 2)

        # --- AEP (CPU) ---
        aeps = cp.zeros(n_particles)
        for i in range(n_particles):
            layout = batch_pos[i]
            x_cpu = [float(layout[j,0]) for j in range(n_turbines)]
            y_cpu = [float(layout[j,1]) for j in range(n_turbines)]
            fmodel_local.set(layout_x=x_cpu, layout_y=y_cpu,
                             wind_data=wind_rose, wind_shear=d_ws)
            fmodel_local.run()
            aeps[i] = fmodel_local.get_farm_AEP() / 1e9
        values = aeps * AEP_WEIGHT

        # --- Cable length reward (reward = shorter cables → lower objective) ---
        # dist (P, T)
        diff_sub = batch_pos - sub_xy[:, None, :]   # (P, T, 2)
        dist_sub = cp.sqrt(cp.sum(diff_sub**2, axis=2))  # (P, T) in metres
        total_cable_m = cp.sum(dist_sub, axis=1)         # (P,)  sum over turbines
        # reward: subtract a value that grows as cable shrinks
        # We negate it in the final score so shorter cable → lower score
        cable_reward = CABLE_REWARD_WEIGHT / (1.0 + total_cable_m / 1000.0)

        # --- GPU penalties (turbines only) ---
        px = batch_pos[:, :, 0:1]
        py = batch_pos[:, :, 1:2]

        dist_field = _dist_to_segs_batch(px, py, l_p1, l_dxdy, l_lensq)
        line_pen   = cp.sum(cp.maximum(0.0, dist_field - FIELD_BORDER_MAX_M)**2, axis=1)

        dist_rn  = _dist_to_segs_batch(px, py, rn_p1, rn_dxdy, rn_lensq)
        road_n   = cp.sum(cp.maximum(0.0, ROAD_EXCL_NORMAL_M - dist_rn)**2, axis=1)
        dist_rt  = _dist_to_segs_batch(px, py, rt_p1, rt_dxdy, rt_lensq)
        road_t   = cp.sum(cp.maximum(0.0, ROAD_EXCL_TOFT_M  - dist_rt)**2, axis=1)
        road_pen = road_n + road_t

        diff_exc = batch_pos[:, :, None, :] - exc_xy[None, None, :, :]
        dist_exc = cp.sqrt(cp.sum(diff_exc**2, axis=3))
        exc_pen  = cp.sum(cp.maximum(0.0, exc_r[None,None,:] - dist_exc)**2, axis=(1,2))

        diff_inter = batch_pos[:,:,None,:] - batch_pos[:,None,:,:]
        dist_inter = cp.sqrt(cp.sum(diff_inter**2, axis=3))
        mask = cp.triu(cp.ones((n_turbines,n_turbines),dtype=bool), k=1)
        dist_inter = cp.where(mask, dist_inter, cp.inf)
        inter_pen  = cp.sum(cp.maximum(0.0, MIN_TURBINE_SPACING-dist_inter)**2, axis=(1,2))

        # Boundary penalty (CPU, Shapely)
        bound_pen_cpu = np.zeros(n_particles)
        batch_pos_cpu = batch_pos.get() if hasattr(batch_pos,'get') else batch_pos
        for i in range(n_particles):
            pos_list = [(batch_pos_cpu[i,j,0], batch_pos_cpu[i,j,1]) for j in range(n_turbines)]
            bound_pen_cpu[i] = boundary_penalty(pos_list)
        bound_pen = cp.array(bound_pen_cpu)

        penalty = (LINE_PENALTY_WEIGHT          * line_pen  +
                   EXCLUSION_PENALTY_WEIGHT      * road_pen  +
                   BOUNDARY_PENALTY_WEIGHT       * bound_pen +
                   EXCLUSION_PENALTY_WEIGHT      * exc_pen   +
                   INTER_TURBINE_PENALTY_WEIGHT  * inter_pen)

        # score: minimise → we want high AEP, short cables, low penalties
        score = -values - cable_reward + penalty
        return score, aeps, penalty

    return _batched_objective

# ---------------------------------------------------------------------------
# Layout plot helper (now draws substation)
# ---------------------------------------------------------------------------
def save_layout_plot(positions, sub_pos, n_turbines, filename, title=""):
    fig, ax = plt.subplots(figsize=(20,12))
    ax.add_patch(plt.Polygon(list(zip(boundary_x,boundary_y)),
                             closed=True, facecolor='green', edgecolor='blue',
                             alpha=0.5, label="Valid Area"))
    first_road = True
    for (x0,y0),(x1,y1) in road_segments:
        buf = road_buffer_polygon(x0,y0,x1,y1,ROAD_BUFFER_M)
        ax.add_patch(plt.Polygon(buf,closed=True,facecolor='blue',edgecolor='blue',
                                 alpha=0.4,label="Invalid Area - Road" if first_road else "_nolegend_"))
        ax.plot([x0,x1],[y0,y1],color='black',label="Road" if first_road else "_nolegend_")
        first_road = False
    first_build = True
    for lat,lon in builds:
        ax.plot(lon,lat,'x',color='darkgreen')
        ax.add_patch(plt.Circle((lon,lat),radius=HH*4,facecolor='red',edgecolor='red',
                                alpha=0.5,label="Invalid Area - Building" if first_build else "_nolegend_"))
        first_build = False
    ax.plot(lon_k_town,lat_k_town,'x',color='purple',label="Town")
    ax.add_patch(plt.Circle((lon_k_town,lat_k_town),radius=1000,facecolor='purple',
                            edgecolor='purple',alpha=0.5,label="Invalid Area - Town"))
    first_wt = True
    for lat,lon in wts:
        ax.plot(lon,lat,'>',color='navy',label="Existing Turbines" if first_wt else "_nolegend_")
        ax.add_patch(plt.Circle((lon,lat),radius=HH*4,facecolor='firebrick',edgecolor='firebrick',
                                alpha=0.5,label="Invalid Area - Wind Turbine" if first_wt else "_nolegend_"))
        first_wt = False
    first_line = True
    for line in lines:
        for l in range(len(line)-1):
            ax.plot((fb[line[l]][1],fb[line[l+1]][1]),(fb[line[l]][0],fb[line[l+1]][0]),
                    color='yellow',label="Field Border" if first_line else "_nolegend_")
            first_line = False
    # Turbines
    first_best = True
    for x,y in positions:
        ax.plot(x,y,'*',color='lime',markersize=14,
                label="Optimised Turbine" if first_best else "_nolegend_")
        ax.add_patch(plt.Circle((x,y),radius=MIN_TURBINE_SPACING,facecolor='none',
                                edgecolor='magenta',linestyle='-',linewidth=1.5,
                                label="Min Spacing" if first_best else "_nolegend_"))
        first_best = False
    # Cables from turbines to substation
    sx, sy = sub_pos
    for tx, ty in positions:
        ax.plot([tx, sx], [ty, sy], color='cyan', linewidth=0.8, alpha=0.6)
    # Substation
    ax.plot(sx, sy, 'D', color='gold', markersize=18, zorder=10, label="Substation")
    ax.annotate("SUB", xy=(sx, sy), xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold', color='gold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
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

# ---------------------------------------------------------------------------
# PSO (extended to handle N+1 "turbines", last slot = substation)
# ---------------------------------------------------------------------------
def run_optimization(n_turbines, seed=42, maxiter=50, disp=True, turbine_type_name=""):
    from pso_optimizer import BatchedGPUParticleSwarm

    # Bounds for turbines + 1 substation slot
    bounds = []
    for i in range(n_turbines + 1):   # +1 for substation
        bounds.append((X_MIN, X_MAX))
        bounds.append((Y_MIN, Y_MAX))

    fmodel_local = FlorisModel(r"/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/Docs/gch.yaml")
    abs_turb_path = os.path.abspath(turbine_yaml_path)
    fmodel_local.set(
        layout_x=[0.0]*n_turbines,
        layout_y=[0.0]*n_turbines,
        wind_data=wind_rose,
        wind_shear=d_ws,
        turbine_type=[abs_turb_path]*n_turbines
    )

    # Use n_turbines+1 as the PSO "turbine count" so the swarm dimension is correct
    pso = BatchedGPUParticleSwarm(
        n_particles=N_PARTICLES,
        n_turbines=n_turbines + 1,   # last slot → substation
        bounds=bounds,
        maxiter=maxiter,
        seed=seed,
        disp=disp,
        turbine_type_name=turbine_type_name
    )

    objective_fn = _make_objective(n_turbines, fmodel_local)

    if SAVE_PLOTS:
        out_dir = os.path.join("plots", turbine_type_name, str(n_turbines))
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs("history_logs",  exist_ok=True)
        os.makedirs("history_plots", exist_ok=True)
        pen_file      = f"history_logs/{turbine_type_name}_{n_turbines}_penalty_hist.txt"
        aep_file      = f"history_logs/{turbine_type_name}_{n_turbines}_AEP_hist.txt"
        hist_plot_file = f"history_plots/history_{turbine_type_name}_{n_turbines}.png"
        open(pen_file,'w').close(); open(aep_file,'w').close()
        local_iters, local_pens, local_aeps = [], [], []

        def on_new_best(flat_pos, score, aep, penalty, it):
            # flat_pos has (n_turbines+1)*2 elements
            pos_list = [(flat_pos[2*i], flat_pos[2*i+1]) for i in range(n_turbines)]
            sub_pos  = (flat_pos[2*n_turbines], flat_pos[2*n_turbines+1])
            title    = f"Iter {it} | Score:{score:.1f} | AEP:{aep:.3f} GWh"
            filename = os.path.join(out_dir, f"best_case_{it}.png")
            save_layout_plot(pos_list, sub_pos, n_turbines, filename, title)
            with open(pen_file,'a') as f: f.write(f"{penalty}\n")
            with open(aep_file,'a') as f: f.write(f"{aep}\n")
            local_iters.append(it); local_pens.append(penalty)
            local_aeps.append(aep/1e9 if aep>1e6 else aep)
            fig, ax1 = plt.subplots(figsize=(10,6))
            ax1.set_xlabel('Iteration'); ax1.set_ylabel('Penalty',color='tab:red')
            ax1.set_yscale('log')
            ax1.plot(local_iters,local_pens,color='tab:red',marker='o')
            ax1.axhline(1e-4,color='tab:red',linestyle='--',alpha=0.5)
            ax1.tick_params(axis='y',labelcolor='tab:red')
            ax2 = ax1.twinx(); ax2.set_ylabel('AEP (GWh)',color='tab:blue')
            ax2.plot(local_iters,local_aeps,color='tab:blue',marker='x')
            ax2.tick_params(axis='y',labelcolor='tab:blue')
            plt.title(f"Live History: {n_turbines} turbines ({turbine_type_name})")
            fig.tight_layout(); plt.savefig(hist_plot_file); plt.close(fig)
    else:
        on_new_best = None

    print(f"Starting PSO for n={n_turbines} (+1 substation)...")
    t0 = time.time()
    flat_best_pos, start_pos_cpu, best_score, history = pso.optimize(objective_fn, callback=on_new_best)
    pso_time = time.time() - t0

    best_positions = [(flat_best_pos[2*i], flat_best_pos[2*i+1]) for i in range(n_turbines)]
    best_substation = (flat_best_pos[2*n_turbines], flat_best_pos[2*n_turbines+1])
    start_positions = [(start_pos_cpu[2*i], start_pos_cpu[2*i+1]) for i in range(n_turbines)]

    # Clean AEP eval (turbines only)
    fmodel_local.set(
        layout_x=[p[0] for p in best_positions],
        layout_y=[p[1] for p in best_positions],
        wind_data=wind_rose, wind_shear=d_ws
    )
    fmodel_local.run()
    exact_aep = fmodel_local.get_farm_AEP()

    return flat_best_pos, start_positions, best_positions, best_substation, exact_aep, history, pso_time

# ---------------------------------------------------------------------------
# LandBOSSE-equivalent BOS cost calculator (self-contained)
# ---------------------------------------------------------------------------
def compute_lcoe(n_turbines, turbine_rating_MW, hub_height_m, rotor_diameter_m,
                 aep_gwh_yr, cable_total_km, distance_to_grid_km,
                 turbine_unit_cost_usd=None, verbose=True):
    """
    Faithful re-implementation of the LandBOSSE 2.3 cost sub-models
    using only NumPy (no pandas 1.0.3 requirement).

    All cost equations are taken verbatim from the LandBOSSE 2.3 source:
      SubstationCost, GridConnectionCost, CollectionCost,
      FoundationCost (simplified), SitePreparationCost (simplified),
      ErectionCost (simplified), DevelopmentCost, ManagementCost.

    Returns a dict of all KPIs.
    """
    project_size_MW     = n_turbines * turbine_rating_MW
    plant_capacity_MW   = project_size_MW
    interconnect_voltage_kV = 0.4398 * project_size_MW + 60.204  # LandBOSSE auto formula
    distance_to_interconnect_mi = max(0.0, 0.009375 * project_size_MW + 0.625)  # auto

    # --- turbine unit cost estimate (if not provided) ---
    # Rule of thumb: ~900 USD/kW for onshore IEA 3-4 MW class turbines
    if turbine_unit_cost_usd is None:
        turbine_unit_cost_usd = 900 * turbine_rating_MW * 1000  # USD per turbine

    turbine_cost_total = turbine_unit_cost_usd * n_turbines

    # ── 1. Substation (SubstationCost.py exact formula) ──────────────────────
    if project_size_MW > 15:
        substation_cost = (11652 * (interconnect_voltage_kV + project_size_MW)
                           + 11795 * project_size_MW**0.3549
                           + 1526800)
    elif project_size_MW > 10:
        substation_cost = 1_000_000
    else:
        substation_cost = 500_000

    # ── 2. Grid connection / transmission (GridConnectionCost.py) ────────────
    if project_size_MW > 15:
        if distance_to_interconnect_mi == 0:
            trans_dist_usd = 0.0
        else:
            # LandBOSSE uses new_switchyard=True by default for utility scale
            interconnect_adder = 18115 * interconnect_voltage_kV + 165944
            trans_dist_usd = (
                1176 * interconnect_voltage_kV
                + 218257
            ) * distance_to_interconnect_mi**(-0.1063) * distance_to_interconnect_mi \
                + interconnect_adder
    else:
        trans_dist_usd = 0.0   # distributed mode

    # ── 3. Collection system / array cables ──────────────────────────────────
    # Simplified from CollectionCost.py:
    #   cable cost ≈ (labour + material) per km of trench
    #   LandBOSSE uses ~$200 k/km for 33 kV MV cable (material + install)
    #   We use the user-supplied cable_total_km (sum of turbine-to-sub distances)
    # Plus homerun trench from substation to interconnection
    cable_cost_per_km = 200_000   # USD/km, typical MV array cable installed
    collection_cost   = cable_total_km * cable_cost_per_km

    # ── 4. Foundation (FoundationCost.py – curve fit for onshore 3-4 MW) ────
    # LandBOSSE empirical: ~$80/kW for foundations at 3-4 MW class
    foundation_cost = 80 * turbine_rating_MW * 1000 * n_turbines

    # ── 5. Site preparation / roads (SitePreparationCost.py) ────────────────
    # ~$10/kW for roads; fraction_new_roads=0.33, road_quality=0.6 baked in
    roads_cost = 10 * turbine_rating_MW * 1000 * n_turbines

    # ── 6. Erection (ErectionCost.py – simplified per turbine) ──────────────
    # LandBOSSE: ~$55k–$95k / turbine depending on hub height
    erection_cost_per_turbine = 55_000 + max(0, hub_height_m - 80) * 1_000
    erection_cost = erection_cost_per_turbine * n_turbines

    # ── 7. Development (DevelopmentCost.py exact formula) ────────────────────
    # project_size_megawatts < 200 branch:
    import math
    development_cost = (7188.5 * n_turbines
                        + round(3.4893 * math.log(n_turbines) - 7.3049, 0) * 16800
                        + 165675)

    # ── 8. Management (ManagementCost.py) ────────────────────────────────────
    # project_value_usd = sum of above (pre-management)
    bos_pre_mgmt = (substation_cost + trans_dist_usd + collection_cost
                    + foundation_cost + roads_cost + erection_cost + development_cost)
    project_value_usd = turbine_cost_total + bos_pre_mgmt

    construct_months = 9  # typical for this scale
    # management formula (< 200 MW branch):
    if project_size_MW < 200:
        pm = (53.333 * construct_months**2 - 3442 * construct_months + 209542
              if construct_months > 22 else
              (construct_months + 2) * 155_000)
    else:
        pm = 0.0
    insurance_cost       = 0.0056 * project_value_usd
    building_permits     = 0.02   * foundation_cost
    highway_permits      = 20_000 * 10  # 10 permits default
    performance_bond     = 0.01   * project_value_usd
    # markup (default percentages from LandBOSSE)
    markup_contingency   = 0.03
    markup_warranty      = 0.0002
    markup_tax           = 0.0
    markup_overhead      = 0.05
    markup_profit        = 0.05
    markup_total         = (markup_contingency + markup_warranty + markup_tax
                            + markup_overhead + markup_profit)
    markup_cost          = markup_total * project_value_usd
    management_cost      = pm + insurance_cost + building_permits + highway_permits \
                           + performance_bond + markup_cost

    # ── Totals ────────────────────────────────────────────────────────────────
    bos_cost = (substation_cost + trans_dist_usd + collection_cost
                + foundation_cost + roads_cost + erection_cost
                + development_cost + management_cost)

    capex_total    = turbine_cost_total + bos_cost       # USD
    capex_per_kw   = capex_total / (project_size_MW * 1000)

    # ── Economic KPIs ─────────────────────────────────────────────────────────
    aep_kwh_yr     = aep_gwh_yr * 1e6                    # kWh/yr
    cf             = aep_kwh_yr / (plant_capacity_MW * 1000 * 8760)

    # OpEx: land rent + O&M
    opex_yr        = (d_o_and_m * capex_total            # 1.2 % of CapEx
                      + d_rent * n_turbines)             # EUR/turbine·yr (site-specific)

    # LCOE  [USD/kWh]
    fcr            = (d_discount * (1 + d_discount)**d_life_time
                      / ((1 + d_discount)**d_life_time - 1))
    lcoe_usd_kwh   = (capex_total * fcr + opex_yr) / aep_kwh_yr
    lcoe_eur_mwh   = lcoe_usd_kwh * 1000 * 0.92  # ~0.92 USD→EUR rough parity

    # NPV (simple: constant revenue at Danish merchant average ~60 €/MWh)
    electricity_price_eur_mwh = 60.0
    annual_revenue = aep_kwh_yr / 1000 * electricity_price_eur_mwh  # EUR/yr
    capex_eur      = capex_total * 0.92
    opex_eur_yr    = opex_yr * 0.92
    net_cash_flow  = annual_revenue - opex_eur_yr
    # NPV = -CapEx + PV of net_cash_flows
    pv_factor = (1 - (1 + d_discount)**(-d_life_time)) / d_discount
    npv        = -capex_eur + net_cash_flow * pv_factor

    # Profitability Index  PI = PV(benefits) / CapEx
    pv_benefits = net_cash_flow * pv_factor
    pi          = pv_benefits / capex_eur

    # Simple Payback Period
    spp = capex_eur / net_cash_flow if net_cash_flow > 0 else float('inf')

    result = dict(
        n_turbines=n_turbines,
        turbine_rating_MW=turbine_rating_MW,
        project_size_MW=project_size_MW,
        aep_gwh_yr=aep_gwh_yr,
        capacity_factor_pct=cf * 100,
        # Cost breakdown (USD)
        turbine_cost_total_USD=turbine_cost_total,
        substation_cost_USD=substation_cost,
        grid_connection_USD=trans_dist_usd,
        collection_cable_USD=collection_cost,
        foundation_USD=foundation_cost,
        roads_USD=roads_cost,
        erection_USD=erection_cost,
        development_USD=development_cost,
        management_USD=management_cost,
        bos_cost_USD=bos_cost,
        capex_total_USD=capex_total,
        capex_per_kw_USD=capex_per_kw,
        opex_per_yr_USD=opex_yr,
        fcr=fcr,
        lcoe_USD_per_kWh=lcoe_usd_kwh,
        lcoe_EUR_per_MWh=lcoe_eur_mwh,
        npv_EUR=npv,
        pi=pi,
        spp_years=spp,
        cable_total_km=cable_total_km,
        distance_to_interconnect_mi=distance_to_interconnect_mi,
        interconnect_voltage_kV=interconnect_voltage_kV,
    )

    if verbose:
        w = 52
        print()
        print("=" * w)
        print(f"  LandBOSSE-equivalent BOS & LCoE Report")
        print("=" * w)
        print(f"  Project size          : {project_size_MW:.1f} MW  ({n_turbines} × {turbine_rating_MW} MW)")
        print(f"  AEP                   : {aep_gwh_yr:.2f}  GWh/yr")
        print(f"  Capacity Factor       : {cf*100:.1f} %")
        print(f"  Hub height            : {hub_height_m} m")
        print(f"  Rotor diameter        : {rotor_diameter_m} m")
        print(f"  Total cable run       : {cable_total_km:.2f} km  (turbine → sub)")
        print(f"  Distance to grid      : {distance_to_interconnect_mi:.1f} mi  (auto-estimated)")
        print(f"  Interconnect voltage  : {interconnect_voltage_kV:.0f} kV  (auto-estimated)")
        print()
        print("  ── Capital Cost Breakdown ──────────────────────────")
        print(f"  Turbine supply        : $ {turbine_cost_total/1e6:>8.2f} M")
        print(f"  Foundation            : $ {foundation_cost/1e6:>8.2f} M")
        print(f"  Site prep / roads     : $ {roads_cost/1e6:>8.2f} M")
        print(f"  Erection              : $ {erection_cost/1e6:>8.2f} M")
        print(f"  Collection cables     : $ {collection_cost/1e6:>8.2f} M")
        print(f"  Substation            : $ {substation_cost/1e6:>8.2f} M")
        print(f"  Grid connection       : $ {trans_dist_usd/1e6:>8.2f} M")
        print(f"  Development           : $ {development_cost/1e6:>8.2f} M")
        print(f"  Management / markup   : $ {management_cost/1e6:>8.2f} M")
        print(f"  {'─'*40}")
        print(f"  BoS total             : $ {bos_cost/1e6:>8.2f} M")
        print(f"  CapEx total           : $ {capex_total/1e6:>8.2f} M")
        print(f"  CapEx / kW            : $ {capex_per_kw:>8.0f} /kW")
        print()
        print("  ── OpEx & Finance ──────────────────────────────────")
        print(f"  Annual OpEx           : $ {opex_yr/1e6:>8.3f} M/yr")
        print(f"  FCR                   :   {fcr*100:>7.2f} %")
        print(f"  Discount rate         :   {d_discount*100:>7.2f} %")
        print(f"  Project lifetime      :   {d_life_time} yr")
        print()
        print("  ── Economic KPIs ───────────────────────────────────")
        print(f"  LCOE                  : $ {lcoe_usd_kwh*1000:>7.2f} /MWh")
        print(f"  LCOE                  : € {lcoe_eur_mwh:>7.2f} /MWh")
        print(f"  NPV                   : € {npv/1e6:>8.2f} M")
        print(f"  Profitability Index   :   {pi:>7.3f}")
        print(f"  Simple Payback Period :   {spp:>7.1f} yr")
        print("=" * w)
        print()

    return result

# ---------------------------------------------------------------------------
# Parallel sweep (one process per (n_turbines, turbine_type) combo)
# ---------------------------------------------------------------------------
def _run_one(args):
    n, t_idx = args

    global HH, ROTOR_DIAMETER_M, MIN_TURBINE_SPACING, EXCLUSION_ZONES
    global ROAD_EXCL_NORMAL_M, ROAD_EXCL_TOFT_M, turbine_yaml_path

    turbine_yaml_path = os.path.join(tub_lib, turbines[t_idx] + ".yaml")
    with open(turbine_yaml_path, 'r') as f:
        tdata = yaml.safe_load(f)
    HH              = tdata['hub_height']
    ROTOR_DIAMETER_M = tdata['rotor_diameter']
    MIN_TURBINE_SPACING = 2 * ROTOR_DIAMETER_M
    ROTOR_RADIUS_M  = ROTOR_DIAMETER_M / 2.0
    EXCLUSION_ZONES = []
    for lat,lon in builds:
        EXCLUSION_ZONES.append((lon,lat,HH*4))
    EXCLUSION_ZONES.append((lon_k_town,lat_k_town,1000.0))
    for lat,lon in wts:
        EXCLUSION_ZONES.append((lon,lat,HH*4))
    ROAD_EXCL_NORMAL_M = ROTOR_RADIUS_M + 10.0
    ROAD_EXCL_TOFT_M   = ROTOR_RADIUS_M + 15.0

    flat_best, x0, pos, sub_pos, exact_aep, history, pso_time = \
        run_optimization(n_turbines=n, maxiter=MAX_ITER, turbine_type_name=turbines[t_idx])

    lp  = line_penalty(pos)
    bp  = boundary_penalty(pos)
    ep  = exclusion_penalty(pos)
    ip  = interturbine_penalty(pos)
    total_pen = lp + bp + ep + ip

    return {"n": n, "t_idx": t_idx, "x0": x0, "positions": pos,
            "substation": sub_pos, "aep": float(exact_aep),
            "total_pen": total_pen, "time": pso_time, "history": history,
            "hub_height": HH, "rotor_diameter": ROTOR_DIAMETER_M,
            "turbine_rating_MW": tdata.get('turbine_rating', tdata.get('turbine_rating_kw', 3300)/1000)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    if SAVE_PLOTS and os.path.exists("plots"):
        shutil.rmtree("plots")
    os.makedirs("plots",         exist_ok=True)
    os.makedirs("history_plots", exist_ok=True)
    os.makedirs("history_logs",  exist_ok=True)
    os.makedirs("optimizedLayout", exist_ok=True)

    # ── Initial site map ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(20,12))
    ax.add_patch(plt.Polygon(list(zip(boundary_x,boundary_y)),
                             closed=True, facecolor='green', edgecolor='blue',
                             alpha=0.5, label="Valid Area"))
    first_road = True
    for (x0,y0),(x1,y1) in road_segments:
        buf = road_buffer_polygon(x0,y0,x1,y1,ROAD_BUFFER_M)
        ax.add_patch(plt.Polygon(buf,closed=True,facecolor='blue',edgecolor='blue',
                                 alpha=0.4,label="Invalid Area - Road" if first_road else "_nolegend_"))
        ax.plot([x0,x1],[y0,y1],color='black',label="Road" if first_road else "_nolegend_")
        first_road = False
    first_build = True
    for lat,lon in builds:
        ax.plot(lon,lat,'x',color='darkgreen')
        ax.add_patch(plt.Circle((lon,lat),radius=HH*4,facecolor='red',edgecolor='red',
                                alpha=0.5,label="Invalid Area - Building" if first_build else "_nolegend_"))
        first_build = False
    ax.plot(lon_k_town,lat_k_town,'x',color='purple',label="Town")
    ax.add_patch(plt.Circle((lon_k_town,lat_k_town),radius=1000,facecolor='purple',
                            edgecolor='purple',alpha=0.5,label="Invalid Area - Town"))
    first_wt = True
    for lat,lon in wts:
        ax.plot(lon,lat,'>',color='navy',label="Wind Turbines" if first_wt else "_nolegend_")
        ax.add_patch(plt.Circle((lon,lat),radius=HH*4,facecolor='firebrick',edgecolor='firebrick',
                                alpha=0.5,label="Invalid Area - Wind Turbine" if first_wt else "_nolegend_"))
        first_wt = False
    first_fb = True
    for line in lines:
        for l in range(len(line)-1):
            ax.plot((fb[line[l]][1],fb[line[l+1]][1]),(fb[line[l]][0],fb[line[l+1]][0]),
                    color='yellow',label="Field Border" if first_fb else "_nolegend_")
            first_fb = False
    ax.set_aspect('equal'); ax.autoscale()
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_xlim(-500,4500); ax.set_ylim(-500,2500)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0,1.0)); plt.tight_layout()

    # ── Parallel optimisation ───────────────────────────────────────────────
    print("Running MULTIPROCESS parallel sweep …")
    all_results = {}
    times_dict  = {}
    start_time  = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_run_one,(n,t)): (n,t)
                   for n in range(N_MIN, N_MAX+1)
                   for t in range(len(turbines))}

        for fut in as_completed(futures):
            r   = fut.result()
            key = f"{r['n']}_{r['t_idx']}"
            all_results[key] = r

            t      = r['time']
            hrs    = int(t // 3600)
            mins   = int((t % 3600) // 60)
            secs   = t % 60
            tstr   = (f"{hrs}h {mins}m {secs:.1f}s" if hrs else f"{mins}m {secs:.1f}s")
            tname  = turbines[r['t_idx']]
            ncurr  = r['n']
            sx, sy = r['substation']

            cable_km = total_cable_km(r['positions'], sx, sy)
            print(f"  n={ncurr} | type={tname} | AEP={r['aep']/1e9:.3f} GWh "
                  f"| cable={cable_km:.1f} km | pen={r['total_pen']:.6f} | {tstr}")

            # Layout text file
            with open(f"optimizedLayout/{tname}_{ncurr}_layout.txt","w") as f:
                f.write(f"=== Best layout: {ncurr} turbines, type {tname} ===\n")
                f.write(f"Opt time : {tstr}\n")
                f.write(f"AEP      : {r['aep']/1e9:.3f} GWh/yr\n")
                f.write(f"Penalty  : {r['total_pen']:.6f}\n")
                f.write(f"Substation: x={sx:.2f} m, y={sy:.2f} m\n")
                f.write(f"Total cable run: {cable_km:.2f} km\n")
                for i,(x,y) in enumerate(r['positions']):
                    f.write(f"  Turbine {i}: x={x:.4f} m, y={y:.4f} m\n")

            # History plot
            hist = r['history']
            iters = [h['iter'] for h in hist]
            pens  = [h['penalty'] for h in hist]
            aeps  = [h['aep']/1e9 for h in hist]
            fig2, ax1 = plt.subplots(figsize=(10,6))
            ax1.set_yscale('log'); ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Penalty',color='tab:red')
            ax1.plot(iters,pens,color='tab:red'); ax1.tick_params(axis='y',labelcolor='tab:red')
            ax2 = ax1.twinx(); ax2.set_ylabel('AEP (GWh)',color='tab:blue')
            ax2.plot(iters,aeps,color='tab:blue'); ax2.tick_params(axis='y',labelcolor='tab:blue')
            plt.title(f"Opt History: {ncurr} turbines ({tname})")
            fig2.tight_layout()
            plt.savefig(f"history_plots/history_{tname}_{ncurr}.png"); plt.close(fig2)

            tname_key = tname
            if tname_key not in times_dict: times_dict[tname_key] = {}
            times_dict[tname_key][ncurr] = r['time']
            with open("times_taken.json","w") as f: json.dump(times_dict,f,indent=4)

    total_time = time.time() - start_time
    print(f"\n--- Done in {total_time:.1f} s ({total_time/60:.1f} min) ---")

    # ── Select best result ──────────────────────────────────────────────────
    PEN_THRESHOLD = 1e-3
    satisfying = {k:r for k,r in all_results.items() if r['total_pen'] < PEN_THRESHOLD}
    candidates = satisfying if satisfying else all_results
    if not satisfying:
        print(f"\nWARNING: no run met penalty < {PEN_THRESHOLD}. Using best available.")
    best = max(candidates.values(), key=lambda r: r['aep'])

    N_TURBINES   = best['n']
    BEST_T_IDX   = best['t_idx']
    positions    = best['positions']
    substation   = best['substation']
    aep_gwh      = best['aep'] / 1e9
    t_name       = turbines[BEST_T_IDX]
    cable_km     = total_cable_km(positions, substation[0], substation[1])

    # Reload turbine data for plotting
    turbine_yaml_path = os.path.join(tub_lib, turbines[BEST_T_IDX] + ".yaml")
    with open(turbine_yaml_path,'r') as f:
        td = yaml.safe_load(f)
    HH              = td['hub_height']
    ROTOR_DIAMETER_M = td['rotor_diameter']
    MIN_TURBINE_SPACING = 2 * ROTOR_DIAMETER_M
    turbine_rating_MW = best.get('turbine_rating_MW', td.get('turbine_rating', 3.3))

    print(f"\n=== Best layout: {N_TURBINES} turbines, type {t_name} ===")
    print(f"AEP      : {aep_gwh:.3f} GWh/yr")
    print(f"Penalty  : {best['total_pen']:.6f}")
    print(f"Substation: x={substation[0]:.1f} m, y={substation[1]:.1f} m")
    print(f"Total cable run: {cable_km:.2f} km")
    for i,(x,y) in enumerate(positions):
        print(f"  Turbine {i}: x={x:.2f} m, y={y:.2f} m")

    # ── Final layout plot ───────────────────────────────────────────────────
    title = (f"Best Layout: {N_TURBINES} turbines ({t_name}) | "
             f"AEP: {aep_gwh:.2f} GWh | Cable: {cable_km:.1f} km")
    out_png = f"optimized_layout_n{N_TURBINES}_t{BEST_T_IDX}.png"
    save_layout_plot(positions, substation, N_TURBINES, out_png, title)
    print(f"\nLayout plot saved → {out_png}")

    # ── LandBOSSE-equivalent LCoE calculation ──────────────────────────────
    lcoe_result = compute_lcoe(
        n_turbines        = N_TURBINES,
        turbine_rating_MW = turbine_rating_MW,
        hub_height_m      = HH,
        rotor_diameter_m  = ROTOR_DIAMETER_M,
        aep_gwh_yr        = aep_gwh,
        cable_total_km    = cable_km,
        distance_to_grid_km = cable_km,   # homerun ≈ cable run
        verbose           = True
    )

    # Save LCoE summary
    with open(f"lcoe_summary_n{N_TURBINES}_t{BEST_T_IDX}.txt","w") as f:
        f.write(f"LandBOSSE-equivalent LCoE Report\n")
        f.write(f"Project: {N_TURBINES} turbines × {turbine_rating_MW:.2f} MW = "
                f"{N_TURBINES*turbine_rating_MW:.1f} MW\n")
        f.write(f"Turbine type: {t_name}\n\n")
        for k,v in lcoe_result.items():
            f.write(f"  {k:<35} : {v:.4f}\n")
    print(f"LCoE summary saved → lcoe_summary_n{N_TURBINES}_t{BEST_T_IDX}.txt")