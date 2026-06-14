import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rated_power = 3370
rated_wind_speed = 9.8
cut_in_wind_speed = 3
cut_out_wind_speed = 25
rotor_dia = 130
hub_height = 120

target_wind_dir = 240
weibull_c = 7.5
weibull_k = 2.72
turbine_working_hours = 1270

df = pd.read_csv("/home/lavender/Studies/Design of Wind Farms/Assignment1/IEA_Reference_3.4MW_130.csv")



