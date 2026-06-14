import numpy as np_cpu
import cupy as np
from floris import FlorisModel
from floris.wind_data import WindRose

# Simple setup mimicking Assignment 6 wind data
WD_BINS = np_cpu.arange(0, 360, 10)
WS_BINS = np_cpu.arange(6, 15, 1)
freq_table = np_cpu.random.rand(len(WD_BINS), len(WS_BINS))
ti_table = 0.06

# Pass CPU numpy arrays, expecting WindRose to convert them to CuPy arrays
wind_rose = WindRose(
    wind_directions=WD_BINS,
    wind_speeds=WS_BINS,
    ti_table=ti_table,
    freq_table=freq_table
)

fmodel = FlorisModel("modules/floris/floris/core/core.json")
fmodel.set(wind_data=wind_rose)
fmodel.run()

aep = fmodel.get_farm_AEP()
print(f"Computed AEP successfully: {aep}")
