import numpy as np
# pyrefly: ignore [missing-import]
import floris_cupy.wind_data as wind_data

wd = np.array([0, 10, 20])
ws = np.array([6, 8, 10])
ti = 0.06

wr = wind_data.WindRose(wind_directions=wd, wind_speeds=ws, ti_table=ti)
print("WindRose initialized successfully!")

# Test plotting which triggered the pandas issue
try:
    wr.plot()
    print("WindRose plotting successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Plotting failed: {e}")
