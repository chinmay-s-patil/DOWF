import os
import time
import numpy as np

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

YAML_PATH = "../Assignments/Assignment3/gch.yaml"

REPEATS = 5
N_DIRECTIONS = 360
N_SPEEDS     = 50

wd_grid, ws_grid = np.meshgrid(np.linspace(0, 350, N_DIRECTIONS), np.linspace(6, 14, N_SPEEDS))
wind_directions = wd_grid.flatten().tolist()
wind_speeds     = ws_grid.flatten().tolist()
turbulence_intensities = [0.06] * (N_DIRECTIONS * N_SPEEDS)

def run_cpu(repeats: int) -> float:
    # pyrefly: ignore [missing-import]
    import floris
    fmodel = floris.FlorisModel(YAML_PATH)
    fmodel.set(wind_directions=wind_directions, wind_speeds=wind_speeds, turbulence_intensities=turbulence_intensities)
    fmodel.run()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fmodel.run()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats

def run_gpu(repeats: int) -> float:
    # pyrefly: ignore [missing-import]
    import floris_cupy
    import cupy as cp
    fmodel = floris_cupy.FlorisModel(YAML_PATH)
    fmodel.set(wind_directions=wind_directions, wind_speeds=wind_speeds, turbulence_intensities=turbulence_intensities)
    fmodel.run()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fmodel.run()
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats

print(f"Directions: {N_DIRECTIONS}, Speeds: {N_SPEEDS}, Total states: {N_DIRECTIONS * N_SPEEDS}")
gpu_time = run_gpu(REPEATS)
cpu_time = run_cpu(REPEATS)
print(f"CPU avg: {cpu_time:.4f} s")
print(f"GPU avg: {gpu_time:.4f} s")
if gpu_time < cpu_time:
    print(f"GPU is {cpu_time/gpu_time:.2f}x faster")
else:
    print(f"CPU is {gpu_time/cpu_time:.2f}x faster")
