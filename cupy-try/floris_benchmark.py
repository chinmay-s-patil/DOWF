"""
FLORIS vs FLORIS-CuPy Performance Benchmark
--------------------------------------------
Compares CPU (floris) vs GPU (floris_cupy) wake simulation time
using the Assignment 3 wind farm setup:
  - 3-turbine row, IEA 3.4 MW turbines
  - Gauss Curl Hybrid (GCH) model
  - Multiple wind directions & speeds for a heavier workload
"""

import os
import time
import numpy as np

# ── AMD ROCm env vars (same as tryrand.ipynb) ────────────────────────────────
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

# ── paths ─────────────────────────────────────────────────────────────────────
YAML_PATH = "../Assignments/Assignment3/gch.yaml"

# ── benchmark parameters ──────────────────────────────────────────────────────
REPEATS = 20          # how many timed runs to average
N_DIRECTIONS = 36    # wind directions swept (0-360°, step 10°)  → heavier load
N_SPEEDS     = 36     # wind speeds swept (6–14 m/s)

wind_directions = np.linspace(0, 350, N_DIRECTIONS).tolist()
wind_speeds     = np.linspace(6, 14, N_SPEEDS).tolist()
turbulence_intensities = [0.06] * N_DIRECTIONS  # one TI per direction


# ─────────────────────────────────────────────────────────────────────────────
# CPU benchmark  (standard floris)
# ─────────────────────────────────────────────────────────────────────────────
def run_cpu(repeats: int) -> float:
    # pyrefly: ignore [missing-import]
    import floris

    fmodel = floris.FlorisModel(YAML_PATH)
    fmodel.set(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=turbulence_intensities,
    )

    # warm-up
    fmodel.run()

    t0 = time.perf_counter()
    for _ in range(repeats):
        fmodel.run()
    t1 = time.perf_counter()

    powers = fmodel.get_farm_power()
    return (t1 - t0) / repeats, powers


# ─────────────────────────────────────────────────────────────────────────────
# GPU benchmark  (floris_cupy)
# ─────────────────────────────────────────────────────────────────────────────
def run_gpu(repeats: int) -> float:
    # pyrefly: ignore [missing-import]
    import floris_cupy
    import cupy as cp

    fmodel = floris_cupy.FlorisModel(YAML_PATH)
    fmodel.set(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=turbulence_intensities,
    )

    # warm-up
    fmodel.run()
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        fmodel.run()
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    powers = fmodel.get_farm_power()
    return (t1 - t0) / repeats, powers


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  FLORIS  vs  FLORIS-CuPy  –  Assignment 3 benchmark")
    print("=" * 55)
    print(f"  Wind directions : {N_DIRECTIONS}  ({wind_directions[0]}° → {wind_directions[-1]}°)")
    print(f"  Wind speeds     : {N_SPEEDS}  ({wind_speeds[0]:.1f} → {wind_speeds[-1]:.1f} m/s)")
    print(f"  Timed repeats   : {REPEATS}")
    print("-" * 55)

    # --- GPU ---
    print("\n[GPU] Running floris_cupy …", flush=True)
    try:
        gpu_time, gpu_powers = run_gpu(REPEATS)
        print(f"  avg time : {gpu_time:.4f} s")
        print(f"  total farm power (first condition): {gpu_powers[0].sum() / 1e6:.3f} MW")
    except Exception as e:
        print(f"  FAILED: {e}")
        gpu_time = None

    # --- CPU ---
    print("\n[CPU] Running floris …", flush=True)
    try:
        cpu_time, cpu_powers = run_cpu(REPEATS)
        print(f"  avg time : {cpu_time:.4f} s")
        print(f"  total farm power (first condition): {cpu_powers[0].sum() / 1e6:.3f} MW")
    except Exception as e:
        print(f"  FAILED: {e}")
        cpu_time = None

    # --- GPU ---
    print("\n[GPU] Running floris_cupy …", flush=True)
    try:
        gpu_time, gpu_powers = run_gpu(REPEATS)
        print(f"  avg time : {gpu_time:.4f} s")
        print(f"  total farm power (first condition): {gpu_powers[0].sum() / 1e6:.3f} MW")
    except Exception as e:
        print(f"  FAILED: {e}")
        gpu_time = None

    # --- Summary ---
    print("\n" + "=" * 55)
    if cpu_time and gpu_time:
        speedup = cpu_time / gpu_time
        faster  = "GPU" if speedup > 1 else "CPU"
        ratio   = speedup if speedup > 1 else 1 / speedup
        print(f"  CPU time  : {cpu_time:.4f} s")
        print(f"  GPU time  : {gpu_time:.4f} s")
        print(f"  → {faster} is {ratio:.2f}x faster")
    elif cpu_time:
        print(f"  CPU time  : {cpu_time:.4f} s  (GPU unavailable)")
    elif gpu_time:
        print(f"  GPU time  : {gpu_time:.4f} s  (CPU unavailable)")
    else:
        print("  Both runs failed – check your floris / floris_cupy install.")
    print("=" * 55)




    # --- CPU ---
    # print("\n[CPU] Running floris …", flush=True)
    # cpu_time, cpu_powers = run_cpu(REPEATS)
    # print(f"  avg time : {cpu_time:.4f} s")
    # print(f"  total farm power (first condition): {cpu_powers[:].sum() / 1e6:.3f} MW")

    # # --- GPU ---
    # print("\n[GPU] Running floris_cupy …", flush=True)
    # gpu_time, gpu_powers = run_gpu(REPEATS)
    # print(f"  avg time : {gpu_time:.4f} s")
    # print(f"  total farm power (first condition): {gpu_powers[:].sum() / 1e6:.3f} MW")