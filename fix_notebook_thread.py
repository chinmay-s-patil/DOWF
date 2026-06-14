from concurrent.futures import ThreadPoolExecutor, as_completed
import cupy as np

N_MIN, N_MAX = 13, 17

def _run_one(n):
    x0, pos, aep_s = run_optimization(n_turbines=n, maxiter=2000)
    line_pen  = line_penalty(pos)
    bound_pen = boundary_penalty(pos)
    excl_pen  = exclusion_penalty(pos)
    inter_pen = interturbine_penalty(pos)
    total_pen = line_pen + bound_pen + excl_pen + inter_pen
    return {"n": n, "x0": x0, "positions": pos, "aep": aep_s, "total_pen": total_pen}

print("Running threaded parallel sweep...")
all_results = {}
with ThreadPoolExecutor(max_workers=5) as ex:
    futures = {ex.submit(_run_one, n): n for n in range(N_MIN, N_MAX + 1)}
    for fut in as_completed(futures):
        r = fut.result()
        all_results[r["n"]] = r
        print(f"  n={r['n']} | AEP={r['aep']/AEP_WEIGHT:.3f} GWh | penalty={r['total_pen']:.6f}")
        print()