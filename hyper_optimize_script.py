import re

with open("Assignments/Assignment6/Assignment 6 - Line Force, Zones, NTurbs.py", "r") as f:
    code = f.read()

# Remove all jupyter comments like # In[...]:
code = re.sub(r'# In\[.*?\]:\n', '', code)
# Remove all # %% or # coding: utf-8
code = re.sub(r'# coding: utf-8\n', '', code)

# We want to put the execution part in an if __name__ == '__main__': block.
# Where does the execution start? "X_MIN = min(longLB, longLT)" is fine.
# Actually, everything from imports to X_MIN is setup. The optimization functions are also setup.
# The execution starts at "from concurrent.futures import ThreadPoolExecutor" or "print('Running threaded parallel sweep...')"

# Let's replace the _make_objective and the executor loop with a hyper-optimized one.

objective_replacement = """
import multiprocessing as mp
import cupy as cp
from concurrent.futures import ProcessPoolExecutor, as_completed

def _make_objective(n_turbines):
    # Pre-copy static boundary / exclusion arrays to GPU inside the worker 
    # to avoid context sharing issues.
    
    # Lines
    seg_p1 = cp.array([(x1, y1) for ((x1, y1), (x2, y2)) in LINE_SEGMENTS])
    seg_p2 = cp.array([(x2, y2) for ((x1, y1), (x2, y2)) in LINE_SEGMENTS])
    dxdy = seg_p2 - seg_p1
    seg_len_sq = cp.sum(dxdy**2, axis=1)
    
    # Exclusion
    exc_arr = cp.array(EXCLUSION_ZONES)
    exc_xy = exc_arr[:, :2]
    exc_r = exc_arr[:, 2]
    
    # Floris Model
    fmodel_local = FlorisModel(r"/home/lavender/Studies/Design of Wind Farms/Assignments/Assignment6/gch.yaml")
    fmodel_local.set(wind_data=wind_rose, wind_shear=d_ws)
    
    def _objective(flat_xy):
        positions = [(flat_xy[2 * i], flat_xy[2 * i + 1]) for i in range(n_turbines)]
        x_cpu = [p[0] for p in positions]
        y_cpu = [p[1] for p in positions]
        
        # Floris run
        fmodel_local.set(layout_x=x_cpu, layout_y=y_cpu)
        fmodel_local.run()
        value = (fmodel_local.get_farm_AEP() / 1e9) * AEP_WEIGHT
        
        # Penalties on GPU
        xy = cp.array(flat_xy).reshape(n_turbines, 2)
        px = xy[:, 0:1]
        py = xy[:, 1:2]
        
        # Line penalty
        t = ((px - seg_p1[:, 0]) * dxdy[:, 0] + (py - seg_p1[:, 1]) * dxdy[:, 1]) / seg_len_sq
        t = cp.clip(t, 0.0, 1.0)
        cx = seg_p1[:, 0] + t * dxdy[:, 0]
        cy = seg_p1[:, 1] + t * dxdy[:, 1]
        dist_sq_line = (px - cx)**2 + (py - cy)**2
        line_pen = cp.sum(cp.min(dist_sq_line, axis=1))
        
        # Exclusion penalty
        diff_exc = xy[:, None, :] - exc_xy[None, :, :]
        dist_exc = cp.sqrt(cp.sum(diff_exc**2, axis=2))
        pen_exc = cp.maximum(0, exc_r[None, :] - dist_exc)
        exc_pen = cp.sum(pen_exc**2)
        
        # Interturbine penalty
        diff_inter = xy[:, None, :] - xy[None, :, :]
        dist_inter = cp.sqrt(cp.sum(diff_inter**2, axis=2))
        mask = cp.triu(cp.ones((n_turbines, n_turbines), dtype=bool), k=1)
        dist_inter = dist_inter[mask]
        pen_inter = cp.maximum(0, MIN_TURBINE_SPACING - dist_inter)
        inter_pen = cp.sum(pen_inter**2)
        
        # Boundary penalty (CPU Shapely)
        bound_pen = boundary_penalty(positions)
        
        penalty  = LINE_PENALTY_WEIGHT       * float(line_pen)
        penalty += BOUNDARY_PENALTY_WEIGHT   * bound_pen
        penalty += EXCLUSION_PENALTY_WEIGHT  * float(exc_pen)
        penalty += INTER_TURBINE_PENALTY_WEIGHT * float(inter_pen)
        
        return -value + penalty
    return _objective
"""

# Let's replace `def _make_objective(n_turbines):...` to the end of that block.
# We'll use regex.
pattern = re.compile(r'def _make_objective\(n_turbines\):.*?(?=def run_optimization)', re.DOTALL)
code = pattern.sub(objective_replacement + '\n', code)


# Now, wrap the execution in if __name__ == '__main__':
main_exec = """
if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # already set

    N_MIN, N_MAX = 13, 17

    def _run_one(n):
        x0, pos, aep_s = run_optimization(n_turbines=n, maxiter=2000)
        # Note: Penalties are recalculated for printing using original functions
        line_pen  = line_penalty(pos)
        bound_pen = boundary_penalty(pos)
        excl_pen  = exclusion_penalty(pos)
        inter_pen = interturbine_penalty(pos)
        total_pen = line_pen + bound_pen + excl_pen + inter_pen
        return {"n": n, "x0": x0, "positions": pos, "aep": aep_s, "total_pen": total_pen}

    print("Running MULTIPROCESS parallel sweep with Spawn...")
    all_results = {}
    
    # Use ProcessPoolExecutor for true parallel python processes
    with ProcessPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_run_one, n): n for n in range(N_MIN, N_MAX + 1)}
        for fut in as_completed(futures):
            r = fut.result()
            all_results[r["n"]] = r
            print(f"  n={r['n']} | AEP={r['aep']/AEP_WEIGHT:.3f} GWh | penalty={r['total_pen']:.6f}")
            print()

"""

# Let's replace from `N_MIN, N_MAX = 13, 17` down to the `best_overall` logic.
pattern2 = re.compile(r'from concurrent.futures import ThreadPoolExecutor.*?(?=PEN_THRESHOLD)', re.DOTALL)
code = pattern2.sub(main_exec + '\n    PEN_THRESHOLD', code)

# Fix indentation of the remaining code (from PEN_THRESHOLD down to plt.show())
pattern3 = re.compile(r'(    PEN_THRESHOLD.*)', re.DOTALL)
match = pattern3.search(code)
if match:
    remaining_code = match.group(1)
    indented_remaining = ""
    for line in remaining_code.split('\n'):
        if line.strip() == "":
            indented_remaining += "\n"
        else:
            indented_remaining += "    " + line + "\n"
    code = code[:match.start()] + indented_remaining

# Also fix the `get_ipython()` calls if any exist from notebook plotting (inline)
code = re.sub(r'get_ipython\(\).*?\n', '', code)

with open("Assignments/Assignment6/Assignment 6 - Line Force, Zones, NTurbs.py", "w") as f:
    f.write(code)

