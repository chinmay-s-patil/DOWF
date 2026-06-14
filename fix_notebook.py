import json

notebook_path = "Assignments/Assignment6/Assignment 6 - Line Force, Zones, NTurbs.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "ProcessPoolExecutor" in source and "all_results" in source:
            new_source = """
N_MIN, N_MAX = 13, 17

def _run_one(n):
    x0, pos, aep_s = run_optimization(n_turbines=n, maxiter=2000)
    line_pen  = line_penalty(pos)
    bound_pen = boundary_penalty(pos)
    excl_pen  = exclusion_penalty(pos)
    inter_pen = interturbine_penalty(pos)
    total_pen = line_pen + bound_pen + excl_pen + inter_pen
    return {"n": n, "x0": x0, "positions": pos, "aep": aep_s, "total_pen": total_pen}

print("Running sequential sweep...")
all_results = {}
for n in range(N_MIN, N_MAX + 1):
    r = _run_one(n)
    all_results[r["n"]] = r
    print(f"  n={r['n']} | AEP={r['aep']/AEP_WEIGHT:.3f} GWh | penalty={r['total_pen']:.6f}")
    print()
"""
            # Split lines and append newlines properly for Jupyter format
            lines = new_source.strip().split('\n')
            cell["source"] = [line + "\n" if i < len(lines)-1 else line for i, line in enumerate(lines)]
            print("Successfully updated cell in the notebook.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
