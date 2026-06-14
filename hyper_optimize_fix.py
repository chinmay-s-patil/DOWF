import re

with open("Assignments/Assignment6/Assignment 6 - Line Force, Zones, NTurbs.py", "r") as f:
    code = f.read()

# Remove _run_one from inside the main block.
# I'll just write a quick parser to extract it and place it before if __name__ == '__main__':

pattern = re.compile(r"    def _run_one\(n\):.*?    print\(\"Running MULTIPROCESS", re.DOTALL)
match = pattern.search(code)
if match:
    run_one_code = """
def _run_one(n):
    x0, pos, aep_s = run_optimization(n_turbines=n, maxiter=2000)
    line_pen  = line_penalty(pos)
    bound_pen = boundary_penalty(pos)
    excl_pen  = exclusion_penalty(pos)
    inter_pen = interturbine_penalty(pos)
    total_pen = line_pen + bound_pen + excl_pen + inter_pen
    return {"n": n, "x0": x0, "positions": pos, "aep": aep_s, "total_pen": total_pen}

"""
    code = code[:match.start()] + '    print("Running MULTIPROCESS' + code[match.end():]
    
    # insert before if __name__ == '__main__':
    main_idx = code.find("if __name__ == '__main__':")
    code = code[:main_idx] + run_one_code + code[main_idx:]

with open("Assignments/Assignment6/Assignment 6 - Line Force, Zones, NTurbs.py", "w") as f:
    f.write(code)
