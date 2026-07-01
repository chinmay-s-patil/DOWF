# Wind Farm Layout Optimization (Denmark Site)

Modularized Python codebase for optimizing wind farm layout using FLORIS, GPU-accelerated PSO, and wake steering.

## Files

| File                  | Description                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `config.py`         | All constants, paths, optimization parameters, penalty weights                                                           |
| `kml_parser.py`     | KML parsing, coordinate conversion (DMS/WGS84 to local meters), site geometry extraction                                 |
| `constraints.py`    | All penalty functions: boundary, exclusion, inter-turbine,**substation exclusion**, **substation proximity** |
| `wind_data.py`      | Wind rose creation from Weibull distributions                                                                            |
| `floris_wrapper.py` | FLORIS model setup, AEP evaluation, yaw optimization                                                                     |
| `plotting.py`       | All visualization functions (site map, layouts, history plots)                                                           |
| `objective.py`      | Batched objective function combining FLORIS AEP + GPU penalties                                                          |
| `turbine_data.py`   | Turbine YAML loading, capacity-aware count calculations                                                                  |
| `pso_optimizer.py`  | GPU-accelerated batched Particle Swarm Optimization                                                                      |
| `main.py`           | Main execution: parallel sweep across turbine types & counts                                                             |
| `run_single.py`     | Run single optimization (for testing/debugging)                                                                          |
| `plot_windrose.py`  | Standalone wind rose plotting                                                                                            |

## Key Changes from Original Monolithic Script

### 1. Substation Handling

- **Substation position** is automatically extracted from KML (looks for "Substation" placemark)
- If not found in KML, computed as the **centroid** of the site boundary polygon
- Substation is plotted as a gold square on all layout plots
- **Substation exclusion zone** (dashed gold circle) is shown on plots

### 2. Substation Constraints

- **Exclusion penalty**: Turbines must stay ≥ `1.5 × rotor_diameter` away from substation (safety)
  - Weight: `SUBSTATION_EXCL_PENALTY_WEIGHT = 1e7`
- **Proximity penalty** (positive constraint): Turbines should stay within `2 × rotor_diameter + 500m` of substation
  - This keeps the farm centered near the substation, minimizing cable costs
  - Weight: `SUBSTATION_PROXIMITY_WEIGHT = 1e3` (light touch, doesn't dominate AEP)

### 3. Unit Fix

- All position outputs now print in **meters (m)** instead of kilometers (km)
- Updated in `optimizedLayout/*.txt` files and console output

### 4. Output Files

Each optimization run saves:

- `optimizedLayout/{turbine}_{n}_layout.txt` — Full layout with substation info, turbine positions in **m**
- `optimizedLayout/{turbine}_{n}_yaw_angles.npy` — Optimal yaw angles (if wake steering used)
- `plots/{turbine}/{n}/iter_####.png` — Per-iteration layout plots
- `history_plots/history_{turbine}_{n}.png` — Convergence history
- `history_logs/{turbine}_{n}_*_hist.txt` — Raw penalty/AEP data

### 5. Parallel Execution

- Uses `ProcessPoolExecutor` with `spawn` start method
- `MAX_WORKERS` processes run concurrently
- Each worker loads its own FLORIS model (avoids CUDA context sharing issues)

## Usage

### Full parallel sweep (all turbine types & valid counts):

```bash
python main.py
```

### Single optimization (for testing):

```bash
python run_single.py
```

### Plot wind rose only:

```bash
python plot_windrose.py
```

## Requirements

- Python 3.9+
- `numpy`, `cupy`, `matplotlib`, `shapely`, `scipy`, `pyyaml`
- `floris_cupy` (FLORIS with CuPy backend)
- CUDA-capable GPU (for CuPy acceleration)

## Directory Structure

```
.
├── config.py
├── constraints.py
├── floris_wrapper.py
├── kml_parser.py
├── main.py
├── objective.py
├── plotting.py
├── pso_optimizer.py
├── run_single.py
├── turbine_data.py
├── wind_data.py
├── plot_windrose.py
├── inputs/
│   ├── Denmark Site - Chin.kml
│   └── gch.yaml
├── turbineData/
│   ├── IEA_3_4MW.yaml
│   ├── BAR_BAU_IEA_3.3MW.yaml
│   └── BAR_BAU_LSP_3.25MW.yaml
├── plots/              # Generated per-run layout plots
├── history_plots/      # Generated convergence plots
├── history_logs/       # Generated raw data
└── optimizedLayout/    # Generated layout text files
```
