#!/usr/bin/env python3
"""
Standalone LCoE Calculator
--------------------------
Reads an optimized layout .txt file, extracts turbine (x, y) positions,
and computes:
  • LCoE  (USD/kWh and EUR/MWh)
  • Total Life-Cycle Cost  (TLCC)
  • Net Present Value  (NPV)
  • Profitability Index  (PI)
  • Internal Rate of Return  (IRR)
  • Cost-breakdown pie chart

Formulas follow TUM "Design of Wind Farms" Lecture 5 (Short et al., 1995).
Cost model mirrors the existing LandBOSSE-equivalent parametric approach.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ── Project / Site constants (Denmark, from Assignment 6) ──────────────
USD_TO_EUR          = 0.92
SPOT_PRICE_EUR_MWH  = 60.0                          # Denmark avg spot price
SPOT_PRICE_USD_MWH  = SPOT_PRICE_EUR_MWH / USD_TO_EUR

DISCOUNT_RATE       = 0.036         # real discount rate  (d_discount / 100)
LIFETIME_YR         = 20            # d_life_time
CONSTRUCTION_MONTHS = 12            # d_construction_time

TURBINE_RATING_MW   = 3.3           # BAR_BAU_IEA_3.3MW
HUB_HEIGHT_M        = 120.0
ROTOR_DIAMETER_M    = 156.0

# ── Cost parameters (USD) ─────────────────────────────────────────────
TURBINE_COST_PER_KW    = 900.0      # USD/kW  (900 k$/MW)
FOUNDATION_PER_TURB    = 264_000.0  # USD  (from summary: 4 224 000 / 16)
ROADS_PER_TURB         = 33_000.0   # USD  (from summary: 528 000 / 16)
ERECTION_PER_TURB      = 95_000.0   # USD  (from summary: 1 520 000 / 16)
DEVELOPMENT_FIXED      = 314_291.0  # USD  (roughly fixed)

# Collection cable: scales with total cable km
CABLE_COST_PER_KM      = 200_000.0  # USD/km  (7 679 398 / 38.397 ≈ 200 k)

# Substation cost model  (from LandBOSSE parametric fit)
SUBSTATION_BASE        = 2_000_000.0
SUBSTATION_PER_MW      = 22_000.0

# Grid connection: proportional to project size
GRID_CONN_PER_MW       = 38_400.0   # USD/MW  (2 027 282 / 52.8)

# Management: ~60 % of hardware BoS (LandBOSSE includes insurance,
# bonding, project mgmt, engineering, permitting, and markups)
MGMT_FRACTION          = 0.604

# O&M cost per year as fraction of CAPEX  (d_o_and_m = 0.012)
OandM_FRACTION         = 0.012


# ═══════════════════════════════════════════════════════════════════════
# 1.  PARSER — read layout txt → list of (x, y)
# ═══════════════════════════════════════════════════════════════════════
def parse_layout(filepath: str) -> tuple[list[tuple[float, float]],
                                          dict]:
    """
    Parse an optimized-layout text file.

    Returns
    -------
    positions : list of (x_m, y_m)  turbine coordinates
    meta      : dict with keys 'aep_gwh', 'n_turbines',
                'substation_xy', 'cable_km' (when available)
    """
    positions = []
    meta = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # AEP line
            m = re.match(r"AEP\s*:\s*([\d.]+)\s*GWh", line)
            if m:
                meta["aep_gwh"] = float(m.group(1))

            # Substation
            m = re.match(r"Substation.*x\s*=\s*([\d.]+)\s*m.*y\s*=\s*([\d.]+)\s*m", line)
            if m:
                meta["substation_xy"] = (float(m.group(1)), float(m.group(2)))

            # Total cable run
            m = re.match(r"Total cable run.*?([\d.]+)\s*km", line)
            if m:
                meta["cable_km"] = float(m.group(1))

            # Turbine line
            m = re.match(r"\s*Turbine\s+\d+:\s*x\s*=\s*([\d.]+)\s*m.*y\s*=\s*([\d.]+)\s*m", line)
            if m:
                positions.append((float(m.group(1)), float(m.group(2))))

    meta["n_turbines"] = len(positions)
    return positions, meta


# ═══════════════════════════════════════════════════════════════════════
# 2.  COST + FINANCIAL MODEL
# ═══════════════════════════════════════════════════════════════════════
def compute_lcoe(positions: list[tuple[float, float]],
                 aep_gwh: float,
                 cable_km: float | None = None,
                 substation_xy: tuple[float, float] | None = None,
                 show_plot: bool = True,
                 save_plot: str | None = None) -> float:
    """
    Compute LCoE and related financial indicators.

    Parameters
    ----------
    positions      : list of (x, y) turbine coordinates in metres
    aep_gwh        : Annual Energy Production in GWh/yr
    cable_km       : total cable run in km (estimated if None)
    substation_xy  : (x, y) of substation in metres (estimated if None)
    show_plot      : whether to plt.show() the pie chart
    save_plot      : filepath to save the pie chart (None = don't save)

    Returns
    -------
    lcoe_usd_kwh : float
    """
    n  = len(positions)
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])

    project_MW  = n * TURBINE_RATING_MW
    project_kW  = project_MW * 1e3
    aep_kwh     = aep_gwh * 1e6          # GWh → kWh
    aep_mwh     = aep_gwh * 1e3          # GWh → MWh

    cap_factor  = aep_gwh * 1e3 / (project_MW * 8760) * 100   # %

    # ── Estimate cable length if not provided ──────────────────────
    if cable_km is None:
        if substation_xy is not None:
            sx, sy = substation_xy
        else:
            sx, sy = xs.mean(), ys.min()           # rough default
        dists = np.sqrt((xs - sx)**2 + (ys - sy)**2)
        cable_km = dists.sum() / 1000.0 * 1.15     # 15 % overhead for MST

    # ── Individual cost items (USD) ────────────────────────────────
    turbine_cost   = project_kW * TURBINE_COST_PER_KW
    foundation     = n * FOUNDATION_PER_TURB
    roads          = n * ROADS_PER_TURB
    erection       = n * ERECTION_PER_TURB
    development    = DEVELOPMENT_FIXED
    cable_cost     = cable_km * CABLE_COST_PER_KM
    substation     = SUBSTATION_BASE + project_MW * SUBSTATION_PER_MW
    grid_conn      = project_MW * GRID_CONN_PER_MW

    bos_hardware   = (foundation + roads + erection + development
                      + cable_cost + substation + grid_conn)
    management     = MGMT_FRACTION * bos_hardware

    bos_total      = bos_hardware + management
    capex          = turbine_cost + bos_total            # total initial investment
    capex_per_kw   = capex / project_kW

    opex_per_yr    = OandM_FRACTION * capex              # annual O&M

    # ── Financial metrics ──────────────────────────────────────────
    d = DISCOUNT_RATE
    N = LIFETIME_YR

    # Fixed Charge Rate  (inverse of annuity)
    fcr     = d * (1 + d)**N / ((1 + d)**N - 1)
    annuity = 1.0 / fcr                                  # A = ((1+d)^N - 1) / (d*(1+d)^N)

    # --- LCoE (Lecture 5, slide 22) ---
    # LCoE = I / (AEP · A) + O&M / AEP
    lcoe_usd_kwh = (capex * fcr + opex_per_yr) / aep_kwh
    lcoe_eur_mwh = lcoe_usd_kwh * 1e3 * USD_TO_EUR

    # --- TLCC  (Lecture 5, slide 18) ---
    # TLCC = Σ_{n=0}^{N} C_n / (1+d)^n
    # C_0 = CAPEX,  C_{1..N} = O&M
    tlcc = capex + opex_per_yr * annuity

    # --- Revenue & Cash Flows ---
    revenue_per_yr_eur = aep_mwh * SPOT_PRICE_EUR_MWH
    opex_per_yr_eur    = opex_per_yr * USD_TO_EUR
    capex_eur          = capex * USD_TO_EUR

    # --- NPV  (Lecture 5, slide 18) ---
    # NPV = Σ_{n=1}^{N} (Revenue - O&M) / (1+d)^n  −  CAPEX
    net_annual_eur = revenue_per_yr_eur - opex_per_yr_eur
    npv_eur        = net_annual_eur * annuity - capex_eur

    # --- PI  (Lecture 5, slide 26) ---
    # PI = 1 + NPV / I
    pi = 1.0 + npv_eur / capex_eur

    # --- Simple Payback Period ---
    spp_years = capex_eur / net_annual_eur

    # --- IRR  (Lecture 5, slide 24) ---
    # Find d* such that NPV(d*) = 0
    #   NPV(d) = Σ_{n=1}^{N} net_annual / (1+d)^n  −  CAPEX  = 0
    def npv_func(r):
        if abs(r) < 1e-12:
            return net_annual_eur * N - capex_eur
        ann = ((1 + r)**N - 1) / (r * (1 + r)**N)
        return net_annual_eur * ann - capex_eur

    try:
        irr = brentq(npv_func, -0.5, 10.0) * 100   # as %
    except ValueError:
        irr = float("nan")

    # ── Print summary ──────────────────────────────────────────────
    print("=" * 62)
    print("  LCoE & Financial Summary")
    print("=" * 62)
    print(f"  Turbines          : {n}")
    print(f"  Rating            : {TURBINE_RATING_MW:.1f} MW  →  {project_MW:.1f} MW total")
    print(f"  AEP               : {aep_gwh:.3f} GWh/yr")
    print(f"  Capacity Factor   : {cap_factor:.2f} %")
    print(f"  Cable Run         : {cable_km:.2f} km")
    print("-" * 62)
    print(f"  Turbine Cost      : ${turbine_cost:>14,.0f}")
    print(f"  Foundation        : ${foundation:>14,.0f}")
    print(f"  Roads             : ${roads:>14,.0f}")
    print(f"  Erection          : ${erection:>14,.0f}")
    print(f"  Development       : ${development:>14,.0f}")
    print(f"  Collection Cable  : ${cable_cost:>14,.0f}")
    print(f"  Substation        : ${substation:>14,.0f}")
    print(f"  Grid Connection   : ${grid_conn:>14,.0f}")
    print(f"  Management        : ${management:>14,.0f}")
    print(f"  BoS Total         : ${bos_total:>14,.0f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  CAPEX Total       : ${capex:>14,.0f}   ({capex_per_kw:.0f} $/kW)")
    print(f"  OPEX / yr         : ${opex_per_yr:>14,.0f}")
    print("-" * 62)
    print(f"  FCR               : {fcr:.4f}")
    print(f"  LCoE              : {lcoe_usd_kwh:.4f} USD/kWh")
    print(f"  LCoE              : {lcoe_eur_mwh:.2f} EUR/MWh")
    print(f"  TLCC              : ${tlcc:>14,.0f}")
    print("-" * 62)
    print(f"  Spot Price        : {SPOT_PRICE_EUR_MWH:.0f} EUR/MWh")
    print(f"  Annual Revenue    : €{revenue_per_yr_eur:>14,.0f}")
    print(f"  NPV               : €{npv_eur:>14,.0f}")
    print(f"  PI                : {pi:.4f}")
    print(f"  SPP               : {spp_years:.2f} years")
    print(f"  IRR               : {irr:.2f} %")
    print("=" * 62)

    # ── Pie chart ──────────────────────────────────────────────────
    labels = [
        "Turbines",
        "Foundation",
        "Roads",
        "Erection",
        "Development",
        "Collection Cable",
        "Substation",
        "Grid Connection",
        "Management",
    ]
    values = [
        turbine_cost,
        foundation,
        roads,
        erection,
        development,
        cable_cost,
        substation,
        grid_conn,
        management,
    ]

    colors = [
        "#2196F3",   # blue – turbines (dominant)
        "#FF9800",   # orange
        "#795548",   # brown
        "#4CAF50",   # green
        "#9C27B0",   # purple
        "#F44336",   # red
        "#00BCD4",   # cyan
        "#FFEB3B",   # yellow
        "#607D8B",   # grey
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%\n(${pct / 100 * capex / 1e6:.1f}M)",
        colors=colors,
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(8)
    for t in texts:
        t.set_fontsize(9)

    ax.set_title(
        f"CAPEX Breakdown — {n} × {TURBINE_RATING_MW:.1f} MW  "
        f"| LCoE = {lcoe_eur_mwh:.1f} €/MWh  "
        f"| PI = {pi:.2f}",
        fontsize=13, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot, dpi=200, bbox_inches="tight")
        print(f"\n  Pie chart saved → {save_plot}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return lcoe_usd_kwh


# ═══════════════════════════════════════════════════════════════════════
# 3.  CLI entry-point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        path = "Assignments/Assignment6/LoCE Opti/optimizedLayout/BAR_BAU_IEA_3.3MW_16_layout.txt"
    else:
        path = sys.argv[1]

    positions, meta = parse_layout(path)

    if not positions:
        print(f"ERROR: no turbine positions found in {path}")
        sys.exit(1)

    print(f"Parsed {meta['n_turbines']} turbines from {path}")

    lcoe = compute_lcoe(
        positions,
        aep_gwh=meta.get("aep_gwh", 0.0),
        cable_km=meta.get("cable_km"),
        substation_xy=meta.get("substation_xy"),
        show_plot=True,
        save_plot="lcoe_pie_chart.png",
    )
