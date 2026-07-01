#!/usr/bin/env python
"""Plotting utilities for wind farm layout optimization."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from config import ROAD_BUFFER_M, PLOT_XLIM, PLOT_YLIM
from kml_parser import road_buffer_polygon


def _draw_site_base(ax, site_geom, hh, min_spacing):
    """Draw the site boundary, roads, buildings, town, existing WTs, field borders."""
    # Boundary
    ax.add_patch(plt.Polygon(
        list(zip(site_geom.boundary_x, site_geom.boundary_y)),
        closed=True, facecolor='green', edgecolor='blue', alpha=0.5,
        label="Valid Area"
    ))

    # Roads
    first_road = True
    for (x0, y0), (x1, y1) in site_geom.road_segments:
        buf = road_buffer_polygon(x0, y0, x1, y1, ROAD_BUFFER_M)
        ax.add_patch(plt.Polygon(
            buf, closed=True, facecolor='blue', edgecolor='blue', alpha=0.4,
            label="Invalid Area - Road" if first_road else "_nolegend_"
        ))
        ax.plot([x0, x1], [y0, y1], color='black',
                label="Road" if first_road else "_nolegend_")
        first_road = False

    # Buildings
    first_build = True
    build_radius = hh * 4
    for lat, lon in site_geom.builds:
        ax.plot(lon, lat, 'x', color='darkgreen')
        ax.add_patch(Circle(
            (lon, lat), radius=build_radius, facecolor='red', edgecolor='red',
            alpha=0.5, label="Invalid Area - Building" if first_build else "_nolegend_"
        ))
        first_build = False

    # Town
    ax.plot(site_geom.town_pos[1], site_geom.town_pos[0], 'x', color='purple', label="Town")
    ax.add_patch(Circle(
        (site_geom.town_pos[1], site_geom.town_pos[0]), radius=1000,
        facecolor='purple', edgecolor='purple', alpha=0.5,
        label="Invalid Area - Town"
    ))

    # Existing WTs
    first_wt = True
    wt_radius = hh * 4
    for lat, lon in site_geom.wts:
        ax.plot(lon, lat, '>', color='navy',
                label="Wind Turbines" if first_wt else "_nolegend_")
        ax.add_patch(Circle(
            (lon, lat), radius=wt_radius, facecolor='firebrick', edgecolor='firebrick',
            alpha=0.5, label="Invalid Area - Wind Turbine" if first_wt else "_nolegend_"
        ))
        first_wt = False

    # Field borders
    first_line = True
    for (x1, y1), (x2, y2) in site_geom.line_segments:
        ax.plot([x1, x2], [y1, y2], color='yellow',
                label="Field Border" if first_line else "_nolegend_")
        first_line = False


def save_layout_plot(positions, substation_pos, n_turbines, site_geom, hh, min_spacing,
                     filename, title="", rotor_diameter=None, substation_excl_radius=None):
    """Save a layout plot with all site features, turbines, substation, and cable lines."""
    fig, ax = plt.subplots(figsize=(20, 12))

    _draw_site_base(ax, site_geom, hh, min_spacing)

    # Substation exclusion zone
    if rotor_diameter and substation_excl_radius:
        sx, sy = substation_pos
        ax.add_patch(Circle(
            (sy, sx), radius=substation_excl_radius,
            facecolor='none', edgecolor='gold', linestyle='--', linewidth=2,
            label=f"Substation Exclusion ({substation_excl_radius:.0f} m)"
        ))

    # Cable connections from substation to each turbine
    sx, sy = substation_pos
    for x, y in positions:
        ax.plot([sy, y], [sx, x], color='gold', alpha=0.4, linewidth=0.8, zorder=1)

    # Substation
    ax.plot(sy, sx, 's', color='gold', markersize=14,
            markeredgecolor='black', markeredgewidth=2,
            label="Substation (optimized)", zorder=5)

    # Turbines
    first_best = True
    for x, y in positions:
        ax.plot(x, y, '*', color='lime', markersize=14,
                label="Optimized turbine" if first_best else "_nolegend_", zorder=5)
        ax.add_patch(Circle(
            (x, y), radius=min_spacing, facecolor='none', edgecolor='magenta',
            linestyle='-', linewidth=1.5,
            label="Min Spacing" if first_best else "_nolegend_"
        ))
        first_best = False

    if title:
        ax.set_title(title, fontsize=16)

    ax.set_aspect('equal')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_final_layout_plot(positions, start_positions, substation_pos, start_substation_pos,
                           site_geom, hh, min_spacing, filename, title="",
                           rotor_diameter=None, substation_excl_radius=None):
    """Save the final best layout plot with start positions and all annotations."""
    fig, ax = plt.subplots(figsize=(20, 12))

    _draw_site_base(ax, site_geom, hh, min_spacing)

    # Substation exclusion zone
    if rotor_diameter and substation_excl_radius:
        sx, sy = substation_pos
        ax.add_patch(Circle(
            (sy, sx), radius=substation_excl_radius,
            facecolor='none', edgecolor='gold', linestyle='--', linewidth=2,
            label=f"Substation Exclusion ({substation_excl_radius:.0f} m)"
        ))

    # Cable connections (optimized)
    sx, sy = substation_pos
    for x, y in positions:
        ax.plot([sy, y], [sx, x], color='gold', alpha=0.5, linewidth=1.0, zorder=1)

    # Start positions (turbines + substation)
    first_start = True
    for x, y in start_positions:
        ax.plot(x, y, 's', color='orange', markersize=10,
                label="Start position" if first_start else "_nolegend_", zorder=3)
        first_start = False
    ssx, ssy = start_substation_pos
    ax.plot(ssy, ssx, 's', color='orange', markersize=10,
            label="Start substation" if first_start else "_nolegend_", zorder=3)

    # Optimized substation
    ax.plot(sy, sx, 's', color='gold', markersize=14,
            markeredgecolor='black', markeredgewidth=2,
            label="Substation (optimized)", zorder=5)

    # Best positions (turbines)
    first_best = True
    for x, y in positions:
        ax.plot(x, y, '*', color='lime', markersize=14,
                label="Optimized turbine" if first_best else "_nolegend_", zorder=5)
        ax.add_patch(Circle(
            (x, y), radius=min_spacing, facecolor='none', edgecolor='magenta',
            linestyle='-', linewidth=1.5,
            label="Min Spacing" if first_best else "_nolegend_"
        ))
        first_best = False

    ax.set_aspect('equal')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_YLIM)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title, fontsize=16)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_history_plot(iters, pens, aeps, cable_dists, filename, title=""):
    """Save optimization history plot with 3 y-axes."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Penalty', color='tab:red')
    ax1.set_yscale('log')
    ax1.plot(iters, pens, color='tab:red', marker='o', label='Penalty')
    ax1.axhline(1e-4, color='tab:red', linestyle='--', alpha=0.5, label='Threshold (1e-4)')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('AEP (GWh)', color='tab:blue')
    ax2.plot(iters, aeps, color='tab:blue', marker='x', label='AEP')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Total Cable (km)', color='tab:green')
    ax3.plot(iters, [c/1000 for c in cable_dists], color='tab:green', marker='^', label='Cable')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    if title:
        plt.title(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)