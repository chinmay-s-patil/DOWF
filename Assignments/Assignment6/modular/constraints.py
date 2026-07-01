#!/usr/bin/env python
"""Constraint penalty functions for wind farm layout optimization."""

import numpy as np
import cupy as cp
from shapely.geometry import Point

from config import (
    FIELD_BORDER_MAX_M, TOWN_RADIUS, BUILD_RADIUS_FACTOR, WT_RADIUS_FACTOR,
    SUBSTATION_EXCLUSION_FACTOR,
    LINE_PENALTY_WEIGHT, BOUNDARY_PENALTY_WEIGHT, EXCLUSION_PENALTY_WEIGHT,
    INTER_TURBINE_PENALTY_WEIGHT, SUBSTATION_EXCL_PENALTY_WEIGHT,
    SUBSTATION_BOUNDARY_PENALTY_WEIGHT
)


# ==
# Helper: point-to-segment distance
# ==

def _point_to_segment_dist_sq(px, py, x1, y1, x2, y2):
    """Squared distance from point (px, py) to segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return (px - x1) ** 2 + (py - y1) ** 2
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    cx, cy = x1 + t * dx, y1 + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2


def _dist_to_segs_batch(px, py, p1, dxdy, lensq):
    """
    px, py : (P, T, 1)
    p1     : (S, 2)
    returns: (P, T)  — min distance to any segment
    """
    if p1 is None:
        return cp.full((px.shape[0], px.shape[1]), cp.inf)
    px_diff = px - p1[:, 0]
    py_diff = py - p1[:, 1]
    t = (px_diff * dxdy[:, 0] + py_diff * dxdy[:, 1]) / lensq
    t = cp.clip(t, 0.0, 1.0)
    cx = p1[:, 0] + t * dxdy[:, 0]
    cy = p1[:, 1] + t * dxdy[:, 1]
    dists = cp.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    return cp.min(dists, axis=2)


# ==
# CPU Penalty Functions
# ==

def min_dist_to_any_line(x, y, line_segments):
    """Field border penalty: penalise if further than 50m from any field border segment."""
    best_field = np.inf
    for (x1, y1), (x2, y2) in line_segments:
        d = _point_to_segment_dist_sq(x, y, x1, y1, x2, y2)
        if d < best_field:
            best_field = d
    dist_field = np.sqrt(best_field) if best_field != np.inf else np.inf
    return max(0.0, dist_field - FIELD_BORDER_MAX_M) ** 2


def line_penalty(positions, line_segments):
    """Total line-constraint penalty for all positions."""
    return sum(min_dist_to_any_line(x, y, line_segments) for x, y in positions)


def boundary_penalty(positions, boundary_polygon):
    """Penalty for positions outside the site boundary."""
    total = 0.0
    for x, y in positions:
        p = Point(x, y)
        if not boundary_polygon.contains(p):
            total += boundary_polygon.exterior.distance(p) ** 2
    return total




def substation_boundary_penalty(substation_pos, boundary_polygon):
    """Dedicated heavy penalty for substation leaving the site boundary."""
    sx, sy = substation_pos
    p = Point(sx, sy)
    if not boundary_polygon.contains(p):
        return boundary_polygon.exterior.distance(p) ** 2
    return 0.0
def exclusion_penalty(positions, exclusion_zones, road_normal_segs, road_toft_segs,
                       road_excl_normal, road_excl_toft):
    """Penalty for positions too close to exclusion zones (buildings, town, roads)."""
    total = 0.0
    for tx, ty in positions:
        # Point exclusions (buildings, town, existing WTs)
        for cx, cy, r in exclusion_zones:
            dist = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
            if dist < r:
                total += (r - dist) ** 2

        # Road normal exclusion
        for (x1, y1), (x2, y2) in road_normal_segs:
            d = np.sqrt(_point_to_segment_dist_sq(tx, ty, x1, y1, x2, y2))
            if d < road_excl_normal:
                total += (road_excl_normal - d) ** 2

        # Road Toft exclusion
        for (x1, y1), (x2, y2) in road_toft_segs:
            d = np.sqrt(_point_to_segment_dist_sq(tx, ty, x1, y1, x2, y2))
            if d < road_excl_toft:
                total += (road_excl_toft - d) ** 2
    return total


def interturbine_penalty(positions, min_spacing):
    """Penalty for turbines too close to each other."""
    total = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < min_spacing:
                total += (min_spacing - dist) ** 2
    return total


def total_penalty(turbine_positions, substation_pos, site_geom, hh, rotor_diameter, min_spacing,
                  road_excl_normal=None, road_excl_toft=None):
    """
    Compute total penalty for a layout (CPU version).

    turbine_positions: list of (x, y) for turbines only
    substation_pos: (x, y) for substation
    """
    if road_excl_normal is None:
        road_excl_normal = rotor_diameter / 2.0 + 10.0
    if road_excl_toft is None:
        road_excl_toft = rotor_diameter / 2.0 + 15.0

    # Build exclusion zones dynamically based on current turbine type
    exclusion_zones = []
    build_radius = hh * BUILD_RADIUS_FACTOR
    wt_radius = hh * WT_RADIUS_FACTOR
    for lat, lon in site_geom.builds:
        exclusion_zones.append((lon, lat, build_radius))
    exclusion_zones.append((site_geom.town_pos[1], site_geom.town_pos[0], TOWN_RADIUS))
    for lat, lon in site_geom.wts:
        exclusion_zones.append((lon, lat, wt_radius))

    substation_excl_radius = rotor_diameter * SUBSTATION_EXCLUSION_FACTOR

    # All positions including substation for boundary/line/exclusion checks
    all_positions = list(turbine_positions) + [substation_pos]

    lp = line_penalty(all_positions, site_geom.line_segments)
    bp = boundary_penalty(all_positions, site_geom.boundary_polygon)
    ep = exclusion_penalty(all_positions, exclusion_zones,
                            site_geom.road_segments_normal,
                            site_geom.road_segments_toft,
                            road_excl_normal, road_excl_toft)

    # Inter-turbine spacing (turbines only, not substation)
    ip = interturbine_penalty(turbine_positions, min_spacing)

    # Substation-turbine exclusion: substation must stay away from turbines
    sep = 0.0
    sx, sy = substation_pos
    for tx, ty in turbine_positions:
        dist = np.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
        if dist < substation_excl_radius:
            sep += (substation_excl_radius - dist) ** 2

    # Dedicated hard-wall penalty for substation boundary
    sbp = substation_boundary_penalty(substation_pos, site_geom.boundary_polygon)

    total = (LINE_PENALTY_WEIGHT * lp +
             BOUNDARY_PENALTY_WEIGHT * bp +
             EXCLUSION_PENALTY_WEIGHT * ep +
             INTER_TURBINE_PENALTY_WEIGHT * ip +
             SUBSTATION_EXCL_PENALTY_WEIGHT * sep +
             SUBSTATION_BOUNDARY_PENALTY_WEIGHT * sbp)

    return total, {
        'line': lp, 'boundary': bp, 'exclusion': ep,
        'inter_turbine': ip, 'substation_excl': sep,
        'substation_boundary': sbp
    }


# ==
# GPU Batched Penalty Functions
# ==

def get_seg_data(segs):
    """Convert line segments to GPU arrays for batched distance computation."""
    if not segs:
        return None, None, None, None
    p1 = cp.array([(x1, y1) for ((x1, y1), (x2, y2)) in segs])
    p2 = cp.array([(x2, y2) for ((x1, y1), (x2, y2)) in segs])
    dxdy = p2 - p1
    lensq = cp.sum(dxdy ** 2, axis=1)
    lensq = cp.where(lensq == 0, 1e-10, lensq)
    return p1, p2, dxdy, lensq


class GPUPenaltyEngine:
    """Pre-computes GPU arrays for batched penalty evaluation."""

    def __init__(self, site_geom, hh, rotor_diameter, min_spacing,
                 road_excl_normal=None, road_excl_toft=None):
        self.site_geom = site_geom
        self.hh = hh
        self.rotor_diameter = rotor_diameter
        self.min_spacing = min_spacing

        # Road exclusion distances (computed dynamically if not provided)
        self.road_excl_normal = road_excl_normal if road_excl_normal is not None else rotor_diameter / 2.0 + 10.0
        self.road_excl_toft = road_excl_toft if road_excl_toft is not None else rotor_diameter / 2.0 + 15.0

        # Line segments (field borders)
        self.l_p1, self.l_p2, self.l_dxdy, self.l_lensq = get_seg_data(site_geom.line_segments)

        # Road segments
        self.rn_p1, self.rn_p2, self.rn_dxdy, self.rn_lensq = get_seg_data(site_geom.road_segments_normal)
        self.rt_p1, self.rt_p2, self.rt_dxdy, self.rt_lensq = get_seg_data(site_geom.road_segments_toft)

        # Exclusion zones (buildings, town, existing WTs)
        build_radius = hh * BUILD_RADIUS_FACTOR
        wt_radius = hh * WT_RADIUS_FACTOR
        exclusion_zones = []
        for lat, lon in site_geom.builds:
            exclusion_zones.append((lon, lat, build_radius))
        exclusion_zones.append((site_geom.town_pos[1], site_geom.town_pos[0], TOWN_RADIUS))
        for lat, lon in site_geom.wts:
            exclusion_zones.append((lon, lat, wt_radius))

        exc_arr = cp.array(exclusion_zones)
        self.exc_xy = exc_arr[:, :2]
        self.exc_r = exc_arr[:, 2]

        # Substation
        self.substation_excl_radius = rotor_diameter * SUBSTATION_EXCLUSION_FACTOR

        # Boundary polygon for CPU boundary penalty
        self.boundary_polygon = site_geom.boundary_polygon

    def evaluate(self, batch_pos, n_turbines):
        """
        batch_pos: (P, T+1, 2) cupy array where last entry is substation
        Returns: (P,) penalty scores, plus component breakdowns
        """
        n_particles = batch_pos.shape[0]

        # Split: turbines = all except last, substation = last
        turbine_pos = batch_pos[:, :-1, :]   # (P, T, 2)
        substation_pos = batch_pos[:, -1:, :]  # (P, 1, 2)

        # All positions for boundary/line/exclusion (turbines + substation)
        all_pos = batch_pos  # (P, T+1, 2)
        px = all_pos[:, :, 0:1]   # (P, T+1, 1)
        py = all_pos[:, :, 1:2]

        # --- Field border penalty ---
        dist_field = _dist_to_segs_batch(px, py, self.l_p1, self.l_dxdy, self.l_lensq)
        pen_field = cp.maximum(0.0, dist_field - FIELD_BORDER_MAX_M)
        line_pen = cp.sum(pen_field ** 2, axis=1)

        # --- Road exclusion penalty ---
        dist_road_n = _dist_to_segs_batch(px, py, self.rn_p1, self.rn_dxdy, self.rn_lensq)
        pen_road_n = cp.maximum(0.0, self.road_excl_normal - dist_road_n)
        road_pen_n = cp.sum(pen_road_n ** 2, axis=1)

        dist_road_t = _dist_to_segs_batch(px, py, self.rt_p1, self.rt_dxdy, self.rt_lensq)
        pen_road_t = cp.maximum(0.0, self.road_excl_toft - dist_road_t)
        road_pen_t = cp.sum(pen_road_t ** 2, axis=1)
        road_pen = road_pen_n + road_pen_t

        # --- Point exclusion zones ---
        diff_exc = all_pos[:, :, None, :] - self.exc_xy[None, None, :, :]
        dist_exc = cp.sqrt(cp.sum(diff_exc ** 2, axis=3))
        pen_exc = cp.maximum(0.0, self.exc_r[None, None, :] - dist_exc)
        exc_pen = cp.sum(pen_exc ** 2, axis=(1, 2))

        # --- Inter-turbine spacing (turbines only) ---
        diff_inter = turbine_pos[:, :, None, :] - turbine_pos[:, None, :, :]
        dist_inter = cp.sqrt(cp.sum(diff_inter ** 2, axis=3))
        mask = cp.triu(cp.ones((n_turbines, n_turbines), dtype=bool), k=1)
        dist_inter = cp.where(mask, dist_inter, cp.inf)
        pen_inter = cp.maximum(0.0, self.min_spacing - dist_inter)
        inter_pen = cp.sum(pen_inter ** 2, axis=(1, 2))

        # --- Substation-turbine exclusion ---
        diff_sub_turb = turbine_pos - substation_pos  # (P, T, 2)
        dist_sub_turb = cp.sqrt(cp.sum(diff_sub_turb ** 2, axis=2))  # (P, T)
        pen_sub_excl = cp.maximum(0.0, self.substation_excl_radius - dist_sub_turb)
        sub_excl_pen = cp.sum(pen_sub_excl ** 2, axis=1)

        # --- Boundary penalty (CPU, Shapely) ---
        bound_pen_cpu = np.zeros(n_particles)
        batch_pos_cpu = batch_pos.get() if hasattr(batch_pos, 'get') else batch_pos
        for i in range(n_particles):
            pos_list = [(batch_pos_cpu[i, j, 0], batch_pos_cpu[i, j, 1])
                        for j in range(n_turbines + 1)]
            bound_pen_cpu[i] = boundary_penalty(pos_list, self.boundary_polygon)
        bound_pen = cp.array(bound_pen_cpu)

        # --- Substation-only boundary penalty (CPU/Shapely) ---
        sub_bound_pen_cpu = np.zeros(n_particles)
        for i in range(n_particles):
            sx, sy = batch_pos_cpu[i, -1, 0], batch_pos_cpu[i, -1, 1]
            sub_bound_pen_cpu[i] = substation_boundary_penalty(
                (sx, sy), self.boundary_polygon
            )
        sub_bound_pen = cp.array(sub_bound_pen_cpu)

        # Total weighted penalty
        penalty = (LINE_PENALTY_WEIGHT * line_pen +
                   EXCLUSION_PENALTY_WEIGHT * road_pen +
                   BOUNDARY_PENALTY_WEIGHT * bound_pen +
                   EXCLUSION_PENALTY_WEIGHT * exc_pen +
                   INTER_TURBINE_PENALTY_WEIGHT * inter_pen +
                   SUBSTATION_EXCL_PENALTY_WEIGHT * sub_excl_pen +
                   SUBSTATION_BOUNDARY_PENALTY_WEIGHT * sub_bound_pen)

        return penalty, {
            'line_pen': line_pen,
            'road_pen': road_pen,
            'bound_pen': bound_pen,
            'exc_pen': exc_pen,
            'inter_pen': inter_pen,
            'sub_excl_pen': sub_excl_pen,
            'sub_bound_pen': sub_bound_pen
        }