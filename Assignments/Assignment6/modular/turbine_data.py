#!/usr/bin/env python
"""Turbine data loading and capacity-aware calculations."""

import os
import yaml
import numpy as np

from config import TURBINE_LIB, CAPACITY_MIN_MW, CAPACITY_MAX_MW


def get_turbine_power_mw(turbine_name):
    """Extract rated power in MW from the turbine YAML."""
    yaml_path = os.path.join(TURBINE_LIB, turbine_name + ".yaml")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    rated_kw = None
    if 'rated_power' in data:
        rated_kw = float(data['rated_power'])
    elif 'power_thrust_table' in data and 'power' in data['power_thrust_table']:
        rated_kw = max(float(p) for p in data['power_thrust_table']['power'])
    else:
        fallback = {
            "IEA_3_4MW": 3370.0,
            "BAR_BAU_IEA_3.3MW": 3300.0,
            "BAR_BAU_LSP_3.25MW": 3250.0,
        }
        rated_kw = fallback.get(turbine_name, 3370.0)

    return rated_kw / 1000.0  # kW -> MW


def get_turbine_specs(turbine_name):
    """Load full turbine specifications from YAML."""
    yaml_path = os.path.join(TURBINE_LIB, turbine_name + ".yaml")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def valid_turbine_counts(turbine_name, cap_min_mw=CAPACITY_MIN_MW, cap_max_mw=CAPACITY_MAX_MW):
    """Return (n_min, n_max, p_mw) for this turbine type to stay within capacity bounds."""
    p_mw = get_turbine_power_mw(turbine_name)
    n_min = int(np.ceil(cap_min_mw / p_mw))
    n_max = int(np.floor(cap_max_mw / p_mw))
    n_min = max(1, n_min)
    n_max = max(n_min, n_max)
    return n_min, n_max, p_mw


def load_turbine_type(turbine_name):
    """Load turbine data and return key parameters."""
    data = get_turbine_specs(turbine_name)
    hh = data['hub_height']
    rotor_diameter = data['rotor_diameter']
    min_spacing = 2 * rotor_diameter
    return {
        'name': turbine_name,
        'hub_height': hh,
        'rotor_diameter': rotor_diameter,
        'min_spacing': min_spacing,
        'yaml_path': os.path.join(TURBINE_LIB, turbine_name + ".yaml")
    }