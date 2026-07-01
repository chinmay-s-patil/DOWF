#!/usr/bin/env python
"""KML parsing, coordinate conversion, and geometry extraction."""

import os
from xml.etree import ElementTree as ET
import numpy as np
from shapely.geometry import Point, Polygon

from config import (
    KML_PATH, LAT_TO_M, LON_TO_M_AT_55, REF_LAT, REF_LON,
    ROAD_BUFFER_M, TOWN_RADIUS
)

_KML_NS = {'kml': 'http://www.opengis.net/kml/2.2'}


def get_ref(stri):
	# This is using the DMS format - Degree Minutes Seconds

	lattLB = 55 + (14/60) + (39.2/(3600))
	longLB = 9 + (00/60) + (41.3/(3600))

	latt_deg = float(stri.strip().split("°")[0])
	latt_min = float(stri.strip().split("'")[0].split("°")[1])
	latt_sec = float(stri.strip().split("\"")[0].split("°")[1].split("'")[1].replace("\\", ""))

	long_deg = float(stri.strip().split("N")[1].split("°")[0])
	long_min = float(stri.strip().split("N")[1].split("'")[0].split("°")[1])
	long_sec = float(stri.strip().split("N")[1].split("\"")[0].split("°")[1].split("'")[1].replace("\\", ""))

	latt = latt_deg + (latt_min/60) + (latt_sec/(3600))
	long = long_deg + (long_min/60) + (long_sec/(3600))

	latt = (latt - REF_LAT) * LAT_TO_M

	long = (long - REF_LON) * LON_TO_M_AT_55

	return latt, long


def get_ref_WGS84(lat, lon):
    """Convert WGS84 lat/lon to local metres relative to reference point."""
    latt = (lat - REF_LAT) * LAT_TO_M
    long = (lon - REF_LON) * LON_TO_M_AT_55
    return latt, long


def _parse_coords(text):
    """Parse a KML coordinates string -> list of (lon, lat) floats."""
    out = []
    for tok in text.strip().split():
        parts = tok.split(',')
        out.append((float(parts[0]), float(parts[1])))
    return out


def _kml_placemarks():
    """Yield (name, placemark_element) for every Placemark in the KML."""
    tree = ET.parse(KML_PATH)
    root = tree.getroot()
    for pm in root.iter('{http://www.opengis.net/kml/2.2}Placemark'):
        n = pm.find('kml:name', _KML_NS)
        yield (n.text if n is not None else ''), pm


def road_buffer_polygon(x0, y0, x1, y1, width):
    """Returns the 4 corners of a rectangle buffering a line segment."""
    dx, dy = x1 - x0, y1 - y0
    length = np.hypot(dx, dy)
    if length == 0:
        return [(x0, y0), (x0, y0), (x0, y0), (x0, y0)]
    px, py = -dy / length, dx / length
    offset = width / 2
    return [
        (x0 + px * offset, y0 + py * offset),
        (x1 + px * offset, y1 + py * offset),
        (x1 - px * offset, y1 - py * offset),
        (x0 - px * offset, y0 - py * offset),
    ]


class SiteGeometry:
    """Holds all site geometry extracted from KML."""

    def __init__(self):
        self.boundary_x = []
        self.boundary_y = []
        self.road_segments_normal = []
        self.road_segments_toft = []
        self.road_segments = []
        self.builds = []
        self.wts = []
        self.line_segments = []
        self.town_pos = None
        self.substation_pos = None
        self._parse_kml()

    def _parse_kml(self):
        # --- Site boundary ---
        poly_coords = None
        for name, pm in _kml_placemarks():
            if name == 'Site Area':
                poly_coords = _parse_coords(
                    pm.find('.//kml:coordinates', _KML_NS).text)
                break

        if poly_coords is None:
            raise ValueError("Site Area not found in KML")

        if poly_coords[-1] == poly_coords[0]:
            poly_coords = poly_coords[:-1]

        boundary_m = [get_ref_WGS84(lat, lon) for lon, lat in poly_coords]
        self.boundary_x = [lon_m for (_lat_m, lon_m) in boundary_m]
        self.boundary_y = [lat_m for (lat_m, _lon_m) in boundary_m]

        # --- Roads ---
        for name, pm in _kml_placemarks():
            ls = pm.find('kml:LineString', _KML_NS)
            if ls is None:
                continue
            pts = _parse_coords(ls.find('kml:coordinates', _KML_NS).text)
            pts_m = [get_ref_WGS84(lat, lon) for lon, lat in pts]
            segs = []
            for i in range(len(pts_m) - 1):
                segs.append(((pts_m[i][1], pts_m[i][0]),
                             (pts_m[i+1][1], pts_m[i+1][0])))
            if 'Kreisstra' in name or 'kreisstra' in name:
                self.road_segments_normal.extend(segs)
            elif 'Staatsstra' in name or 'staatsstra' in name:
                self.road_segments_toft.extend(segs)

        self.road_segments = self.road_segments_normal + self.road_segments_toft

        # --- Town (Kastrup) ---
        self.town_pos = get_ref("55°15'46.7\"N 9°04'46.7\"E")

        # --- Substation ---
        substation_found = False
        for name, pm in _kml_placemarks():
            if 'substation' in name.lower():
                pt = pm.find('kml:Point', _KML_NS)
                if pt is not None:
                    c = _parse_coords(pt.find('kml:coordinates', _KML_NS).text)[0]
                    self.substation_pos = get_ref_WGS84(c[1], c[0])
                    substation_found = True
                    break

        if not substation_found:
            # Compute centroid of site boundary as substation position
            self.substation_pos = (
                np.mean(self.boundary_y),
                np.mean(self.boundary_x)
            )

        # --- Buildings / Houses ---
        for name, pm in _kml_placemarks():
            if name.startswith('House') or name == 'Untitled placemark':
                pt = pm.find('kml:Point', _KML_NS)
                if pt is None:
                    continue
                c = _parse_coords(pt.find('kml:coordinates', _KML_NS).text)[0]
                self.builds.append(get_ref_WGS84(c[1], c[0]))

        # --- Existing Wind Turbines ---
        for name, pm in _kml_placemarks():
            if name.startswith('WTG'):
                pt = pm.find('kml:Point', _KML_NS)
                if pt is None:
                    continue
                c = _parse_coords(pt.find('kml:coordinates', _KML_NS).text)[0]
                self.wts.append(get_ref_WGS84(c[1], c[0]))

        # --- Field Borders ---
        fb = []
        for name, pm in _kml_placemarks():
            if name == 'Borders Green Fields':
                coords_text = pm.find('.//kml:Polygon//kml:outerBoundaryIs'
                                      '//kml:coordinates', _KML_NS).text
                raw = _parse_coords(coords_text)
                if raw[-1] == raw[0]:
                    raw = raw[:-1]
                fb = [get_ref_WGS84(lat, lon) for lon, lat in raw]
                break

        # Build LINE_SEGMENTS from polygon edges
        for i in range(len(fb) - 1):
            self.line_segments.append([(fb[i][1], fb[i][0]),
                                        (fb[i+1][1], fb[i+1][0])])
        if len(fb) > 2:
            self.line_segments.append([(fb[-1][1], fb[-1][0]),
                                        (fb[0][1], fb[0][0])])

        # Add "Untitled path" line segments
        for name, pm in _kml_placemarks():
            if name == 'Untitled path':
                ls = pm.find('kml:LineString', _KML_NS)
                if ls is None:
                    continue
                pts = _parse_coords(ls.find('kml:coordinates', _KML_NS).text)
                pts_m = [get_ref_WGS84(lat, lon) for lon, lat in pts]
                for i in range(len(pts_m) - 1):
                    self.line_segments.append(
                        [(pts_m[i][1], pts_m[i][0]),
                         (pts_m[i+1][1], pts_m[i+1][0])])

        print(f"KML loaded: {len(self.boundary_x)} boundary pts, "
              f"{len(self.road_segments)} road segs, {len(self.builds)} buildings, "
              f"{len(self.wts)} WTGs, {len(fb)} field border pts, "
              f"{len(self.line_segments)} line segments, "
              f"substation at ({self.substation_pos[0]:.1f}, {self.substation_pos[1]:.1f})")

    @property
    def boundary_polygon(self):
        """Return shapely Polygon of site boundary."""
        return Polygon(zip(self.boundary_x, self.boundary_y))

    @property
    def x_min(self):
        return min(self.boundary_x)

    @property
    def x_max(self):
        return max(self.boundary_x)

    @property
    def y_min(self):
        return min(self.boundary_y)

    @property
    def y_max(self):
        return max(self.boundary_y)

    @property
    def bounds(self):
        """Return optimizer bounds for max possible turbines."""
        from config import N_MAX
        n = N_MAX
        return [(self.x_min, self.x_max) if i % 2 == 0 else (self.y_min, self.y_max)
                for i in range(2 * n)]