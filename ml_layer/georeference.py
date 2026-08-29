import pandas as pd
import numpy as np
from shapely.geometry import Polygon, mapping

def pixel_to_wgs84(pixel_x, pixel_y, patch_width, patch_height, corners):
    """
    Interpolates pixel coordinates (x, y) into WGS84 (lon, lat) using 4 corner coordinates.
    corners = {
        'ul': (lon, lat), 'ur': (lon, lat),
        'br': (lon, lat), 'bl': (lon, lat)
    }
    """
    u = pixel_x / patch_width
    v = pixel_y / patch_height

    # Bilinear interpolation
    lon = (1 - u) * (1 - v) * corners['ul'][0] + \
          u * (1 - v) * corners['ur'][0] + \
          u * v * corners['br'][0] + \
          (1 - u) * v * corners['bl'][0]

    lat = (1 - u) * (1 - v) * corners['ul'][1] + \
          u * (1 - v) * corners['ur'][1] + \
          u * v * corners['br'][1] + \
          (1 - u) * v * corners['bl'][1]

    return lon, lat

def bbox_to_geojson_polygon(bbox, patch_width, patch_height, corners):
    xmin, ymin, xmax, ymax = bbox
    
    p_ul = pixel_to_wgs84(xmin, ymin, patch_width, patch_height, corners)
    p_ur = pixel_to_wgs84(xmax, ymin, patch_width, patch_height, corners)
    p_br = pixel_to_wgs84(xmax, ymax, patch_width, patch_height, corners)
    p_bl = pixel_to_wgs84(xmin, ymax, patch_width, patch_height, corners)

    # GeoJSON coordinates format: [[lon, lat], ...] closed polygon
    coords = [p_ul, p_ur, p_br, p_bl, p_ul]
    poly = Polygon(coords)
    return mapping(poly)