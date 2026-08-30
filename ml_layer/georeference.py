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

def bbox_to_geojson_polygon(bbox_pixels, patch_width, patch_height, corners):
    """
    Translates YOLO pixel bounding boxes to WGS84 coordinates using Bilinear Interpolation 
    to account for satellite orbit rotation and image skew.
    """
    xmin, ymin, xmax, ymax = bbox_pixels
    
    ul_lon, ul_lat = corners['ul']
    ur_lon, ur_lat = corners['ur']
    bl_lon, bl_lat = corners['bl']
    br_lon, br_lat = corners['br']
    
    def pixel_to_coords(x, y):
        # Normalize pixel coordinates (0.0 to 1.0)
        nx = x / patch_width
        ny = y / patch_height
        
        # 1. Interpolate across the Top and Bottom edges (Longitude)
        top_lon = ul_lon + nx * (ur_lon - ul_lon)
        bottom_lon = bl_lon + nx * (br_lon - bl_lon)
        
        # 2. Interpolate across the Top and Bottom edges (Latitude)
        top_lat = ul_lat + nx * (ur_lat - ul_lat)
        bottom_lat = bl_lat + nx * (br_lat - bl_lat)
        
        # 3. Interpolate vertically down to the specific Y pixel
        lon = top_lon + ny * (bottom_lon - top_lon)
        lat = top_lat + ny * (bottom_lat - top_lat)
        
        return [round(lon, 6), round(lat, 6)]

    # Convert the 4 pixel corners of the bounding box
    top_left = pixel_to_coords(xmin, ymin)
    top_right = pixel_to_coords(xmax, ymin)
    bottom_right = pixel_to_coords(xmax, ymax)
    bottom_left = pixel_to_coords(xmin, ymax)
    
    return {
        "type": "Polygon",
        "coordinates": [[
            top_left, 
            top_right, 
            bottom_right, 
            bottom_left, 
            top_left
        ]]
    }