class CornerValidationError(ValueError):
    """Raised when the four boundary corners are missing, malformed, or
    describe a footprint this module can't safely interpolate (e.g. an
    antimeridian crossing)."""


def bbox_to_geojson_polygon(bbox_pixels, patch_width, patch_height, corners):
    """
    Translates a YOLO pixel bounding box [xmin, ymin, xmax, ymax] into a
    WGS84 GeoJSON polygon using bilinear interpolation across the four
    corner coordinates, to account for satellite orbit rotation and
    image skew.

    corners = {
        'ul': (lon, lat), 'ur': (lon, lat),
        'bl': (lon, lat), 'br': (lon, lat)
    }
    """
    required = {"ul", "ur", "bl", "br"}
    if set(corners) != required:
        raise CornerValidationError(f"corners must have exactly keys {sorted(required)}")

    xmin, ymin, xmax, ymax = bbox_pixels

    ul_lon, ul_lat = corners["ul"]
    ur_lon, ur_lat = corners["ur"]
    bl_lon, bl_lat = corners["bl"]
    br_lon, br_lat = corners["br"]

    def pixel_to_coords(x, y):
        nx = x / patch_width
        ny = y / patch_height

        top_lon = ul_lon + nx * (ur_lon - ul_lon)
        bottom_lon = bl_lon + nx * (br_lon - bl_lon)
        top_lat = ul_lat + nx * (ur_lat - ul_lat)
        bottom_lat = bl_lat + nx * (br_lat - bl_lat)

        lon = top_lon + ny * (bottom_lon - top_lon)
        lat = top_lat + ny * (bottom_lat - top_lat)

        return [round(lon, 6), round(lat, 6)]

    top_left = pixel_to_coords(xmin, ymin)
    top_right = pixel_to_coords(xmax, ymin)
    bottom_right = pixel_to_coords(xmax, ymax)
    bottom_left = pixel_to_coords(xmin, ymax)

    return {
        "type": "Polygon",
        "coordinates": [[top_left, top_right, bottom_right, bottom_left, top_left]],
    }