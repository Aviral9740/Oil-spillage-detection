import json
import uuid
import math
from datetime import datetime
from pyproj import Geod
from shapely.geometry import Polygon

def estimate_spill_age_heuristic(area_m2, perimeter_m):
    """
    Estimates spill age based on geometric spreading (Isoperimetric quotient).
    Fresh spills are compact (circular); older spills are highly streaky/elongated.
    """
    if perimeter_m == 0:
        return 2.0
        
    # Calculate compactness (Circle = 1.0, infinitely thin line approaches 0.0)
    compactness = (4 * math.pi * area_m2) / (perimeter_m ** 2)
    
    # Cap compactness to prevent math errors on weird geometries
    compactness = max(0.01, min(compactness, 1.0))
    
    # Map compactness to an empirical age bound (e.g., 2 hours to 72 hours)
    # The less compact it is, the older it is assumed to be.
    age_hours = 72 * (1 - compactness)
    
    # Return bounded estimate
    return max(2.0, min(round(age_hours, 1), 72.0))

def build_backend_payload(wgs84_polygon_coords, acquisition_time, confidence, image_reference):
    """
    Transforms georeferenced polygon coordinates into the backend JSON schema.
    Now includes the hosted image URL for the frontend.
    """
    poly = Polygon(wgs84_polygon_coords)
    centroid_lon, centroid_lat = poly.centroid.x, poly.centroid.y
    
    # Calculate physical area and perimeter using WGS84 ellipsoid projection
    geod = Geod(ellps="WGS84")
    area_m2, perimeter_m = geod.geometry_area_perimeter(poly)
    area_m2 = abs(area_m2) 
    area_km2 = area_m2 / 1_000_000
    
    # Dynamically estimate age based on polygon shape degradation
    estimated_age = estimate_spill_age_heuristic(area_m2, perimeter_m)
    
    payload = {
        "spill_id": f"spill_{uuid.uuid4().hex[:6]}",
        "detected_at": acquisition_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "centroid": {
            "lon": round(centroid_lon, 4),
            "lat": round(centroid_lat, 4)
        },
        "polygon": [[round(lon, 4), round(lat, 4)] for lon, lat in wgs84_polygon_coords],
        "area_km2": round(area_km2, 2),
        "estimated_age_hours": estimated_age,
        "confidence_score": round(float(confidence), 2),
        "image_reference": image_reference
    }
    
    return payload

if __name__ == "__main__":
    # Mock data bridging the YOLO output to the backend
    mock_coords = [
        [-88.3100, 28.5100], [-88.3080, 28.5150], 
        [-88.3000, 28.5140], [-88.3100, 28.5100]
    ]
    mock_time = datetime.utcnow()
    mock_image_url = "https://res.cloudinary.com/bro6lw9c/image/upload/v1788199127/oc-0375.jpg"
    
    final_json = json.dumps(build_backend_payload(mock_coords, mock_time, 0.8754, mock_image_url), indent=2)
    print(final_json)