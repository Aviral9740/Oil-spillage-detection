import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon, mapping
from ais_correlation import find_suspect_vessels

def test_correlation():
    project_root = Path(__file__).resolve().parent.parent
    
    # 1. Load the synthetic AIS data
    ais_path = project_root / "data" / "synthetic_ais_data.csv"
    if not ais_path.exists():
        print(f"Error: Could not find synthetic AIS data at {ais_path}")
        return
        
    ais_df = pd.read_csv(ais_path)
    
    # 2. Fetch the first spill location to match our synthetic data generation
    meta_df = pd.read_excel(project_root / "data" / "data_table.xlsx")
    
    # Find tag, time, and coordinate columns dynamically
    tag_col = next(c for c in meta_df.columns if 'tag' in str(c).lower() or 'ID' in str(c))
    time_col = next(c for c in meta_df.columns if 'start_time' in str(c).lower())
    lon_col = next(c for c in meta_df.columns if 'patch_ul_lon' in str(c).lower())
    lat_col = next(c for c in meta_df.columns if 'patch_ul_lat' in str(c).lower())
    
    # Grab the first valid oil record
    oil_record = meta_df[meta_df[tag_col].astype(str).str.startswith(('oc', 'ow'))].iloc[0]
    
    spill_time = pd.to_datetime(oil_record[time_col])
    base_lon = float(oil_record[lon_col])
    base_lat = float(oil_record[lat_col])
    
    # 3. Create a mock GeoJSON polygon around this coordinate (approx 10x10 km)
    offset = 0.05
    poly = Polygon([
        (base_lon - offset, base_lat - offset),
        (base_lon + offset, base_lat - offset),
        (base_lon + offset, base_lat + offset),
        (base_lon - offset, base_lat + offset),
        (base_lon - offset, base_lat - offset)
    ])
    spill_geojson = mapping(poly)
    
    # 4. Run the correlation
    print(f"Testing intersection for spill at {spill_time}...")
    suspects = find_suspect_vessels(
        spill_geojson=spill_geojson,
        acquisition_time=spill_time,
        ais_records_df=ais_df,
        time_window_hours=12,
        spatial_buffer_deg=0.05
    )
    
    # 5. Display results
    print(f"\n--- Found {len(suspects)} suspect vessel(s) ---")
    for s in suspects:
        print(f"MMSI: {s['mmsi']} | Type: {s['ship_type']} | Status: {s['status']}")

if __name__ == "__main__":
    test_correlation()