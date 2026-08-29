import pandas as pd
import numpy as np
from datetime import timedelta
import random

def generate_synthetic_ais(metadata_path, output_csv="data/synthetic_ais_data.csv"):
    """
    Reads dataset metadata and generates synthetic AIS vessel trajectories 
    around the locations and times of confirmed oil slicks.
    """
    df = pd.read_excel(metadata_path)
    
    # Filter only for the oil set (oc and ow subsets)
    # Inside generate_synthetic_ais.py
    tag_col = next(c for c in df.columns if 'tag' in str(c).lower() or 'ID' in str(c))
    time_col = next(c for c in df.columns if 'start_time' in str(c).lower())
    lon_col = next(c for c in df.columns if 'patch_ul_lon' in str(c).lower())
    lat_col = next(c for c in df.columns if 'patch_ul_lat' in str(c).lower())
    img_col = next(c for c in df.columns if 'jpg' in str(c).lower() or 'IMAGE' in str(c))
    
    # Apply drop_duplicates here so only 3 vessels are generated per image patch
    oil_df = df[df[tag_col].astype(str).str.startswith(('oc', 'ow'))].drop_duplicates(subset=[img_col]).head(50)
    
    ais_records = []
    
    for _, row in oil_df.iterrows():
        try:
            spill_time = pd.to_datetime(row[time_col])
            base_lon = float(row[lon_col])
            base_lat = float(row[lat_col])
        except Exception:
            continue
            
        # Simulate 3 vessels per spill location
        for i in range(3):
            mmsi = random.randint(200000000, 700000000)
            ship_types = ["Crude Oil Tanker", "Bulk Carrier", "Container Ship"]
            ship_type = random.choice(ship_types)
            
            # Vessel 0 intersects the spill (Suspect). Vessels 1 & 2 pass nearby (Decoys).
            lon_offset_start = -0.1 if i == 0 else random.uniform(-0.5, 0.5)
            lat_offset_start = -0.1 if i == 0 else random.uniform(-0.5, 0.5)
            lon_drift = 0.02
            lat_drift = 0.02
            
            # Generate a 12-hour trajectory with 30-minute ping intervals
            for hour_step in range(24):
                ping_time = spill_time - timedelta(hours=12) + timedelta(minutes=hour_step * 30)
                
                # Suspect vessel crosses the exact coordinates roughly 2 hours before the satellite pass
                if i == 0 and hour_step == 20:
                    current_lon = base_lon
                    current_lat = base_lat
                else:
                    current_lon = base_lon + lon_offset_start + (hour_step * lon_drift)
                    current_lat = base_lat + lat_offset_start + (hour_step * lat_drift)
                
                ais_records.append({
                    "mmsi": mmsi,
                    "timestamp": ping_time,
                    "latitude": current_lat,
                    "longitude": current_lon,
                    "speed_over_ground": random.uniform(8.0, 15.0),
                    "course_over_ground": random.uniform(0, 360),
                    "vessel_name": f"Vessel_{mmsi}",
                    "ship_type": ship_type
                })
                
    synthetic_ais_df = pd.DataFrame(ais_records)
    synthetic_ais_df.to_csv(output_csv, index=False)
    print(f"Generated {len(synthetic_ais_df)} synthetic AIS pings saved to {output_csv}")

if __name__ == "__main__":
    generate_synthetic_ais(
        metadata_path="D:/MINE/Projects/SIH/data/data_table.xlsx", 
        output_csv="D:/MINE/Projects/SIH/data/synthetic_ais_data.csv"
    )