import pandas as pd
from shapely.geometry import shape, Point, LineString
from datetime import datetime, timedelta

def find_suspect_vessels(spill_geojson, acquisition_time, ais_records_df, time_window_hours=12, spatial_buffer_deg=0.05):
    """
    Identifies vessels whose historical trajectory intersects the spill area 
    within a prior time window.
    """
    spill_poly = shape(spill_geojson).buffer(spatial_buffer_deg)
    
    # 1. Temporal filter: look back N hours before the satellite pass
    start_time = acquisition_time - timedelta(hours=time_window_hours)
    end_time = acquisition_time + timedelta(hours=1)
    
    ais_records_df['timestamp'] = pd.to_datetime(ais_records_df['timestamp'])
    filtered_ais = ais_records_df[
        (ais_records_df['timestamp'] >= start_time) & 
        (ais_records_df['timestamp'] <= end_time)
    ]
    
    suspects = []
    
    # 2. Group by vessel (MMSI) and construct trajectories
    for mmsi, group in filtered_ais.groupby('mmsi'):
        sorted_group = group.sort_values('timestamp')
        
        if len(sorted_group) < 2:
            # Single ping: check point distance
            pt = Point(sorted_group.iloc[0]['longitude'], sorted_group.iloc[0]['latitude'])
            if spill_poly.contains(pt):
                suspects.append({
                    "mmsi": mmsi,
                    "vessel_name": sorted_group.iloc[0].get('vessel_name', 'Unknown'),
                    "ship_type": sorted_group.iloc[0].get('ship_type', 'Unknown'),
                    "closest_time": sorted_group.iloc[0]['timestamp'],
                    "status": "Point Inside Buffer"
                })
        else:
            # Line trajectory
            coords = list(zip(sorted_group['longitude'], sorted_group['latitude']))
            trajectory = LineString(coords)
            
            if trajectory.intersects(spill_poly):
                suspects.append({
                    "mmsi": mmsi,
                    "vessel_name": sorted_group.iloc[0].get('vessel_name', 'Unknown'),
                    "ship_type": sorted_group.iloc[0].get('ship_type', 'Unknown'),
                    "first_seen": sorted_group['timestamp'].min(),
                    "last_seen": sorted_group['timestamp'].max(),
                    "status": "Trajectory Intersected"
                })
                
    return suspects