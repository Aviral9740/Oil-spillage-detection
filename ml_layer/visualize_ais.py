import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def visualize_intersection():
    # Enforce a dark aesthetic for high-contrast geospatial plotting
    plt.style.use('dark_background')
    
    project_root = Path(__file__).resolve().parent.parent
    
    # Adjust to "data" or "dataset_root" depending on where your file is located
    meta_path = project_root / "dataset_root" / "data_table.xlsx"
    if not meta_path.exists():
        meta_path = project_root / "data" / "data_table.xlsx"
        
    ais_path = project_root / "data" / "synthetic_ais_data.csv"
    
    # 1. Load Metadata and AIS Data
    meta_df = pd.read_excel(meta_path)
    ais_df = pd.read_csv(ais_path)
    
    # Locate necessary columns dynamically
    tag_col = next(c for c in meta_df.columns if 'tag' in str(c).lower() or 'ID' in str(c))
    img_col = next(c for c in meta_df.columns if 'jpg' in str(c).lower() or 'IMAGE' in str(c))
    time_col = next(c for c in meta_df.columns if 'start_time' in str(c).lower())
    
    # Ensure synthetic generation only runs once per image, not once per object
    oil_df = meta_df[meta_df[tag_col].astype(str).str.startswith(('oc', 'ow'))].drop_duplicates(subset=[img_col]).head(50)
    
    # Grab the target valid oil record (matching your synthetic test case index)
    oil_record = oil_df.iloc[35]
    image_filename = oil_record[img_col]
    spill_time = pd.to_datetime(oil_record[time_col])
    
    # Locate the physical image file
    img_path = next(project_root.rglob(image_filename), None)
    if not img_path:
        print(f"Image {image_filename} not found.")
        return

    # 2. Extract bounding box extent (Longitude / Latitude)
    lon_cols = [c for c in meta_df.columns if 'lon' in c.lower() and 'patch' in c.lower()]
    lat_cols = [c for c in meta_df.columns if 'lat' in c.lower() and 'patch' in c.lower()]
    
    min_lon, max_lon = oil_record[lon_cols].min(), oil_record[lon_cols].max()
    min_lat, max_lat = oil_record[lat_cols].min(), oil_record[lat_cols].max()
    
    # 3. Prepare the Image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot the image mapped to its geographic coordinates
    ax.imshow(img, cmap='gray', extent=[min_lon, max_lon, min_lat, max_lat], origin='upper')
    
    # 4. Plot AIS Trajectories
    # Convert AIS timestamps to datetime for accurate filtering
    ais_df['timestamp'] = pd.to_datetime(ais_df['timestamp'])
    
    # Filter AIS data to only ships near this image spatially AND temporally (± 24 hours)
    local_ais = ais_df[
        (ais_df['longitude'] >= min_lon - 0.2) & (ais_df['longitude'] <= max_lon + 0.2) &
        (ais_df['latitude'] >= min_lat - 0.2) & (ais_df['latitude'] <= max_lat + 0.2) &
        (ais_df['timestamp'] >= spill_time - pd.Timedelta(hours=24)) & 
        (ais_df['timestamp'] <= spill_time + pd.Timedelta(hours=24))
    ]
    
    colors = ['#00FFCC', '#FF00FF', '#FFFF00'] # Neon cyan, magenta, yellow
    
    for idx, (mmsi, group) in enumerate(local_ais.groupby('mmsi')):
        sorted_group = group.sort_values('timestamp')
        color = colors[idx % len(colors)]
        
        # Plot trajectory line
        ax.plot(
            sorted_group['longitude'], sorted_group['latitude'], 
            color=color, linewidth=2, linestyle='--', label=f"Vessel {mmsi}"
        )
        # Plot individual pings as markers
        ax.scatter(
            sorted_group['longitude'], sorted_group['latitude'], 
            color=color, s=30, zorder=5
        )

    # 5. Format the Chart
    ax.set_xlim(min_lon - 0.05, max_lon + 0.05)
    ax.set_ylim(min_lat - 0.05, max_lat + 0.05)
    ax.set_title(f"AIS Intersects for Satellite Pass: {image_filename}\nAcquired: {spill_time}", fontsize=14, color='white', pad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", facecolor='black', edgecolor='white')
    ax.grid(color='#333333', linestyle=':', linewidth=1)
    
    plt.show()

if __name__ == "__main__":
    visualize_intersection()