import requests
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

from inference_stage1 import run_stage1_inference
from georeference import bbox_to_geojson_polygon
from export_payload import build_backend_payload

def process_batch_directory(val_images_dir, weights_path, metadata_path, backend_api_url):
    df = pd.read_excel(metadata_path)
    img_col = next(c for c in df.columns if 'jpg' in str(c).lower() or 'IMAGE' in str(c))
    time_col = next(c for c in df.columns if 'start_time' in str(c).lower())
    
    val_dir = Path(val_images_dir)
    image_paths = list(val_dir.glob("*.jpg"))
    
    if not image_paths:
        print(f"No images found in {val_images_dir}")
        return

    print(f"Starting batch processing of {len(image_paths)} images...")
    
    success_count = 0
    
    for img_path in image_paths:
        img_filename = img_path.name
        print(f"\nProcessing {img_filename}...")
        
        # 1. Run Inference
        detections = run_stage1_inference(weights_path, str(img_path), conf_threshold=0.50)
        if not detections:
            print("No spills detected. Skipping.")
            continue

        # 2. Extract Metadata
        record = df[df[img_col] == img_filename]
        if record.empty:
            print("Metadata missing. Cannot georeference.")
            continue
            
        record = record.iloc[0]
        acquisition_time = pd.to_datetime(record[time_col])

        corners = {
            'ul': (float(record['Longitude (patch_ul_lon)']), float(record['Latitude (patch_ul_lat)'])),
            'ur': (float(record['Longitude (patch_ur_lon)']), float(record['Latitude (patch_ur_lat)'])),
            'br': (float(record['Longitude (patch_br_lon)']), float(record['Latitude (patch_br_lat)'])),
            'bl': (float(record['Longitude (patch_bl_lon)']), float(record['Latitude (patch_bl_lat)']))
        }
        
        patch_width = float(record.get('Width [pixel] (patch_width)', 1250))
        patch_height = float(record.get('Height [pixel] (patch_height)', 1250))

        # 3. Process Detections
        for det in detections:
            bbox_pixels = det["bbox_pixels"]
            confidence = det["confidence"]

            geojson_poly = bbox_to_geojson_polygon(bbox_pixels, patch_width, patch_height, corners)
            wgs84_coords = geojson_poly['coordinates'][0]
            payload = build_backend_payload(wgs84_coords, acquisition_time, confidence)
            
            # 4. Transmit
            try:
                response = requests.post(backend_api_url, json=payload, headers={"Content-Type": "application/json"})
                if response.status_code in [200, 201]:
                    print(f"  -> Transmitted payload {payload['spill_id']}")
                    success_count += 1
                else:
                    print(f"  -> Backend rejected: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"  -> Connection failed: {e}")
                
            # Throttle requests to avoid overwhelming the server
            time.sleep(1)

    print(f"\nBatch processing complete. Successfully transmitted {success_count} spills.")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    process_batch_directory(
        val_images_dir=str(PROJECT_ROOT / "data" / "yolo_dataset" / "images" / "val"),
        weights_path=str(PROJECT_ROOT /"runs" / "detect" / "oil_spill_detection" / "stage1_yolo" / "weights" / "best.pt"),
        metadata_path=str(PROJECT_ROOT / "data" / "data_table.xlsx"),
        backend_api_url="	https://webhook.site/06dfc3bf-1d80-44c1-82ad-a8ad33d02e4a"  # Replace with actual API once ready
    )