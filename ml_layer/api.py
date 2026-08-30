from fastapi import FastAPI, UploadFile, File
from .inference_stage1 import run_stage1_inference
from .georeference import bbox_to_geojson_polygon
from .export_payload import build_backend_payload
import shutil
import os
from datetime import datetime
from pathlib import Path

app = FastAPI()
CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = CURRENT_DIR / "weights" / "best.pt"

@app.post("/predict")
async def predict_spill(file: UploadFile = File(...)):
    # 1. Save uploaded image temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Run Inference
    # Ensure the path to your weights file is correct for your deployment structure
    detections = run_stage1_inference("weights/best.pt", temp_path, conf_threshold=0.50)
    os.remove(temp_path)
    
    if not detections:
        return {"status": "no_spill_detected", "detections": []}
        
    # 3. Define fallback metadata variables to satisfy Pylance
    current_time = datetime.utcnow()
    
    # Mock geographic coordinates (approximate bounding box in the Gulf of Mexico)
    dummy_corners = {
        'ul': (-88.3100, 28.5150),
        'ur': (-88.3000, 28.5150),
        'br': (-88.3000, 28.5100),
        'bl': (-88.3100, 28.5100)
    }
        
    # 4. Process and Return JSON Schema
    results = []
    for det in detections:
        # Pass the newly defined dummy_corners and current_time
        geojson_poly = bbox_to_geojson_polygon(det["bbox_pixels"], 1250, 1250, dummy_corners)
        wgs84_coords = geojson_poly['coordinates'][0]
        
        payload = build_backend_payload(wgs84_coords, current_time, det["confidence"])
        results.append(payload)
        
    return {"status": "success", "detections": results}

@app.get("/health")
async def health_check():
    return {
        "status": "active",
        "timestamp": datetime.utcnow().isoformat()
    }