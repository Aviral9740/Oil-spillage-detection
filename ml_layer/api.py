from fastapi import FastAPI, UploadFile, File, Form
from datetime import datetime
from pathlib import Path
import shutil
import os
from ultralytics import YOLO

from .georeference import bbox_to_geojson_polygon
from .export_payload import build_backend_payload

app = FastAPI()

CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = CURRENT_DIR / "weights" / "best.pt"

model = YOLO(str(WEIGHTS_PATH))

@app.post("/predict")
async def predict_spill(
    file: UploadFile = File(...),
    ul_lon: float = Form(24.0000),
    ul_lat: float = Form(35.1000),
    ur_lon: float = Form(24.1000),
    ur_lat: float = Form(35.1000),
    bl_lon: float = Form(24.0000),
    bl_lat: float = Form(35.0000),
    br_lon: float = Form(24.1000),
    br_lat: float = Form(35.0000)
):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Pass the image directly to the globally loaded model
    inference_results = model(temp_path, conf=0.50)
    os.remove(temp_path)
    
    # Extract bounding boxes and exact image dimensions dynamically
    detections = []
    for r in inference_results:
        img_height, img_width = r.orig_shape
        for box in r.boxes:
            detections.append({
                "bbox_pixels": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "width": img_width,
                "height": img_height
            })
    
    if not detections:
        return {"status": "no_spill_detected", "detections": []}
        
    current_time = datetime.utcnow()
    
    corners = {
        'ul': (ul_lon, ul_lat),
        'ur': (ur_lon, ur_lat),
        'bl': (bl_lon, bl_lat),
        'br': (br_lon, br_lat)
    }
        
    results = []
    for det in detections:
        geojson_poly = bbox_to_geojson_polygon(
            det["bbox_pixels"], 
            det["width"], 
            det["height"], 
            corners
        )
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

@app.get("/")
async def root():
    return {"message": "Oil Spill Detection API is running. Send POST requests to /predict."}