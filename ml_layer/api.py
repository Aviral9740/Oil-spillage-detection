from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from datetime import datetime
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from .georeference import bbox_to_geojson_polygon
from .export_payload import build_backend_payload

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = CURRENT_DIR / "weights" / "best.onnx"

model = YOLO(str(WEIGHTS_PATH))

# NOTE: Removed 'async' from def predict_spill to run the heavy model on a background thread
@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_spill(
    file: UploadFile = File(..., description="Satellite image (.jpg or .png)"),
    capture_time: str = Form(..., description="Satellite capture timestamp in ISO format (e.g., 2026-09-01T11:05:00)"),
    # Upper-Left
    ul_lat: float = Form(..., description="Upper-Left Latitude"),
    ul_lon: float = Form(..., description="Upper-Left Longitude"),
    # Upper-Right
    ur_lat: float = Form(..., description="Upper-Right Latitude"),
    ur_lon: float = Form(..., description="Upper-Right Longitude"),
    # Bottom-Left
    bl_lat: float = Form(..., description="Bottom-Left Latitude"),
    bl_lon: float = Form(..., description="Bottom-Left Longitude"),
    # Bottom-Right
    br_lat: float = Form(..., description="Bottom-Right Latitude"),
    br_lon: float = Form(..., description="Bottom-Right Longitude"),
    conf_threshold: float = Form(0.50, description="Confidence threshold for detection")
) -> Dict[str, Any]:
    
    # 1. Validate & Parse Timestamp
    parsed_time = None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed_time = datetime.strptime(capture_time, fmt)
            break
        except ValueError:
            continue

    if not parsed_time:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid capture_time format. Supported formats: 'YYYY-MM-DDTHH:MM:SS', 'YYYY-MM-DDTHH:MM:SSZ', or 'YYYY-MM-DD HH:MM:SS'"
        )

    # 2. Package Spatial Bounding Corners
    corners = {
        'ul': (ul_lon, ul_lat),
        'ur': (ur_lon, ur_lat),
        'bl': (bl_lon, bl_lat),
        'br': (br_lon, br_lat)
    }

    try:
        # 3. Read image directly into RAM (Bypasses Disk I/O completely)
        img = Image.open(file.file)
        
        # 4. Run YOLO ONNX Inference on RAM object with resolution locked to 640
        inference_results = model(img, conf=conf_threshold, imgsz=640)
        
        detections: List[Dict[str, Any]] = []

        for r in inference_results:
            img_height, img_width = r.orig_shape
            for box in r.boxes:
                bbox_pixels = box.xyxy[0].tolist()
                confidence = float(box.conf[0])

                # Convert pixel bounding box to geographic WGS84 polygon
                geojson_poly = bbox_to_geojson_polygon(bbox_pixels, img_width, img_height, corners)
                wgs84_coords = geojson_poly['coordinates'][0]

                # Use original filename as reference
                payload = build_backend_payload(
                    wgs84_coords,
                    parsed_time,
                    confidence,
                    file.filename
                )
                detections.append(payload)

        return {
            "status": "success",
            "filename": file.filename,
            "detected_at": parsed_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "spills_found": len(detections),
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "active",
    }

@app.get("/")
async def root():
    return {"message": "Oil Spill Detection API is running. Send POST requests to /predict."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)