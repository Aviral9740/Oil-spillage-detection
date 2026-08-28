from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def run_stage1_inference(weights_path, image_path, conf_threshold=0.25):
    """
    Runs YOLO inference on a normalized patch or SAR scene tile.
    Returns detected bounding boxes in pixel coordinates: [xmin, ymin, xmax, ymax, conf].
    """
    model = YOLO(weights_path)
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project="inference_results",
        name="stage1_preds",
        exist_ok=True
    )
    
    detections = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() 
        confs = r.boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confs):
            xmin, ymin, xmax, ymax = box
            detections.append({
                "bbox_pixels": [float(xmin), float(ymin), float(xmax), float(ymax)],
                "confidence": float(conf)
            })
            
    print(f"Detected {len(detections)} potential slick(s) in {image_path}")
    return detections

if __name__ == "__main__":
    weights = "D:/MINE/Projects/SIH/runs/detect/oil_spill_detection/stage1_yolo/weights/best.pt"
    test_image = "D:/MINE/Projects/SIH/data/yolo_dataset/images/val/ow-0009.jpg"
    
    if Path(test_image).exists() and Path(weights).exists():
        preds = run_stage1_inference(weights, test_image)
        print("Detections:", preds)
    else:
        print("Please verify the paths to best.pt and your test image.")