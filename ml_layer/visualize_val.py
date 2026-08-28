from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def visualize_validation_predictions(weights_path, val_images_dir, num_samples=5, conf_thresh=0.25):
    model = YOLO(weights_path)
    val_dir = Path(val_images_dir)
    image_paths = list(val_dir.glob("*.jpg"))[:num_samples]
    
    if not image_paths:
        print(f"No .jpg images found in {val_images_dir}")
        return

    for img_path in image_paths:
        results = model.predict(source=str(img_path), conf=conf_thresh, verbose=False)
        annotated_frame = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(annotated_rgb)
        plt.title(f"Predictions for: {img_path.name} (Detections: {len(results[0].boxes)})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "oil_spill_detection" / "stage1_yolo" / "weights" / "best.pt"
    VAL_DIR = PROJECT_ROOT / "data" / "oil" / "water" 
    
    visualize_validation_predictions(
        weights_path=str(WEIGHTS),
        val_images_dir=str(VAL_DIR),
        num_samples=5,
        conf_thresh=0.25
    )