from ultralytics import YOLO

def train_stage_one():
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="D:/MINE/Projects/SIH/ml_layer/oil_spill.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="cpu",
        patience=15,          # Early stopping
        save=True,
        project="oil_spill_detection",
        name="stage1_yolo",
        exist_ok=True,
        cls=1.0,              # Classification loss weight
        box=7.5,              # Box loss weight (CIoU)
        dfl=1.5               # Distribution Focal Loss weight
    )
    
    print("Stage 1 YOLO training complete. Weights saved to oil_spill_detection/stage1_yolo/weights/best.pt")

if __name__ == "__main__":
    train_stage_one()