import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import cv2
import torch
from dataset import OilSpillDataset

def convert_to_yolo_format(box, img_width, img_height):
    """Converts [xmin, ymin, xmax, ymax] to YOLO [x_center, y_center, width, height] normalized."""
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return [x_center, y_center, width, height]

def build_yolo_dataset(raw_dataset_dir, metadata_path, output_dir="data/yolo_dataset"):
    out_path = Path(output_dir)
    for split in ['train', 'val']:
        (out_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 1. Read metadata and identify correct columns (PANGAEA headers can be messy)
    df = pd.read_excel(metadata_path)
    
    # Find the column containing the .jpg filenames and the Sentinel IDs
    img_col = next((col for col in df.columns if 'jpg' in str(col).lower() or 'IMAGE' in str(col)), None)
    id_col = next((col for col in df.columns if 'Sentinel_ID' in str(col)), None)
    
    if not img_col or not id_col:
        raise ValueError(f"Could not find required columns in excel file. Columns found: {df.columns.tolist()}")

    # 2. Generate Leakage-Free Splits using Sentinel_ID
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df[id_col]))
    
    # Store the actual 'ow-0001.jpg' filenames in sets for fast lookup
    train_patches = set(df.iloc[train_idx][img_col].astype(str))
    val_patches = set(df.iloc[val_idx][img_col].astype(str))

    # 3. Iterate through our PyTorch Dataset to apply physics normalization
    dataset = OilSpillDataset(root_dir=raw_dataset_dir)
    
    train_count, val_count = 0, 0
    
    for idx in range(len(dataset)):
        img_path = dataset.image_paths[idx]
        filename = img_path.name
        
        if filename in val_patches:
            split = 'val'
            val_count += 1
        else:
            split = 'train'
            train_count += 1
        img_tensor, target = dataset[idx]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        img_height, img_width, _ = img_np.shape
        cv2.imwrite(str(out_path / 'images' / split / filename), img_np)
        label_filename = img_path.with_suffix('.txt').name
        label_path = out_path / 'labels' / split / label_filename
        
        with open(label_path, 'w') as f:
            if len(target["boxes"]) > 0:
                for box in target["boxes"].tolist():
                    yolo_box = convert_to_yolo_format(box, img_width, img_height)
                    line = f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
                    f.write(line)

    print(f"YOLO dataset generated! Train images: {train_count}, Val images: {val_count}")
    if val_count == 0:
        print("WARNING: Validation set is still empty. Check your data_table.xlsx format.")

if __name__ == "__main__":
    build_yolo_dataset(
        raw_dataset_dir="D:/MINE/Projects/SIH/data", 
        metadata_path="D:/MINE/Projects/SIH/data/data_table.xlsx",
        output_dir="D:/MINE/Projects/SIH/data/yolo_dataset"
    )