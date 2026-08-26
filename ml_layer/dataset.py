import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class OilSpillDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = Path(root_dir)
        self.transforms = transforms

        self.image_paths = list(self.root_dir.rglob('*.jpg'))

    def apply_sigmoid_normalization(self, image_array):
        img_float = image_array.astype(np.float32)
        beta = np.median(img_float)
        alpha = 3 * np.std(img_float)
        
        
        if alpha == 0:
            return image_array
            
        normalized = 255 / (1 + np.exp(-(img_float - beta) / alpha))
        return np.clip(normalized, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = img_path.name
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = self.apply_sigmoid_normalization(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        boxes = []
        labels = []
        is_oil = filename.startswith('oc') or filename.startswith('ow')
        
        if is_oil:
            xml_path = img_path.with_suffix('.xml')
            
            if xml_path.exists():
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)  # Class 1: Oil Spill

        target = {}
        if boxes:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.int64)
            
        if self.transforms:
            transformed = self.transforms(image=img, bboxes=target["boxes"], class_labels=target["labels"])
            img = transformed['image']
            target["boxes"] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)

        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
        return img, target

    def __len__(self):
        return len(self.image_paths)