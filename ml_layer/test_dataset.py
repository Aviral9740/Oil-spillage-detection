import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import OilSpillDataset

def validate_dataset(dataset_path):
    dataset = OilSpillDataset(root_dir=dataset_path)
    print(f"Total image patches loaded: {len(dataset)}")

    for idx in range(len(dataset)):
        img_tensor, target = dataset[idx]
        
        if len(target["boxes"]) > 0:
            print(f"\n--- Success: Oil Slick Found at Index {idx} ---")
            print(f"Tensor Shape: {img_tensor.shape}")
            print(f"Bounding Boxes: \n{target['boxes']}")
            print(f"Labels: {target['labels']}")
            
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(img_np)
            
            for box in target["boxes"]:
                xmin, ymin, xmax, ymax = box.tolist()
                width = xmax - xmin
                height = ymax - ymin
                
                rect = patches.Rectangle(
                    (xmin, ymin), width, height, 
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
            
            plt.title("Normalized SAR Image with Oil Slick Bounding Box")
            plt.axis('off')
            plt.show()
            break  

if __name__ == "__main__":
    validate_dataset("D:/MINE/Projects/SIH")