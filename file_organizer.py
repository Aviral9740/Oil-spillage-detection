import shutil
from pathlib import Path

def organize_dataset(source_folder, target_folder):
    source_dir = Path(source_folder)
    target_dir = Path(target_folder)
    subdirs = [
        "oil/coast", 
        "oil/water"
    ]
    subdirs += [f"no_oil/coast/c{i:02d}" for i in range(5)]
    subdirs += [f"no_oil/water/c{i:02d}" for i in range(12)]
    
    for d in subdirs:
        (target_dir / d).mkdir(parents=True, exist_ok=True)
    for file_path in source_dir.iterdir():
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        prefix = filename[:2]
        
        if prefix == 'oc':
            dest = target_dir / "oil" / "coast" / filename
        elif prefix == 'ow':
            dest = target_dir / "oil" / "water" / filename
        elif prefix == 'nc':
            cluster_idx = filename.split('-')[2]
            dest = target_dir / "no_oil" / "coast" / f"c{cluster_idx}" / filename
        elif prefix == 'nw':
            cluster_idx = filename.split('-')[2]
            dest = target_dir / "no_oil" / "water" / f"c{cluster_idx}" / filename
        elif filename in ['data_table.xlsx', 'DARTIS_2019.tab']:
            dest = target_dir / filename
        else:
            continue
            
        shutil.move(str(file_path), str(dest))
        
    print("Files successfully organized into the target directory.")
organize_dataset(
    source_folder="D:/MINE/Projects/SIH", 
    target_folder="D:/MINE/Projects/SIH/data"
)