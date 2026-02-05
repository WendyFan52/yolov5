import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_coco(json_path, save_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(save_dir, exist_ok=True)
    # Establish an ID mapping to ensure a continuous sequence starting from 0.
    cat_map = {cat['id']: i for i, cat in enumerate(data['categories'])}
    images = {img['id']: img for img in data['images']}

    for ann in tqdm(data['annotations'], desc="Converting"):
        if 'segmentation' not in ann or ann['iscrowd']: continue
        img = images[ann['image_id']]
        w, h = img['width'], img['height']
        
        # Normalised coordinate points (x/w, y/h)
        for seg in ann['segmentation']:
            if len(seg) < 6: continue
            norm_points = [str(v/w if i%2==0 else v/h) for i, v in enumerate(seg)]
            line = f"{cat_map[ann['category_id']]} {' '.join(norm_points)}\n"
            
            with open(os.path.join(save_dir, Path(img['file_name']).stem + ".txt"), 'a') as f_out:
                f_out.write(line)

# convert_coco('/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/annotations/instances_tree_train2023.json', '/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/annotations/train2017')
# convert_coco('/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/annotations/instances_tree_eval2023.json', '/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/annotations/val2017')
convert_coco('/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/annotations/instances_tree_TestSet12023.json', '/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/annotations/test2017')
convert_coco('/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/annotations/instances_tree_TestSet22023.json', '/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/annotations/test2017')