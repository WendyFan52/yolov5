# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import cv2
from osgeo import gdal, ogr, osr

# Import the necessary components for YOLO
sys.path.append(os.getcwd())
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes as scale_coords
from utils.segment.general import process_mask
from utils.augmentations import letterbox

# ==================== 1. Configuration path ====================
WEIGHTS_PATH = 'runs/train-seg/exp/weights/best.pt'
IMG_DIR = '/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/test2023/testimg'
OUTPUT_DIR = '/dss/dsstbyfs02/pn49cu/pn49cu-dss-0011/data/Bamberg/BambergDataset/coco2048/predict_output_shp_test'
CONF_THRES = 0.5  # Confidence threshold
IOU_THRES = 0.5  # NMS IOU 
CLASS_NAMES = ['tree'] # 
# =====================================================================

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the model
    model = DetectMultiBackend(WEIGHTS_PATH, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)  

    # 2. Retrieve images
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.tif', '.tiff'))]
    driver = ogr.GetDriverByName("ESRI Shapefile")

    model.warmup(imgsz=(1, 3, *imgsz))  
    print(f"开始推理，找到 {len(img_files)} 个文件...")

    for img_file in img_files:
        img_path = os.path.join(IMG_DIR, img_file)
        pure_name = os.path.splitext(img_file)[0]
        
        ds = gdal.Open(img_path)
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection)

        im0 = cv2.imread(img_path) # BGR
        img = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        if len(img.shape) == 3: img = img[None]

        # 3. infer
        pred, proto = model(img, augment=False, visualize=False)[:2]

        # 4. NMS
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, nm=32)

        # 5. create Shapefile
        img_out_dir = os.path.join(OUTPUT_DIR, pure_name)
        os.makedirs(img_out_dir, exist_ok=True)
        shp_path = os.path.join(img_out_dir, f"{pure_name}_trees.shp")
        if os.path.exists(shp_path): driver.DeleteDataSource(shp_path)
        
        out_ds = driver.CreateDataSource(shp_path)
        out_layer = out_ds.CreateLayer("trees", srs, ogr.wkbPolygon)
        out_layer.CreateField(ogr.FieldDefn("Conf", ogr.OFTReal))

        # 6. result
        for i, det in enumerate(pred):
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)
                
                # Map the coordinates back to the original image dimensions
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                    mask = (masks[j] > 0.5).cpu().numpy().astype(np.uint8)
                    # Adjust the mask to the original image size
                    mask = cv2.resize(mask, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    poly = ogr.Geometry(ogr.wkbPolygon)
                    for contour in contours:
                        if len(contour) < 3: continue
                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        for pt_cv in contour:
                            px, py = pt_cv[0]
                            # Geographical Coordinate Conversion
                            gx = geo_transform[0] + px * geo_transform[1] + py * geo_transform[2]
                            gy = geo_transform[3] + px * geo_transform[4] + py * geo_transform[5]
                            ring.AddPoint(gx, gy)
                        ring.CloseRings()
                        poly.AddGeometry(ring)

                    if not poly.IsEmpty():
                        feat = ogr.Feature(out_layer.GetLayerDefn())
                        feat.SetGeometry(poly)
                        feat.SetField("Conf", float(conf))
                        out_layer.CreateFeature(feat)
        
        out_ds.FlushCache()
        print(f"Finish: {img_file}")

if __name__ == "__main__":
    main()