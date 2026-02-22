import os
import numpy as np
import torch
import cv2
from pathlib import Path
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)
from gaussian_splatting.SIBR_viewers.src.projects.dataset_tools.preprocess.fullColmapProcess.read_write_model import read_model, write_model

MODEL_ROOT = "/home/hayrap/repos/msh/mesh-splatting/scene/15FPS_iron_no_mask"
SPARSE_ROOT = os.path.join(MODEL_ROOT, "sparse","0")
MASK_DIR = os.path.join(MODEL_ROOT, "masks")
OUTPUT_ROOT = os.path.join("/home/hayrap/repos/gl_image_to_3d/external/SuGaR/utils/15FPS_iron_no_mask", "sparse", "0_filtered")

def load_mask(mask_path):    
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found for image {os.path.basename(mask_path)} at path {mask_path}")
    return mask
def filter_point(point, images, mask_dir, threshold_number_of_appearecne_in_mask):
    number_of_point_appearances_in_mask = 0
    for img_id, kp_idx in zip(point.image_ids, point.point2D_idxs):
        img = images[img_id]
        xy = img.xys[kp_idx]   # (x, y) pixel coordinate
        mask = load_mask(os.path.join(mask_dir, img.name.split(".")[0] + ".png"))
        x, y = int(round(xy[0])), int(round(xy[1]))
        h, w = mask.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            # skip observations outside mask bounds
            continue
        if mask[y, x] > 128:  # Assuming mask is binary with values
            number_of_point_appearances_in_mask += 1
            if number_of_point_appearances_in_mask>=threshold_number_of_appearecne_in_mask:
                break
    return number_of_point_appearances_in_mask >= threshold_number_of_appearecne_in_mask
def main():
    filtered = {}

    pool = Pool(processes=cpu_count())


    cameras, images, points3D = read_model(SPARSE_ROOT, ext=".bin")
    threshold_number_of_appearecne_in_mask = 10
    filtered_points = pool.starmap(filter_point, [(point, images, MASK_DIR,threshold_number_of_appearecne_in_mask) for point in points3D.values()])

    for point_id, point in zip(points3D.keys(), points3D.values()):
        if not filtered_points.pop(0):
            continue
        filtered[point_id] = point
    output_path =  OUTPUT_ROOT+f"_thr{threshold_number_of_appearecne_in_mask}"
    os.makedirs(output_path,exist_ok=True)
    write_model(cameras, images, filtered, output_path, ext=".bin")

            
if __name__ == "__main__":
    main()

