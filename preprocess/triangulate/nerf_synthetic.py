import numpy as np 
import pathlib
import argparse
from glob import glob
import os
import shutil
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import json
import math

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="/home/nas4_dataset/3D/NeRF/NeRF_Data/nerf_synthetic")
    args = args.parse_args()
    scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"] 

    
    # Get a list of frame segments of static scenes
    for scene in scenes:
        data_dir = pathlib.Path(args.data_dir)
        data_dir = data_dir / scene
        cache_dir = f".cache/colmap_ws/{scene}"
        cache_img_dir = os.path.join(cache_dir, "images")
        os.makedirs(cache_img_dir, exist_ok=True)
        save_idx = 1
        
        # save images.txt
        with open(os.path.join(data_dir, "transforms_train.json")) as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]
            
            frames = contents["frames"]
            for idx, frame in tqdm(enumerate(frames), total=len(frames)):
                img = os.path.join(data_dir, frame["file_path"][2:] + ".png")
                # save image file to cache
                save_img_name = f"{str(save_idx).zfill(6)}.png"
                shutil.copyfile(img, os.path.join(cache_img_dir, save_img_name))

                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1
                w2c = np.linalg.inv(c2w)
                R = rotmat2qvec(w2c[:3, :3])
                T = w2c[:3, 3].squeeze()
                f = open(os.path.join(cache_dir, "images.txt"), "a")
                f.write(f"{save_idx} {R[0]} {R[1]} {R[2]} {R[3]} {T[0]} {T[1]} {T[2]} 1 {save_img_name}\n")
                f.write("\n")
                f.close()
                
                save_idx += 1
               

        # save cameras.txt
        f = open(os.path.join(cache_dir, "cameras.txt"), "a")
        img = Image.open(img)
        w, h = img.size
        focal_x = fov2focal(fovx, w)
        f.write(f"1 PINHOLE {w} {h} {focal_x} {focal_x} {w//2} {h//2}\n")
        f.close()

        # # create a blank points3d.txt file
        file = open(os.path.join(cache_dir, "points3D.txt"), "w")
        file.close() 

        # # Start COLMAP
        print("Starting COLMAP..")
        database_path = os.path.join(cache_dir, "database.db")
        
        # Feature extraction            
        os.system(f"colmap feature_extractor --database_path {database_path} --image_path {cache_img_dir}")

        # Feature matching
        os.system(f"colmap exhaustive_matcher --database_path {database_path}")

        # Triangulation with known camera parameters
        dest_dir = str(data_dir)
        os.system(f"colmap point_triangulator --database_path {database_path} --image_path {cache_img_dir} --input_path {cache_dir} --output_path {dest_dir} --clear_points 1")
        # Convert to PLY 
        os.system(f"colmap model_converter --input_path {dest_dir} --output_path {dest_dir} --output_type TXT")
        os.system(f"colmap model_converter --input_path {dest_dir} --output_path {dest_dir}/points3D.ply --output_type PLY")


        # remove cache
        shutil.rmtree(cache_dir)

        # Remove unnecessary file in dest_dir
        # os.remove(os.path.join(dest_dir, "cameras.txt"))
        # os.remove(os.path.join(dest_dir, "images.txt"))
        # os.remove(os.path.join(dest_dir, "points3D.txt"))
        # os.remove(os.path.join(dest_dir, "cameras.bin"))
        # os.remove(os.path.join(dest_dir, "images.bin"))
        # os.remove(os.path.join(dest_dir, "points3D.bin"))

        
