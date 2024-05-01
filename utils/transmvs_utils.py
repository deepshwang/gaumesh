import os
import numpy as np
from utils.graphics_utils import fov2focal 

def read_pairs(data_path):
    pair_file = "{}/mvs_pairs.txt".format(data_path)
    # read the pair file
    all_src_views = []
    with open(os.path.join(pair_file)) as f:
        lines = f.readlines()
        lines = [line.rstrip().split()[0::2] for line in lines]
    lines = [[int(x) for x in line] for line in lines]
    return lines 

def read_depths(data_path):
    filename = "{}/depth_ranges.txt".format(data_path)
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip().split(" ") for line in lines]
    lines = [[float(x) for x in line] for line in lines]
    return lines

def getprojmat(train_camera):
    proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
    proj_mat[0, :3, :3] = train_camera.R.T
    proj_mat[0, :3, 3] = train_camera.T
    proj_mat[0, 3, 3] = 1.0
    proj_mat[1, 0, 2] = train_camera.image_width / 2
    proj_mat[1, 1, 2] = train_camera.image_height / 2
    f = fov2focal(train_camera.FoVx, train_camera.image_width)
    proj_mat[1, 0, 0] = f
    proj_mat[1, 1, 1] = f
    proj_mat[1, 2, 2] = 1.0
    proj_mat[1, :2, :] /= 4.0 # https://github.com/megvii-research/TransMVSNet/blob/master/datasets/bld_train.py#L61
    return proj_mat
