from scene.transmvsnet import TransMVSNet
import torch
import os, sys
from scene.dataset_readers import readNerfSyntheticInfo
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos
from argparse import ArgumentParser
import numpy as np
import math
import cv2
from tqdm import tqdm
import torchvision

def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color

def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def build_pairs(data_path):
    pair_file = "{}/mvs_pairs.txt".format(data_path)
    # read the pair file
    with open(os.path.join(pair_file)) as f:
        lines = f.readlines()
        lines = [line.rstrip().split()[0::2] for line in lines]
    lines = [[int(x) for x in line] for line in lines]
    return lines 

def build_depths(data_path):
    filename = "{}/depth_ranges.txt".format(data_path)
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip().split(" ") for line in lines]
    lines = [[float(x) for x in line] for line in lines]
    return lines

def getprojmat(train_camera, scale=4.0):
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
    proj_mat[1, :2, :] /= scale # https://github.com/megvii-research/TransMVSNet/blob/master/datasets/bld_train.py#L61
    return proj_mat

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    # add output_dir to parser
    lp = ModelParams(parser)

    data_dir = "/home/nas4_dataset/3D/NeRF/NeRF_Data/nerf_synthetic"
    scenes = ["lego", "drums", "hotdog", "chair", "materials", "mic", "ship"]
    # scenes = ["lego"]

    # # Load TransMVS model
    model = TransMVSNet(refine=False,
                        ndepths=[48, 32, 8], 
                        depth_interals_ratio=[4, 1, 0.5],
                        share_cr=False,
                        cr_base_chs=[8, 8, 8],
                        grad_method="detach")
    # state_dict = torch.load("checkpoints/TransMVSNet/model_dtu.ckpt", map_location=torch.device("cpu"))
    state_dict = torch.load("checkpoints/TransMVSNet/model_bld.ckpt", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model.to("cuda")
    model.eval()

    for scene in scenes:
        n_views = 5 
        ndepths = 192
        outdir = os.path.join(data_dir, scene, "mvs_outputs")
        os.makedirs(outdir, exist_ok=True)

        ## Read Data
        lp.resolution = 1
        scene_info = readNerfSyntheticInfo(os.path.join(data_dir, scene), white_background=False, eval=True, point_random_init=False)
        train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, lp)
        test_cameras = cameraList_from_camInfos(scene_info.test_cameras, 1.0, lp)

        pair_metas = build_pairs(os.path.join(data_dir, scene)) 
        depth_metas = build_depths(os.path.join(data_dir, scene)) # depth_min, depth_interval, depth_num, depth_max 

        pbar = tqdm(enumerate(train_cameras), total=len(train_cameras))
        pbar.set_description(f"[ Processing {scene}... ]")
        for i, train_camera in pbar:
            pairs = pair_metas[i][:n_views-1]
            images = [train_camera.original_image] + [train_cameras[p].original_image for p in pairs]
            images = torch.stack(images)[None, ...]

            # Depth range for reference view
            depth_meta = depth_metas[i]
            depth_max = depth_meta[-1]
            depth_min = depth_meta[0]
            depth_interval = float(depth_max - depth_min) / ndepths
            depth_values = torch.from_numpy(np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)).to("cuda")[None, ...]

            # Camera projection matrices
            proj_matrices = np.stack([getprojmat(train_camera)] + [getprojmat(train_cameras[p]) for p in pairs])
            proj_matrices = np.stack(proj_matrices)
            stage2_pjmats = proj_matrices.copy()
            stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
            stage3_pjmats = proj_matrices.copy()
            stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

            proj_matrices_ms = {
                "stage1": torch.from_numpy(proj_matrices).to("cuda")[None, ...],
                "stage2": torch.from_numpy(stage2_pjmats).to("cuda")[None, ...],
                "stage3": torch.from_numpy(stage3_pjmats).to("cuda")[None, ...]
            }

            outputs = model(images, proj_matrices_ms, depth_values)
            outputs = tensor2numpy(outputs)

            for depth_est, photometric_confidence, conf_1, conf_2 in zip(outputs["depth"], outputs["photometric_confidence"],  outputs['stage1']["photometric_confidence"], outputs['stage2']["photometric_confidence"]):
                filename = f"view-{i}" + "_{}{}" 
                H,W = photometric_confidence.shape
                conf_1 = cv2.resize(conf_1, (W,H))
                conf_2 = cv2.resize(conf_2, (W,H))
                conf_final = photometric_confidence * conf_1 * conf_2

                depth_filename = os.path.join(outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(outdir, filename.format('images', '.jpg'))
                ply_filename = os.path.join(outdir, filename.format('ply_local', '.ply'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                
                #save depth maps
                save_pfm(depth_filename, depth_est)
                depth_color = visualize_depth(depth_est)
                cv2.imwrite(os.path.join(outdir, filename.format('depth_est', '.png')), depth_color)
                
                #save confidence maps
                save_pfm(confidence_filename, conf_final)
                cv2.imwrite(os.path.join(outdir, filename.format('confidence', '_3.png')), visualize_depth(photometric_confidence))
                cv2.imwrite(os.path.join(outdir, filename.format('confidence', '_1.png')),visualize_depth(conf_1))
                cv2.imwrite(os.path.join(outdir, filename.format('confidence', '_2.png')),visualize_depth(conf_2))
                cv2.imwrite(os.path.join(outdir, filename.format('confidence', '_final.png')),visualize_depth(conf_final))

                # save camera
                write_cam(cam_filename, getprojmat(train_camera, scale=1.0))

                # save image
                torchvision.utils.save_image(images[0][0], img_filename)