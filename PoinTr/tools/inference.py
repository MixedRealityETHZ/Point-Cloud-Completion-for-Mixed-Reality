##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
print("import started")
import argparse
import os
import numpy as np
import cv2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
from tqdm import tqdm

from builder import *
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import os

print("import finished")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_config', 
        help = 'yaml config file')
    parser.add_argument(
        'model_checkpoint', 
        help = 'pretrained weight')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')   
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud') 
    parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args

def inference_single(model, pc_path, args, config, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path
    # read single point cloud
    pc_ndarray = IO.get(pc_file).astype(np.float32)
    # transform it according to the model 
    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # normalize it to fit the model on ShapeNet-55/34
        print("input data is normalized")
        centroid = np.mean(pc_ndarray, axis=0)
        pc_ndarray = pc_ndarray - centroid
        m = np.max(np.sqrt(np.sum(pc_ndarray**2, axis=1)))
        pc_ndarray = pc_ndarray / m

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    
    pc_ndarray_normalized = transform({'input': pc_ndarray})
    # inference
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()

    if config.dataset.train._base_['NAME'] == 'ShapeNet':
        # denormalize it to adapt for the original input
        dense_points = dense_points * m
        dense_points = dense_points + centroid

    if args.out_pc_root != '':
        target_path = os.path.join(args.out_pc_root, os.path.splitext(pc_path)[0])
        os.makedirs(target_path, exist_ok=True)

        np.save(os.path.join(target_path, 'fine.npy'), dense_points)
        if args.save_vis_img:
            input_img = misc.get_ptcloud_img(pc_ndarray_normalized['input'].numpy())
            print("debug simon", dense_points.shape, dense_points.dtype)
            dense_img = misc.get_ptcloud_img(dense_points)
            cv2.imwrite(os.path.join(target_path, 'input.jpg'), input_img)
            cv2.imwrite(os.path.join(target_path, 'fine.jpg'), dense_img)
    
    # correct chamfer calculation
    # get ground truth point cloud
    id = pc_path.split('/')[-1].split('.')[0].split('_')[0]
    gt_root = './data/PCN/test/complete/10'
    gt_path = os.path.join(gt_root, id + '.pcd')
    gt_points_np = IO.get(gt_path).astype(np.float32)
    gt_points = torch.from_numpy(gt_points_np)
    pred_points = ret[-1]
    
    calc_chamfer_l1 = ChamferDistanceL1()
    chamfer_dist_l1 = calc_chamfer_l1(pred_points, gt_points.unsqueeze(0).to(args.device.lower()))
    # save it to txt file
    with open(os.path.join(target_path, 'chamfer_dist_l1.txt'), 'w') as f:
        f.write(str(chamfer_dist_l1.item()*1000))
    
    calc_chamfer_l2 = ChamferDistanceL2()
    chamfer_dist_l2 = calc_chamfer_l2(pred_points, gt_points.unsqueeze(0).to(args.device.lower()))
    # save it to txt file
    with open(os.path.join(target_path, 'chamfer_dist_l2.txt'), 'w') as f:
        f.write(str(chamfer_dist_l2.item()*1000))

    return

def main():
    print("inference started")
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = model_builder(config.model)
    load_model(base_model, args.model_checkpoint)
    base_model.to(args.device.lower())
    base_model.eval()

    if args.pc_root != '':
        pc_file_list = os.listdir(args.pc_root)
        for pc_file in tqdm(pc_file_list):
            inference_single(base_model, pc_file, args, config, root=args.pc_root)
    else:
        inference_single(base_model, args.pc, args, config)

if __name__ == '__main__':
    main()