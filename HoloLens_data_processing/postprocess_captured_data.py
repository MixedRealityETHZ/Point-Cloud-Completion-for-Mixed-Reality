import os
import numpy as np
import open3d as o3d
import shutil

def copy_metadata(path, save_path):
    save_path_meta = os.path.join(save_path, 'meta')
    os.makedirs(save_path_meta, exist_ok=True)
    save_path_pose = os.path.join(save_path, 'pose')
    os.makedirs(save_path_pose, exist_ok=True)
    file_path = os.path.join(path + '_segmented', 'standardized')
    for metafile in os.listdir(file_path):
        if metafile.startswith('meta_'):
            shutil.copy(os.path.join(file_path, metafile), os.path.join(save_path_meta, metafile))
        if metafile.startswith('object_pose'):
            shutil.copy(os.path.join(file_path, metafile), os.path.join(save_path_pose, metafile))

def transform_back(path, save_path):
    prediction_path = path + '_prediction'
    save_path_ply = os.path.join(save_path, 'ply')
    os.makedirs(save_path_ply, exist_ok=True)
    for frame in os.listdir(prediction_path):
        if frame in '.DS_Store':
            continue
        # get point cloud
        file_path = os.path.join(prediction_path, frame, 'fine.npy')
        points = np.load(file_path)
        # get transformation
        transform_path = os.path.join(path + '_segmented', 'standardized', frame +'_standardize.npy')
        transform = np.load(transform_path, allow_pickle=True).item()
        shift = transform['mean']
        scale = transform['scale']
        # transform back
        points_transformed = points * scale + shift
        points_transformed[:, 2] *= -1

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_transformed)
        # save as .ply
        o3d.io.write_point_cloud(os.path.join(save_path_ply, frame + '.ply'), point_cloud, write_ascii=True)

if __name__ == '__main__':
    base_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/HoloLens'
    recording = 'PRINTER_TUTORIAL5'
    path = os.path.join(base_path, recording)
    save_path = path + '_final'
    os.makedirs(save_path, exist_ok=True)
    transform_back(path, save_path)
    copy_metadata(path, save_path)
