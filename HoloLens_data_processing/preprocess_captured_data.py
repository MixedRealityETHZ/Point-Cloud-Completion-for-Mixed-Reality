import numpy as np
import os
import open3d as o3d
import shutil
from copy import copy

def readMetadata(file):
    M = np.zeros((4, 4))
    with open(file) as f:
        timeStamp = int(f.readline())
        for i in range(4):
            M[i, :] = np.array(list(map(lambda x: float(x),f.readline().strip().split(", "))))
    M = M.T
    # M[:3, 3] *= 1000
    M = np.linalg.inv(M)
    return M

def normalize_pcd(pcd):

    pcd_array = np.array(pcd.points)

    # centralize data
    pcd_centralized = pcd_array - np.mean(pcd_array, axis=0)

    # normalize data
    # m = np.max(np.sqrt(np.sum(pcd_centralized**2, axis=1)))
    m = 0.1057
    pcd_normalized = pcd_centralized / m

    pcd.points = o3d.utility.Vector3dVector(pcd_normalized)
    return pcd, np.mean(pcd_array, axis=0), m


def visualize_depth(ply_file, M, base_path):
    print("frame", frame)
    # read .ply file
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.array(pcd.points) # * 1000
    points[:, 2] *= -1
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_transformed = copy(pcd)
    pcd_transformed.transform(M)


    # origin = np.array([0, 0, 0, 1])
    # add a coordinate frame
    mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd_transformed, mesh_origin])
    # remove points around the origin from pcd with threshold
    points = np.asarray(pcd_transformed.points)
    dist = np.linalg.norm(points, axis=1)
    mean_pc = np.mean(points, axis=0)
    threshold = abs(mean_pc[2]) # 0.45
    # create a mask with dist > threshold - 0.35 and dist < threshold + 0.35
    mask = np.logical_and(dist > threshold - 0.15, dist < threshold + 0.15)
    # mask_smaller = dist > threshold - 0.35
    # keep only points where mask_smaller is True for z axis points
    pcd = pcd.select_by_index(np.where(mask)[0])
    # o3d.visualization.draw_geometries([pcd, mesh_origin])
    # mask_bigger = dist < threshold + 0.35
    # pcd = pcd.select_by_index(np.where(mask_bigger)[0])


    pcd, ind_stat = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.8)
    # o3d.visualization.draw_geometries([pcd, mesh_origin])

    pcd_normalized, normalize_mean, normalize_scale = normalize_pcd(pcd)

    pcd_normalized_points = np.array(pcd_normalized.points)
    np.random.shuffle(pcd_normalized_points)
    pcd_normalized_points = pcd_normalized_points[0:2048]
    # subsample point cloud
    pcd_sampled = o3d.geometry.PointCloud()
    pcd_sampled.points = o3d.utility.Vector3dVector(pcd_normalized_points)

    # save pcd as .pcd file
    save_path = os.path.join(base_path, 'standardized')
    os.makedirs(os.path.join(save_path, 'pcd'), exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(save_path, 'pcd', frame.split('.')[0] + '.pcd'), pcd_sampled, write_ascii=True)

    standardize_json = {
        'mean': normalize_mean.tolist(),
        'scale': normalize_scale
    }
    np.save(os.path.join(save_path, frame.split('.')[0] + '_standardize.npy'), standardize_json)


if __name__ == "__main__":
    base_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/HoloLens/PRINTER_TUTORIAL5_segmented/'
    segmentation_path = os.path.join(base_path, 'segmented')
    save_path = os.path.join(base_path, 'standardized')
    for frame in os.listdir(segmentation_path):
        if 'DS_Store' in frame:
            continue
        if frame.endswith('.txt'):
            #copy file to save_path location
            os.makedirs(save_path, exist_ok=True)
            shutil.copy(os.path.join(segmentation_path, frame), os.path.join(save_path, frame))
            continue
        # frame = '000290'
        file = os.path.join(segmentation_path, 'meta_' + frame.split('.')[0] + '.txt')
        M = readMetadata(file)
        ply_file = os.path.join(segmentation_path, frame)
        visualize_depth(ply_file, M, base_path)
