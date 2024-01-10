from scipy.spatial import KDTree
import open3d as o3d
import numpy as np
from tqdm import tqdm
import os

def calc_pck(pc_pred, pc_target, radius):
    # Assuming points_cloud_a and points_cloud_b are NumPy arrays containing the point coordinates
    target_tree = KDTree(pc_target)
    # make an index set
    close_points_set = set()
    counter_close_points = 0
    for idx, point_a in enumerate(pc_pred):
        
        close_points_indices = target_tree.query_ball_point(point_a, r=radius)
        # convert it to a set
        close_points_cur = pc_target[close_points_indices]
        if len(close_points_indices) > 0:
            counter_close_points += 1
            close_points_set.update([idx]) 

    pck = counter_close_points/len(pc_pred)
    # print("PCK", counter_close_points, len(pc_pred), pck)
    # return pck

    # VISUALIZATION
    # convert set to list
    # close_points_indices = list(close_points_set)
    # close_points = pc_pred[close_points_indices]
    # far_points = np.delete(pc_pred, close_points_indices, axis=0)
    # pcd_pred = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(far_points))
    # pcd_target = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_target))
    # pcd_close_points = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(close_points))

    # pcd_pred.paint_uniform_color([1, 0, 0])
    # pcd_target.paint_uniform_color([0, 1, 0])
    # pcd_close_points.paint_uniform_color([0, 0, 1])


    # o3d.visualization.draw_geometries([pcd_pred, pcd_close_points, pcd_target])

    return pck



if __name__ == "__main__":
    base_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/predictions/inference_freihand_test_all'

    for radius in [0.2]:
    # for radius in [0.01, 0.025, 0.05, 0.075, 0.1, 0.2]:
        pck_list = []
        id_list = []
        for id in tqdm(os.listdir(base_path), desc=f'radius: {radius}'):
            if "pck" in id:
                continue
            file_path = os.path.join(base_path, id)
            pred_path = os.path.join(file_path, 'fine.npy')
            target_path = os.path.join(file_path, 'gt.npy')
            # pc_pred = o3d.io.read_point_cloud(pred_path)
            # pc_target = o3d.io.read_point_cloud(target)

            # pc_pred_array = np.array(pc_pred.points)
            # pc_target_array = np.array(pc_target.points)

            pc_pred_array = np.load(pred_path)
            pc_target_array = np.load(target_path)
            
            pck = calc_pck(pc_pred_array, pc_target_array, radius)
            pck_list.append(pck)
            id_list.append(id)

        # combine pck and id list
        pck_list = np.array(pck_list)
        id_list = np.array(id_list)
        pck_array = np.vstack((pck_list, id_list))

            
        # save pck list
        np.save(os.path.join(base_path, f'pck_{radius}.npy'), pck_array)