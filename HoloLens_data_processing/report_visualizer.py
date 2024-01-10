import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

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

    # VISUALIZATION
    # convert set to list
    close_points_indices = list(close_points_set)
    close_points = pc_pred[close_points_indices]
    far_points = np.delete(pc_pred, close_points_indices, axis=0)
    pcd_far = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(far_points))
    pcd_target = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_target))
    pcd_close_points = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(close_points))

    pcd_far.paint_uniform_color([0, 0, 1])
    # make target violet
    pcd_target.paint_uniform_color([1, 0, 1])
    pcd_close_points.paint_uniform_color([0, 0, 1])

    # Visualize the point clouds and spheres
    render_option = o3d.visualization.RenderOption()
    render_option.point_size = 5  # Adjust the point size according to your preference

    # Visualize the point clouds with the specified rendering options
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis.add_geometry(pcd_far)
    vis.add_geometry(pcd_target)
    # add render option
    vis.get_render_option().point_size = 8
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # load .npy file
    path_partial = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/HoloLens/Recording_3_12_2023_prediction/000048/input.npy'
    partial_np = np.load(path_partial)
    pcd_partial = o3d.geometry.PointCloud()
    pcd_partial.points = o3d.utility.Vector3dVector(partial_np)

    path_complete = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/HoloLens/Recording_3_12_2023_prediction/000048/fine.npy'
    complete_np = np.load(path_complete)
    pcd_complete = o3d.geometry.PointCloud()
    pcd_complete.points = o3d.utility.Vector3dVector(complete_np)

    calc_pck(complete_np, partial_np, 0.03)


    # pcd_partial.paint_uniform_color([0.5, 0.5, 0.5])
    # pcd_complete.paint_uniform_color([0, 0, 1])
    # o3d.visualization.draw_geometries([pcd_partial, pcd_complete])