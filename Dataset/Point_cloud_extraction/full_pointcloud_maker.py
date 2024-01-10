from extraction import Depth_To_Pointcloud
from sklearn.ensemble import IsolationForest
import sys
import numpy as np
import io,os
import open3d as o3d
import pickle
import tqdm
from tqdm import tqdm

"""
Returns the full pointcloud (merged together)
L. Schüepp
"""

# We need toolkits for the mano data (stems from original codebase)
path = os.path.abspath("toolkits")
sys.path.append(path)  # Add the parent directory to the Python path
from toolkit import *

class Full_Pointcloud_Maker(Depth_To_Pointcloud):

    def __init__(self,file_path,idx):
        # Object containing the single image point cloud extractor function
        self.estimator = Depth_To_Pointcloud(file_path)

        # Contains all four point clouds of the same scene (3D or color)
        self.Four_Pointclouds = []
        self.Four_Colorclouds = []
        for iv in range(4):
            point_cloud, color_cloud = self.estimator.Generate_Pointcloud(idx,iv)
            self.Four_Pointclouds.append(point_cloud)
            self.Four_Colorclouds.append(color_cloud)

        # Extract the homogeneous Transf. w.r.t the fourth image coordinate system
        file_path = os.path.join(file_path, 'calib.pkl')
        with open(file_path, 'rb') as f:
            self.camera_pose_map = pickle.load(f)

    # Get homogeneous transformation
    # Note that the transf betweeen the four images is contained in the calib.pkl file!
    def Get_Transform(self,iv):
        cam_list = ['840412062035','840412062037','840412062038','840412062076']
        return self.camera_pose_map[cam_list[iv]]

    # Transforms a pointcloud to the frame of the fourth image
    def PointCloud_Transform(self,iv):
        partial_cloud = np.array(self.Four_Pointclouds[iv])
        
        if iv == 3: # reference frame
            return partial_cloud
        
        # Extract homog. transformation
        Transform = self.Get_Transform(iv)

        # Now we perform the whole transformation
        n = partial_cloud.shape[0]
        partial_cloud = np.concatenate((partial_cloud,np.ones((n,1))),axis=1)
        NEW_partial_cloud = (Transform @ partial_cloud.T).T

        # Back to euclidean space
        return np.array(NEW_partial_cloud[:,0:3])
    

    # Complete the full pointcloud by adding them together
    def Point_Cloud_completion(self):
        # Last point cloud is ref. frame
        Full_Point_Cloud = np.array(self.Four_Pointclouds[3])
        Full_color_cloud = np.array(self.Four_Colorclouds[3])
        for iv in range(3):
            Partial_pointcloud = self.PointCloud_Transform(iv)
            Full_Point_Cloud = np.concatenate((Full_Point_Cloud, Partial_pointcloud), axis = 0)
            Full_color_cloud = np.concatenate((Full_color_cloud, self.Four_Colorclouds[iv]), axis = 0)
        
        # Convert pointcloud to meters
        Full_Point_Cloud /= 1000

        # Center the data around zero (for full cloud, not relevant for data)
        centroid = np.mean(Full_Point_Cloud, axis=0)
        Full_Point_Cloud = Full_Point_Cloud - centroid
        
        # We convert to numpy array to o3d.pointcloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(Full_Point_Cloud)  # set pcd_np as the point cloud points
        #pcd_o3d.colors =  o3d.utility.Vector3dVector(Full_color_cloud) # Add the color

        # Cropping the image in a cube centered in the middle of the hand
        min_bound = [-0.20, -0.20, -0.20]
        max_bound = [0.20, 0.20, 0.20]
        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_o3d = pcd_o3d.crop(crop_box)

        # Now we normalize the data
        pcd_array = np.array(pcd_o3d.points)
        scale = np.max(np.sqrt(np.sum(pcd_array**2, axis=1)))
        pcd_array = pcd_array / scale

        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_array)
        
        return pcd_o3d
    
    # Normalize
    def Normalize(self, PointCloud, scaler):
        # We need to scale the true depth and the mano pointclouds with the same scaler

        # Convert data to numpy (easier to work with)
        pcd_array = PointCloud / scaler

        # Now we convert it back
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_array)
        
        return pcd_o3d
    
    # Function specifically for cropping, cnetering and normlaizing the partial pointcloud !
    def center_crop_Partialcloud(self, cloud, center):
        # Center the data around zero
        #self.centroid = np.mean(Partial_cloud, axis=0)
        Complete = cloud - center

        # We convert to numpy array to o3d.pointcloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(Complete)  # set pcd_np as the point cloud points

        # Cropping the image in a cube centered in the middle of the hand
        min_bound = [-0.20, -0.20, -0.20]
        max_bound = [0.20, 0.20, 0.20]
        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_o3d = pcd_o3d.crop(crop_box)

        return(np.array(pcd_o3d.points))


if __name__ == "__main__":
    print("Starting extraction process")
    file_path = "/Users/lukasschuepp/framework/hand_data/data/9-25-1-2"
    manoPath = "/Users/lukasschuepp/framework/hand_data/multiviewDataset/MANO_RIGHT.pkl"

    data_folder = "Hand_Data_9-25-1-2"
    complete_folder = os.path.join(data_folder, "Complete")
    partial_folder = os.path.join(data_folder, "Partial")

    # Make sure the directories exist
    os.makedirs(complete_folder, exist_ok=True)
    os.makedirs(partial_folder, exist_ok=True)


    amount_of_data = 5300
    for idx in tqdm(range(amount_of_data), desc = "loading"):
        # Complete pointcloud
        cloud_maker = Full_Pointcloud_Maker(file_path,idx)
        Complete_cloud = cloud_maker.Point_Cloud_completion() # return pointcloud

        # Define name and create directory for partial clouds 
        id_name = "%05d" % (idx)
        os.makedirs(os.path.join(partial_folder, id_name), exist_ok=True)

        # mano data full cloud
        Mano_Maker = MultiviewDatasetDemo(loadManoParam=True,file_path=file_path,manoPath=manoPath)
        Mano_Maker.renderSingleMesh(idx)
        Complete_cloud_mano = Mano_Maker.pcd_mano #return pointcloud
        Complete_cloud_mano = np.array(Complete_cloud_mano.points)
        np.random.shuffle(Complete_cloud_mano)
        Complete_cloud_mano = Complete_cloud_mano[0:16384,:]
        # Define the center using mano data
        center = np.mean(Complete_cloud_mano, axis=0)
        Complete_cloud_mano = cloud_maker.center_crop_Partialcloud(Complete_cloud_mano, center)
        # Define the scale value of the mano hand
        scaler = np.max(np.sqrt(np.sum(Complete_cloud_mano**2, axis=1)))
        Complete_cloud_mano = cloud_maker.Normalize(Complete_cloud_mano, scaler)

        # Save the files
        filename_mano = os.path.join(complete_folder, id_name + ".pcd")

        # Save the point cloud to the PCD file
        o3d.io.write_point_cloud(filename_mano, Complete_cloud_mano, write_ascii=True)

        for iv in range(4):

            # True depth data partial cloud 
            Partial_cloud = cloud_maker.PointCloud_Transform(iv) #return numpy array
            Partial_cloud /= 1000
            np.random.shuffle(Partial_cloud)
            Partial_cloud = Partial_cloud[0:2048]
            Partial_cloud = cloud_maker.center_crop_Partialcloud(Partial_cloud, center)
            Partial_cloud = cloud_maker.Normalize(Partial_cloud, scaler)

            # Save the files
            filename_sub_partial = os.path.join(partial_folder, id_name, str(iv) + ".pcd")

            # Save the point cloud to the PCD file
            o3d.io.write_point_cloud(filename_sub_partial, Partial_cloud, write_ascii=True)

            # Add color for visualization
            # Partial_cloud.paint_uniform_color([0,1,0])
            # Complete_cloud_mano.paint_uniform_color([1,0,0])

            # Plotting with coord system
            #coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            #draw_geometries = [Complete_cloud_mano, Partial_cloud, coord]
            #o3d.visualization.draw_geometries(draw_geometries)


""""
Additional code fragments:

1) filtering the data with random tree:
        Cloud = np.concatenate((Full_Point_Cloud,Full_color_cloud), axis=1)

        Full = IsolationForest(warm_start=True)
        Full.fit(Cloud)

        # Predict anomaly scores for each data point
        anomaly_scores = Full.decision_function(Cloud)

        # Identify inliers (non-anomalous data points)
        Full = Cloud[anomaly_scores >= 0]

        Full_Point_Cloud = Full[:,0:3]
        Full_color_cloud = Full[:,3:6]

2) Point cloud completion from original codebase
    # original codebase (# Does the exact same thing as my alg.)
    def get_pointcloud_from_rgbd(self, idx, iv, trans_mat=None):
        # Read in the repsective images (rgb and depth)
        rgb_frm = self.readRGB(idx,iv)
        depth_frm = self.readDepth(idx,iv)

        depth_frm = depth_frm.squeeze()
        fg = np.logical_and(depth_frm<2000, depth_frm>50)
        rgb_pts = rgb_frm[fg, :].astype(np.float)
        rgb_pts /= 255.0
 
        width, height = rgb_frm.shape[1], rgb_frm.shape[0]
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
 
        u_pts = u_grid[fg]
        v_pts = v_grid[fg]
        d_pts = depth_frm[fg]
        uvd_pts = np.stack([u_pts, v_pts, d_pts], axis=-1)
 
        xyz_pts = self.perspective_back_projection(iv, uvd_pts)
        return xyz_pts, rgb_pts
    
    def perspective_back_projection(self,iv, uvd_point):

        fx = self.CameraIntrinsics[self.cam_list[iv]].fx 
        fy = self.CameraIntrinsics[self.cam_list[iv]].fy
        cx= self.CameraIntrinsics[self.cam_list[iv]].cx
        cy= self.CameraIntrinsics[self.cam_list[iv]].cy

        if uvd_point.ndim == 1:
            xyz_point = np.zeros((3))
            xyz_point[0] = (uvd_point[0] - cx) * uvd_point[2] / fx
            xyz_point[1] = (uvd_point[1] - cy) * uvd_point[2] / fy
            xyz_point[2] = uvd_point[2]
        elif uvd_point.ndim == 2:
            num_point = uvd_point.shape[0]
            xyz_point = np.zeros((num_point, 3))
            xyz_point[:, 0] = (uvd_point[:, 0] - cx) * \
                uvd_point[:, 2] / fx
            xyz_point[:, 1] = (uvd_point[:, 1] - cy) * \
                uvd_point[:, 2] / fy
            xyz_point[:, 2] = uvd_point[:, 2]
        else:
            raise ValueError('unknown input point shape')
        return xyz_point
"""