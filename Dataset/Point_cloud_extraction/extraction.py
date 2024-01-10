import numpy as np
import matplotlib as plt
import cv2
import io,os
import open3d as o3d
import pickle

from camera_params import CameraIntrinsics
"""
Returns single pointclouds
L.Schüepp
"""

class Depth_To_Pointcloud():

    def __init__(self,file_path):
        # Directory witht the depth images
        self.path = file_path
        self.cam_list = ['840412062035','840412062037','840412062038','840412062076']
        self.CameraIntrinsics = CameraIntrinsics
        
        # Choose if pointclouds should have color
        self.Artificial_color = False
        # True color
        self.True_color = True
        

    # idx: correpsonds to picture sequence
    # iv: corresponds to camera {0,1,2,3} 
    def readRGB(self,idx,iv):
        rgbpath = os.path.join(self.path, 'rgb')
        rgbpath = os.path.join(rgbpath, "%05d" % (idx) + '_' + str(iv) + '.jpg')
        return cv2.imread(rgbpath)

    # idx: correpsonds to picture sequence
    # iv: corresponds to camera {0,1,2,3}
    def readDepth(self,idx,iv):
        depthpath = os.path.join(self.path, 'depth')
        depthpath = os.path.join(depthpath, "%05d" % (idx) + '_' + str(iv) + '.png')
        depth_image = cv2.imread(depthpath)

        # Decoding the rgb image into the real depth image (depth given in milimeters)
        r, g, _ = depth_image[:, :, 0], depth_image[:, :, 1], depth_image[:, :, 2]
        depth = (r.astype(np.uint64) + g.astype(np.uint64) * 256).astype(np.uint16)

        # We try to increase the depth quality
        #depth = self.visualize_better_qulity_depth_map(depth)
        return depth

    def visualize_better_qulity_depth_map(self,depth_image):
        vis_depth_image = depth_image.copy().astype(np.float32)
        vis_depth_image = np.clip(vis_depth_image,0,2000)
        vis_depth_image = vis_depth_image * 255 / 2000
        vis_depth_image[vis_depth_image < 10] = 255
        vis_depth_image=vis_depth_image*3-50
        mask=(vis_depth_image>60)&(vis_depth_image<100)
        vis_depth_image[mask]=vis_depth_image[mask]*3-50
        vis_depth_image[~mask]*=2
        return vis_depth_image
        #vis_depth_image = np.clip(vis_depth_image, 0, 255)
        #vis_depth_image = vis_depth_image.astype(np.uint8)

        #return cv2.cvtColor(vis_depth_image, cv2.COLOR_GRAY2BGR)
    
    # Returnes the mask that segments the hand from the scene (inaccurate!!)
    def readMask(self,idx,iv):
        maskpath = os.path.join(self.path, 'mask')
        maskpath = os.path.join(maskpath, "%05d" % (idx) + '_' + str(iv) + '.jpg')
        #maskpath = os.path.join(self.path, 'Seg_anything_mask')
        #maskpath = os.path.join(maskpath, "mask_" + str(iv) + ".jpg")
        mask = cv2.imread(maskpath)
        mask = mask[:,:,0]/255
    
        # We filter out all low values 
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (mask[i,j] >= 0.95):
                    mask[i,j] = 1
                else:
                    mask[i,j] = 0
        return mask
        
    # Gnerates a single point cloud of a single picture
    def Generate_Pointcloud(self,idx,iv):
        # Initialize depth image
        depth = self.readDepth(idx,iv)
        mask = self.readMask(idx,iv)
        rgb = self.readRGB(idx,iv)

        # Lay mask over depth image
        depth = mask * depth
                    
        #introduce internal params, self.cam_list[iv] returns string name of the iv camera
        fx = self.CameraIntrinsics[self.cam_list[iv]].fx 
        fy = self.CameraIntrinsics[self.cam_list[iv]].fy
        cx= self.CameraIntrinsics[self.cam_list[iv]].cx
        cy= self.CameraIntrinsics[self.cam_list[iv]].cy

        point_cloud = []
        color_cloud = []
        for i in range(depth.shape[0]):
            for k in range(depth.shape[1]):
                z = depth[i,k]
                x = z * (k-cx) / fx
                y = z * (i-cy) / fy
                # Only consider 3d points that are relevant
                if (z > 0):
                    point = [x,y,z]
                    point_cloud.append(point)
                    # if true ground color shoud be used
                    if self.True_color == True:
                        color = [rgb[i,k,2]/255,rgb[i,k,1]/255,rgb[i,k,0]/255]
                        color_cloud.append(color)
        
        point_cloud = np.array(point_cloud)

        # If we use artificial colors to differ the four images
        if self.Artificial_color == True:
            # Red, Green, Blue, Yelllow
            colors_iv = [[255,0,0],[0,255,0],[0,0,255],[255,255,0]]

            # add color to pointcloud (as a function of camera iv)
            color_to_add = colors_iv[iv]
            color_cloud = np.tile(color_to_add, (point_cloud.shape[0],1))

        color_cloud = np.array(color_cloud)
        
        return point_cloud, color_cloud

    def plot_some_stuff(self, image):

        cv2.imshow('image',image)
        cv2.waitKey(0)
        print(f"Image resolution: {image.shape}")
        print(f"Data type: {image.dtype}")
        print(f"Min value: {np.min(image)}")
        print(f"Max value: {np.max(image)}")

# Thid main function was mainly for testing anc doesn't need to be run
if __name__ == "__main__":
    print("Starting extraction process")
    file_path = "/Users/lukasschuepp/framework/hand_data/data/7-14-1-2"

    # idx is frame
    iv = 0
    idx = 0

    heyooo = Depth_To_Pointcloud(file_path)
    Complete, color = heyooo.Generate_Pointcloud(idx,iv)
    Complete/=1000 # Converting to m
    #Complete2, Color2 = heyooo.get_pointcloud_from_rgbd(idx,iv)

    # Center the data around zero
    centroid = np.mean(Complete, axis=0)
    Complete = Complete - centroid

    # Sample it down
    Complete = Complete[::7]
    print(Complete.shape)
        
    # We convert to numpy array to o3d.pointcloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(Complete)  # set pcd_np as the point cloud points

    # Cropping the image in a cube centered in the middle of the hand
    min_bound = [-0.25, -0.25, -0.25]
    max_bound = [0.25, 0.25, 0.25]
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcd_o3d = pcd_o3d.crop(crop_box)

    # normalize data
    pcd_array = np.array(pcd_o3d.points)
    m = np.max(np.sqrt(np.sum(pcd_array**2, axis=1)))
    pcd_array = pcd_array / m

    pcd_o3d.points = o3d.utility.Vector3dVector(pcd_array)

    # Save the cloud
    # Specify the filename for the PCD file
    pcd_filename = "partial_hand03.pcd"

    # Save the point cloud to the PCD file
    o3d.io.write_point_cloud(pcd_filename, pcd_o3d, write_ascii=True)

    print(f"Point cloud saved to {pcd_filename}")   

    """
    pcd_o3d2 = o3d.geometry.PointCloud()
    pcd_o3d2.points = o3d.utility.Vector3dVector(Complete2)
    pcd_o3d2.paint_uniform_color([0, 1, 0])
    
    combined = pcd_o3d + pcd_o3d2
    """
    #pcd_o3d.colors =  o3d.utility.Vector3dVector(Color)

    # Outlier removal with statistical approach
    #pcd_stat, ind_stat = pcd_o3d.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)

    # Outliers removal with radial approach
    #pcd_rad, ind_r = pcd_o3d.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)

    # Cropping the image
    #min_bound = [-math.inf, -math.inf, 0.3]
    #max_bound = [math.inf, math.inf, 0.9]
    #crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    #pcd_o3d = pcd_o3d.crop(crop_box)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    draw_geometries = [pcd_o3d, coord]
    o3d.visualization.draw_geometries(draw_geometries)


