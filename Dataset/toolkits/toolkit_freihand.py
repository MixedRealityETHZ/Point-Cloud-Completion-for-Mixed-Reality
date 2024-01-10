import sys
sys.path.append("..")
sys.path.append(".")

print("sys path", sys.path)
import os
print("current", os.getcwd())

from manolayer import MANO_SMPL
from globalCamera.util import visualize_better_qulity_depth_map
from globalCamera.camera import CameraIntrinsics,perspective_projection,perspective_back_projection
from globalCamera.constant import Constant
import io,os,pickle
import pyrender
import trimesh
import open3d as o3d
from manopth.manolayer import ManoLayer
from manopth import demo

import numpy as np
import torch
import cv2
import json
import time
from tqdm import tqdm


# Helping class (needed for pickle.load)
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            pass
        return super().find_class(module, name)

def AxisRotMat(angles,rotation_axis):
    x,y,z=rotation_axis
    xx,xy,xz,yy,yz,zz=x*x,x*y,x*z,y*y,y*z,z*z
    c = np.cos(angles)
    s = np.sin(angles)
    i = 1 - c
    rot_mats=np.eye(4).astype(np.float32)
    rot_mats[0,0] =  xx * i + c
    rot_mats[0,1] =  xy * i -  z * s
    rot_mats[0,2] =  xz * i +  y * s

    rot_mats[1,0] =  xy * i +  z * s
    rot_mats[1,1] =  yy * i + c
    rot_mats[1,2] =  yz * i -  x * s

    rot_mats[2,0] =  xz * i -  y * s
    rot_mats[2,1] =  yz * i +  x * s
    rot_mats[2,2] =  zz * i + c
    rot_mats[3,3]=1
    return rot_mats
class MultiviewDatasetDemo():
    def __init__(self,manoPath,
                 file_path,
                 loadManoParam=False,
    ):
        if(manoPath==None):
            self.mano_right = None
        else:
            self.mano_right = MANO_SMPL(manoPath, flat_hand_mean=True, ncomps=45)
        self.loadManoParam=loadManoParam
        self.readNotFromBinary=True
        baseDir = file_path
        self.baseDir=baseDir
        self.date = baseDir[baseDir.rfind('/') + 1:]
        calib_path = os.path.join(baseDir, 'calib.pkl')
        print("baseDir", baseDir)
        print("currentDir", os. getcwd())
        with open(calib_path, 'rb') as f:
            camera_pose_map = pickle.load(f)

        cam_list = ['840412062035','840412062037','840412062038','840412062076']
        self.cam_list = cam_list
        cam_list.sort() 
        camera, camera_pose,Ks = [], [],[]
        for camera_ser in cam_list:
            camera.append(CameraIntrinsics[camera_ser])
            camera_pose.append(camera_pose_map[camera_ser])
            K = np.eye(3)
            K[0, 0] = CameraIntrinsics[camera_ser].fx
            K[1, 1] = CameraIntrinsics[camera_ser].fy
            K[0, 2] = CameraIntrinsics[camera_ser].cx
            K[1, 2] = CameraIntrinsics[camera_ser].cy
            Ks.append(K.copy())
        for i in range(4):
            if (np.allclose(camera_pose[i], np.eye(4))):
                rootcameraidx = i
        # print("camera_pose",camera_pose)
        self.camera_pose,self.camera,self.Ks=camera_pose,camera,Ks
        self.rootcameraidx=rootcameraidx
        print('self.rootcameraidx',self.rootcameraidx)

        joints = np.load(os.path.join(baseDir, "mlresults", self.date + '-joints.npy'))
        self.N=joints.shape[0]
        self.joints=joints.reshape(self.N,21,4,1).astype(np.float32)

        joints4view = np.ones((4, self.N, 21, 4, 1)).astype(np.int64)
        for dev_idx, rs_dev in enumerate(cam_list):
            inv = np.linalg.inv(camera_pose[dev_idx])
            joints4view[dev_idx] = inv @ self.joints
        self.joints4view=joints4view

        self.avemmcp=np.mean(joints4view[rootcameraidx,:,5,:3],axis=0)

        self.mano_freihand = self.get_freihand_data()

        if(loadManoParam):
            with open(os.path.join(self.baseDir,self.date+'manoParam.pkl'), 'rb') as f:
                #self.manoparam = torch.load(f, map_location=torch.device('cpu'))
                self.manoparam = CPU_Unpickler(f).load()


    def getCameraIntrinsic(self,iv):
        cam_list = ['840412062035', '840412062037', '840412062038', '840412062076']
        return self.camera[cam_list[iv]]
    def getCameraPose(self):
        calib_path = os.path.join(self.baseDir, 'calib.pkl')
        with open(calib_path, 'rb') as f:
            camera_pose_map = pickle.load(f)
        cam_list = ['840412062035', '840412062037', '840412062038', '840412062076']
        camera_pose = []
        for camera_ser in cam_list:
            camera_pose.append(camera_pose_map[camera_ser])
        return camera_pose


    def readRGB(self,idx,iv):
        rgbpath = os.path.join(self.baseDir, 'rgb')
        rgbpath = os.path.join(rgbpath, "%05d" % (idx) + '_' + str(iv) + '.jpg')
        return cv2.imread(rgbpath)


    def getMask(self,idx,uselist=False):
        dms = []
        for iv in range(4): dms.append(self.readMask(idx, iv))
        if (uselist):
            return dms
        else:
            return np.hstack(dms)

    def readMask(self,idx,iv):
        rgbpath = os.path.join(self.baseDir, 'mask')
        rgbpath = os.path.join(rgbpath, "%05d" % (idx) + '_' + str(iv) + '.jpg')
        return cv2.imread(rgbpath)

    def decodeDepth(self,rgb:np.ndarray):
        """ Converts a RGB-coded depth into depth. """
        assert (rgb.dtype==np.uint8)
        r, g, _ = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        depth = (r.astype(np.uint64) + g.astype(np.uint64) * 256).astype(np.uint16)
        return depth

    def readDepth(self,idx,iv):
        dpath = os.path.join(self.baseDir, 'depth')
        dpath = os.path.join(dpath, "%05d" % (idx) + '_' + str(iv) + '.png')
        return self.decodeDepth(cv2.imread(dpath))

    def getImgs(self,idx,uselist=False):
        if(uselist):return self.getImgsList(idx)
        color=[]
        for iv in range(4):color.append(self.readRGB(idx,iv))
        return np.hstack(color)
    def getImgsList(self,idx,facemask=True):
        color=[]
        for iv in range(4):
            img=self.readRGB(idx,iv)
            color.append(img)
        return color
    def getDepth(self,idx,uselist=False):
        dms = []
        for iv in range(4): dms.append(self.readDepth(idx,iv))
        if(uselist):return dms
        else:return np.hstack(dms)
    def getBetterDepth(self,idx,uselist=False):
        dlist = []
        for iv in range(4):
            depth = self.readDepth(idx,iv)
            dlist.append(visualize_better_qulity_depth_map(depth))
        if (uselist): return dlist
        return np.hstack(dlist)

    # transform manoParams to mano vertices
    def getManoVertex(self,idx):
        results = self.mano_freihand[idx]
        vertex, joint_pre = \
            self.mano_right.get_mano_vertices(results['pose_aa'][:, 0:1, :],
                                         results['pose_aa'][:, 1:, :], results['shape'],
                                         results['scale'], results['transition'],
                                         pose_type='euler', mmcp_center=False)
        vertex=vertex.cpu()
        vertices = (vertex)[0].cpu().detach().numpy() * 1000
        vertices = np.concatenate([vertices, np.ones([vertices.shape[0], 1])], axis=1)
        vertices = np.expand_dims(vertices, axis=-1)
        self.vertices=vertices
        return vertices

    # retrieve manoParams
    def getManoParamFromDisk(self,idx):
        #return a dictionary which includes mano pose parameters, scale, and transition
        assert (self.loadManoParam==True)
        results=self.manoparam[idx]
        c=np.sqrt(np.sum((self.joints[idx,5,:3,0]/1000-self.joints[idx,1,:3,0]/1000)**2))
        scale=torch.tensor(c,dtype=torch.float32)
        joint_root=torch.tensor(self.joints[idx,5,:3,0]/1000,dtype=torch.float32)
        return results,scale, joint_root
    
    def backproject_ortho(self, uv, scale,  # kind of the predictions
                      focal, pp):  # kind of the camera calibration
        """ Calculate 3D coordinates from 2D coordinates and the camera parameters. """
        uv = uv.copy()
        uv -= pp
        xyz = np.concatenate([np.reshape(uv, [-1, 2]),
                            np.ones_like(uv[:, :1])*focal], 1)
        xyz /= scale
        return xyz

    def recover_root(self, uv_root, scale,
                    focal, pp):
        uv_root = np.reshape(uv_root, [1, 2])
        xyz_root = self.backproject_ortho(uv_root, scale, focal, pp)
        return xyz_root
    
    def get_focal_pp(self, K):
        """ Extract the camera parameters that are relevant for an orthographic assumption. """
        focal = 0.5 * (K[0, 0] + K[1, 1])
        pp = K[:2, 2]
        return focal, pp
    
    def _assert_exist(self, p):
        msg = 'File does not exists: %s' % p
        assert os.path.exists(p), msg

    def json_load(self, p):
        self._assert_exist(p)
        with open(p, 'r') as fi:
            d = json.load(fi)
        return d

    def load_db_annotation(self, base_path, set_name=None):
        if set_name is None:
            # only training set annotations are released so this is a valid default choice
            set_name = 'training'

        print('Loading FreiHAND dataset index ...')
        t = time.time()

        # assumed paths to data containers
        k_path = os.path.join(base_path, '%s_K.json' % set_name)
        mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
        xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

        # load if exist
        K_list = self.json_load(k_path)
        mano_list = self.json_load(mano_path)
        xyz_list = self.json_load(xyz_path)

        # should have all the same length
        assert len(K_list) == len(mano_list), 'Size mismatch.'
        assert len(K_list) == len(xyz_list), 'Size mismatch.'

        print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
        return K_list, mano_list, xyz_list

    # transfrom 3D hand mesh vertices into 4 views
    def get4viewManovertices(self,idx):
        vertices=self.getManoVertex(idx) # shape (778,4,1)
        vertices4view=np.zeros([4,778,4,1])
        for iv in range(4):
            inv = np.linalg.inv(self.camera_pose[iv])
            vertices4view[iv] = (inv @ vertices)
        self.vertices4view=vertices4view
        return vertices4view

    # render the 4 views
    def render4mesh(self,idx,ratio=1):
        #the ratio=10 can make the rendered image be black
        vertices4view=self.get4viewManovertices(idx)
        import trimesh
        import pyrender
        from pyrender import RenderFlags
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [3, 3])
        recolorlist=[]
        for iv in range(4):
            xyz=vertices4view[iv,:778,:3,0].copy()
            cv = xyz @ np_rot_x
            tmesh = trimesh.Trimesh(vertices=cv / 1000*ratio, faces=self.mano_right.faces)
            # tmesh.visual.vertex_colors = [.9, .7, .7, 1]
            # tmesh.show()
            mesh = pyrender.Mesh.from_trimesh(tmesh)
            scene = pyrender.Scene()
            scene.add(mesh)
            pycamera = pyrender.IntrinsicsCamera(self.camera[iv].fx, self.camera[iv].fy, self.camera[iv].cx, self.camera[iv].cy, znear=0.0001,
                                                 zfar=3000)
            ccamera_pose = self.camera_pose[self.rootcameraidx]
            scene.add(pycamera, pose=ccamera_pose)
            light = pyrender.SpotLight(color=np.ones(3), intensity=2.0, innerConeAngle=np.pi / 16.0)
            # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            scene.add(light, pose=ccamera_pose)
            r = pyrender.OffscreenRenderer(640, 480)
            # flags = RenderFlags.SHADOWS_DIRECTIONAL
            recolor, depth = r.render(scene)
            # cv2.imshow("depth", depth)
            recolorlist.append(recolor[:, :, :3])
        meshcolor = np.hstack(recolorlist)
        return meshcolor

    # overlay rendered meshes with real images, idx is the frame number
    def drawMesh(self,idx):
        recolor=self.render4mesh(idx)
        color=np.hstack(self.getImgsList(idx))
        recolor[recolor == 255] = color[recolor == 255]
        c = cv2.addWeighted(color, 0.1, recolor, 0.9, 0.0)
        return c

    def getPose2D(self,idx,view):
        ujoints = self.joints4view[view, idx, :21, :3, 0].copy()
        uvdlist=[]
        for jdx in range(21):
            rgbuvd = perspective_projection(ujoints[jdx], self.camera[view]).astype(int)[:2]
            uvdlist.append(rgbuvd)
        return np.array(uvdlist).reshape(21,2)#uvd array

    def getPose3D(self,idx,view):
        return self.joints4view[view, idx, :21, :3, 0].copy()

    def drawPose4view(self,idx,view=4):
        assert (view == 1 or view == 4), "only support 4 and 1 view"
        lineidx = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
        uvdlist = []
        imgs=self.getImgsList(idx)
        for iv in range(4):
            ujoints=self.joints4view[iv,idx,:21,:3,0].copy()
            for jdx in range(21):
                rgbuvd = perspective_projection(ujoints[jdx], self.camera[iv]).astype(int)[:2]
                uvdlist.append(rgbuvd)

                color=np.array(Constant.joint_color[jdx]).astype(int)
                imgs[iv] = cv2.circle(imgs[iv], tuple(rgbuvd), 3, color.tolist(), -1)
                if (jdx in lineidx):
                    imgs[iv] = cv2.line(imgs[iv], tuple(rgbuvd), tuple(uvdlist[-2]), color.tolist(), thickness=2)
        if(view==1):imgs = imgs[0].copy()
        else:imgs = np.hstack(imgs)
        return imgs
    
    def mano_visualizer(self, pose, shape):
        # Initialize MANO layer
        mano_layer = ManoLayer(
            mano_root='mano/models', use_pca=False, ncomps=45, flat_hand_mean=False)

        # Generate random shape parameters
        # random_shape = torch.rand(batch_size, 10)
        # Generate random pose parameters, including 3 values for global axis-angle rotation
        # random_pose = torch.rand(batch_size, ncomps + 3)

        # Forward pass through MANO layer
        hand_verts, hand_joints = mano_layer(pose, shape)
        # demo.display_hand({
        #     'verts': hand_verts,
        #     'joints': hand_joints
        # },
        #                 mano_faces=mano_layer.th_faces)
        # print("done")
        return hand_verts

    def get_freihand_data(self):
        path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/FreiHand/training_mano.json'
        json_file = open(path)
        data = json.load(json_file)
        complete_data = []
        # for idx in range(10):
        for idx in tqdm(range(len(data))):
            cur_data = data[idx][0]
            poses = cur_data[:48]
            shapes = cur_data[48:58]
            # convert to numpy
            poses = np.array(poses).astype(np.float32)
            shapes = np.array(shapes).astype(np.float32)
            shapes = torch.from_numpy(shapes).unsqueeze(0)
            poses = torch.from_numpy(poses).unsqueeze(0)
            hand_verts = self.mano_visualizer(poses, shapes)
            complete_data.append(hand_verts)
        return complete_data
    

    def renderSingleMesh(self, idx, ratio=1):
        vertices = self.mano_freihand[idx]
        # convert to numpy
        vertices = vertices.cpu().detach().numpy()

        # Create a trimesh
        cv = vertices[0, :, :3].copy() / 1000 * ratio
        tmesh = trimesh.Trimesh(vertices=cv, faces=self.mano_right.faces)

        # Create an Open3D TriangleMesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(cv)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(self.mano_right.faces)

        # Compute the normals for the Open3D TriangleMesh
        mesh_o3d.compute_vertex_normals()

        # Create point cloud with n points sampled from the mesh
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=16384)

        # Convert to numpy to normalize
        Full_Point_Cloud = np.array(pcd.points)

        self.pcd_mano = o3d.geometry.PointCloud()
        self.pcd_mano.points = o3d.utility.Vector3dVector(Full_Point_Cloud)

        #coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        #draw_geometries = [self.pcd_mano,coord]
        #o3d.visualization.draw_geometries(draw_geometries)

        
        # Create an Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the TriangleMesh to the visualizer
        vis.add_geometry(pcd)

        # Start the visualization
        vis.run()

        # Close the visualizer when done
        vis.destroy_window()
        

    def renderPartialMesh(self, idx, ratio=1):
        vertices = self.mano_freihand[idx]
        # convert to numpy
        vertices = vertices.cpu().detach().numpy()

        # Create a trimesh
        cv = vertices[0, :, :3].copy() / 1000 * ratio

        # Create an Open3D TriangleMesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(cv)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(self.mano_right.faces)

        # Compute the normals for the Open3D TriangleMesh
        mesh_o3d.compute_vertex_normals()

        # Create point cloud with n points sampled from the mesh
        pcd_raw = mesh_o3d.sample_points_uniformly(number_of_points=16384)

        # normalize the pcd
        pcd_points = np.asarray(pcd_raw.points)
        pcd_centralized = pcd_points - np.mean(pcd_points, axis=0)
        m = np.max(np.sqrt(np.sum(pcd_centralized**2, axis=1)))
        pcd_normalized = pcd_centralized / m

        # create a new pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_normalized)

        id = "%05d" % (idx)
        id = '33' + id
        # save_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/our_data/FreiHand'
        save_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/our_data/PCN/train'
        # Define parameters used for hidden_point_removal"
        for idx, perspective in enumerate([[5, -5, 5], [-5, -5, 5], [5, -5, -5], [-5, -5, -5]]):
            diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
            camera = perspective
            radius = diameter * 10000

            # Get all points that are visible from the given view point
            _, pt_map = pcd.hidden_point_removal(camera, radius)
            pcd_pt = pcd.select_by_index(pt_map)

            # Create a larger visual representation for the camera point
            camera_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # Adjust the radius as needed
            camera_mesh.translate(camera)  # Set the position of the camera mesh
            # camera_pcd = o3d.geometry.PointCloud()
            # camera_pcd.points = o3d.utility.Vector3dVector([camera])

            # opt = o3d.visualization.Visualizer()
            # opt.create_window()

            # add pcd and with grey color
            # pcd.paint_uniform_color([0.5, 0.5, 0.5])
            # opt.add_geometry(pcd)

            # opt.add_geometry(pcd_pt) # Add the partial point cloud
            # opt.add_geometry(camera_mesh)  # Add the custom camera mesh

            # opt.add_geometry(camera_pcd)
            # opt.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0]))
            # render = opt.get_render_option()
            # render.show_coordinate_frame = True

            # save the partial point cloud
            # make a id with 5 digits with idx at the end
            # random sample 2048 points from the partial point cloud
            pcd_pt_points = np.asarray(pcd_pt.points)
            np.random.shuffle(pcd_pt_points)
            pcd_pt_points = pcd_pt_points[0:2048]
            pcd_pt.points = o3d.utility.Vector3dVector(pcd_pt_points)
            os.makedirs(os.path.join(save_path, 'partial', '10', id), exist_ok=True)
            o3d.io.write_point_cloud(os.path.join(save_path, 'partial', '10', id, f"{idx}.pcd"), pcd_pt, write_ascii=True)

            # opt.run()
            # opt.destroy_window()
        os.makedirs(os.path.join(save_path, 'complete', '10'), exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_path, 'complete', '10', f"{id}.pcd"), pcd, write_ascii=True)



if __name__ == "__main__":
    # I put in the manual paths b.c ../ didn't work. Need to be changed!
    file_path1 = "/Users/simonschlapfer/Documents/ETH/Master/MixedReality/data/7-14-1-2"
    file_path2 = "/Users/lukasschuepp/framework/hand_data/data/9-10-1-2"
    file_path3 = "/Users/lukasschuepp/framework/hand_data/data/9-17-1-2"
    file_path4 = '/Users/lukasschuepp/framework/hand_data/data/9-25-1-2'
    file_paths = [file_path1,file_path2, file_path3, file_path4]
    manoPath = "/Users/simonschlapfer/Documents/ETH/Master/MixedReality/multiviewDataset/MANO_RIGHT.pkl"
    #for path in file_paths:
    path = file_path1
    demo=MultiviewDatasetDemo(loadManoParam=True,file_path=path,manoPath=manoPath)
    # demo.renderSingleMesh(5)
    # demo.renderPartialMesh(0)
    for i in tqdm(range(32560)):
        demo.renderPartialMesh(i)
    #     meshcolor=demo.drawMesh(i)
    #     cv2.imshow("meshcolor", meshcolor)
        # imgs=demo.drawPose4view(i)
        # cv2.imshow("imgs", imgs)
        # depth = demo.getBetterDepth(i)
        # cv2.imshow("depth", depth)
        # depth = demo.getMask(i)
        # cv2.imshow("depth", depth)
        # cv2.waitKey(0)       
        