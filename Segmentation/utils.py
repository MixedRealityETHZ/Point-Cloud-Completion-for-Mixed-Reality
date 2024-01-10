import numpy as np
import open3d as o3d
import png
import cv2
from scipy.interpolate import NearestNDInterpolator
import os
from pathlib import Path
import segmentation
import time

try:
    from itertools import imap
except ImportError:
    # Python 3...
    imap = map


def get_handpose_connectivity():
    # Hand joint information is in https://github.com/microsoft/psi/tree/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter
    return [
        [0, 1],

        # Thumb
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],

        # Index
        [1, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],

        # Middle
        [1, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],

        # Ring
        [1, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [19, 20],

        # Pinky
        [1, 21],
        [21, 22],
        [22, 23],
        [23, 24],
        [24, 25]
    ]


def read_hand_pose_txt_new(hand_path, is_stereokit=False):
    #  The format for each entry is: Time, IsGripped, IsPinched, IsTracked, IsActive, {Joint values}, {Joint valid flags}, {Joint tracked flags}
    hand_array = []
    with open(hand_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            hand = []
            if is_stereokit:
                line_data = list(map(float, line.split('\t')))

                if line_data[3] == 0.0:  # if hand pose does not exist.
                    # add empty hand location
                    hand_array.append(line_data[:4]+[0]*3*26)
                elif line_data[3] == 1.0:  # if hand pose does exist.
                    line_data_reshape = np.reshape(
                        line_data[4:], (-1, 4, 4))  # (x,y,z) 3, 7 ,11

                    line_data_xyz = []
                    for line_data_reshape_elem in line_data_reshape:
                        # To get translation of the hand joints
                        location = np.dot(line_data_reshape_elem,
                                        np.array([[0, 0, 0, 1]]).T)
                        line_data_xyz.append(location[:3].T[0])

                    line_data_xyz = np.array(line_data_xyz).T
                    hand = line_data[:4]
                    hand.extend(line_data_xyz.reshape(-1))
                    hand_array.append(hand)
            else:
                line_data = list(map(float, line.split('\t')))
                line_data_reshape = np.reshape(
                    line_data[2:-52], (-1, 4, 4))  # (x,y,z) 3, 7 ,11

                line_data_xyz = []
                for line_data_reshape_elem in line_data_reshape:
                    # To get translation of the hand joints
                    location = np.dot(line_data_reshape_elem,
                                    np.array([[0, 0, 0, 1]]).T)
                    line_data_xyz.append(location[:3].T[0])

                line_data_xyz = np.array(line_data_xyz).T
                hand = line_data[:4]
                hand.extend(line_data_xyz.reshape(-1))
                hand_array.append(hand)
        hand_array = np.array(hand_array)
    return hand_array


def read_hand_pose_txt_old(hand_path):
    hand_array = []
    with open(hand_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            hand = []
            line_data = list(map(float, line.split('\t')))
            if line_data[3] == 0.0:  # if hand pose does not exist.
                # add empty hand location
                hand_array.append(line_data[:4]+[0]*3*26)
            elif line_data[3] == 1.0:  # if hand pose does exist.
                line_data_reshape = np.reshape(
                    line_data[4:], (-1, 4, 4))  # (x,y,z) 3, 7 ,11

                line_data_xyz = []
                for line_data_reshape_elem in line_data_reshape:
                    # To get translation of the hand joints
                    location = np.dot(line_data_reshape_elem,
                                      np.array([[0, 0, 0, 1]]).T)
                    line_data_xyz.append(location[:3].T[0])

                line_data_xyz = np.array(line_data_xyz).T
                hand = line_data[:4]
                hand.extend(line_data_xyz.reshape(-1))
                hand_array.append(hand)
        hand_array = np.array(hand_array)
    return hand_array



def read_hand_pose_txt(hand_path, is_stereokit=False):
    #  The format for each entry is: Time, IsGripped, IsPinched, IsTracked, IsActive, {Joint values}, {Joint valid flags}, {Joint tracked flags}
    hand_array = []
    with open(hand_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line == '':  # end of the lines.
                break
            hand = []
            if is_stereokit:
                line_data = list(map(float, line.split('\t')))
                if line_data[3] == 0.0:  # if hand pose does not exist.
                    # add empty hand location
                    hand_array.append(line_data[:4]+[0]*3*26)
                elif line_data[3] == 1.0:  # if hand pose does exist.
                    line_data_reshape = np.reshape(
                        line_data[4:], (-1, 4, 4))  # (x,y,z) 3, 7 ,11

                    line_data_xyz = []
                    for line_data_reshape_elem in line_data_reshape:
                        # To get translation of the hand joints
                        location = np.dot(line_data_reshape_elem,
                                        np.array([[0, 0, 0, 1]]).T)
                        line_data_xyz.append(location[:3].T[0])

                    line_data_xyz = np.array(line_data_xyz).T
                    hand = line_data[:4]
                    hand.extend(line_data_xyz.reshape(-1))
                    hand_array.append(hand)
            else:
                line_data = list(map(float, line.split('\t')))
                line_data_reshape = np.reshape(
                    line_data[2:-52], (-1, 4, 4))  # For version2: line_data[5:-52]

                line_data_xyz = []
                for line_data_reshape_elem in line_data_reshape:
                    # To get translation of the hand joints
                    location = np.dot(line_data_reshape_elem,
                                    np.array([[0, 0, 0, 1]]).T)
                    line_data_xyz.append(location[:3].T[0])

                line_data_xyz = np.array(line_data_xyz).T
                hand = line_data[:4]
                hand.extend(line_data_xyz.reshape(-1))
                hand_array.append(hand)
        hand_array = np.array(hand_array)
    return hand_array


def depthConversion(PointDepth, f, cx, cy):
    H = PointDepth.shape[0]
    W = PointDepth.shape[1]

    columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
    distance_from_center = ((rows - cy)**2 + (columns - cx)**2) ** 0.5
    plane_depth = PointDepth / (1 + (distance_from_center / f)**2) ** 0.5

    return plane_depth


axis_transform = np.linalg.inv(
    np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))


def generatepointcloud(depth, Fx, Fy, Cx, Cy):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    depth_scale = 1
    z = depth * depth_scale

    x = z * (c - Cx) / Fx
    y = z * (r - Cy) / Fy
    points = np.dstack((x, y, z))
    points = points.reshape(-1, 3)
    points = points[~np.all(points == 0, axis=1)]
    return points


# Removes plane from the point cloud
def remove_plane(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    _, inliers = pcd.segment_plane(distance_threshold=.01, ransac_n=3,
                                   num_iterations=1000)
    return pcd.select_by_index(inliers, invert=True)


def load_depth(path):
    # d = imageio.imread(path)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    d = d[:, :, 1]*1. + d[:, :, 2] * 256.
    d = d.astype(np.float32)
    return d


def save_depth(path, im):
    # PyPNG library is used since it allows to save 16-bit PNG
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    im_uint16 = np.round(im).astype(np.uint16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def project_2d(points3d, K, R=np.eye(3), t=np.zeros(3),
               dist_coeffs=np.zeros(5,)):
    
    pts2d, _ = cv2.projectPoints(points3d, R, t, K, dist_coeffs)
    return pts2d

def flip_y_coordinates(points3d, h2o_data=False):
    if h2o_data:
        flipped_points = points3d.copy()
        flipped_points[:, 1] = -flipped_points[:, 1]
    return flipped_points

def project_2d_kinect(points3d, cam_calib_data):
    R = np.array(cam_calib_data['M_color'])[:3, :3]
    t = np.array(cam_calib_data['M_color'])[:3, 3]
    K = np.array(cam_calib_data['K_color'])
    # TODO CHANGE FOR H2O / recorded data !!!! TODO
    # R = np.array([[0, 0, 0]]).astype(np.float32)
    # t = np.array([[0, 0, 0]]).astype(np.float32)

    pts2d = project_2d(points3d, K, R, t)
    return pts2d


def project_joints(joints: np.ndarray, cam_calib_rgb) -> np.ndarray:
    if len(joints) == 0:
        return np.array([])

    joints2d = project_2d_kinect(joints, cam_calib_rgb).reshape((-1, 2))
    # joints2d = np.array([p for p in joints2d
    #                      if p[0] < 860 and p[0] > 0 and
    #                      p[1] < 504 and p[1] > 0])
    return joints2d


def get_best_joint(joints: np.ndarray, cam_calib_rgb) -> np.ndarray:
    # Joints good joints to track as they are not likely to be
    # obscured by objects
    indices = [1, 3, 4]
    joints = joints[joints.any(axis=1)]
    all_joints = project_2d_kinect(np.array(joints), cam_calib_rgb)
    all_joints = all_joints.reshape((-1, 2))
    for idx in indices:
        joint = joints[idx]
        if not joint.any():
            continue
        # Make sure the joint projects onto the image
        joint2d = project_2d_kinect(np.array([joint]), cam_calib_rgb)
        joint2d = np.reshape(joint2d, (-1, 2))
        # Joint is not visible in image
        if joint2d[0, 0] < 0 or joint2d[0, 1] < 0 or \
                joint2d[0, 0] >= 896 or joint2d[0, 1] >= 504:
            continue

        return joint2d

    return None


def get_colored_pcd(pcd, rgb, cam_calib_rgb, M_depth, joints,
                    filename=None):

    # If a point is between MIN_DIST and MAX_DIST from the closest joint,
    # we assume that it is part of the object
    MIN_DIST = 10  # mm
    MAX_DIST = 100  # mm


    #pcd = o3d.cpu.pybind.geometry.PointCloud(pcd)
    pcd = o3d.geometry.PointCloud(pcd)

    pcd.transform(np.linalg.inv(M_depth))
    points = np.array(pcd.points)

    if len(points) == 0:
        print("Point cloud does not contain any points")
        return None

    if rgb is None:
        print("no rgb image given")
        return None

    points2d = project_2d_kinect(points, cam_calib_rgb)
    height, width = rgb.shape[:2]

    colors = np.zeros_like(points, dtype='float32')
    indices = []

    # Perform transformation on joints to get actual world coordinates
    joints *= 1000
    # TODO JOINTS * -1 changed for training data  # TODO: REMOVE OR ADD FOR RECORDED DATA / H2O !!!! TODO
    joints[:, 2] *= -1
    # Remove untracked joints (joints at the origin)
    joints = joints[joints.any(axis=1)]

    joints2d = project_joints(joints, cam_calib_rgb)

    # No hands in frame
    if len(joints2d) == 0:
        print("No hands in frame")
        return None

    segmentation.set_image(rgb)
    hand_mask = segmentation.get_prediction(joints2d)

    # Assign color to hands, potentially some on the object
    for i in range(points2d.shape[0]):
        dx = int(points2d[i, 0, 0])
        dy = int(points2d[i, 0, 1])

        if dx < width and dx > 0 and dy < height and dy > 0\
                and hand_mask[dy, dx]:
            colors[i, :] = rgb[dy, dx, ::-1]/255.
            indices.append(i)

    # Find the point closest to a joint that is not part of either hand
    obj_points = []
    # set to True if you only want hand point clouds
    only_hand_mask = True
    if not only_hand_mask:
        for i in range(max(np.min(indices) - 1000, 0),
                    min(np.max(indices) + 1001, len(points))):
            if i in indices:
                continue

            # Compute the distance to the closest joint
            dist = np.min(np.linalg.norm(points[i] - joints, axis=1))

            # Check if the point is too close to a joint (probably an outlier
            # laying on the hand) or too far away (probably not on the object)
            if dist <= MIN_DIST or dist >= MAX_DIST:
                continue

            obj_points.append(points[i])

    obj_mask = None
    obj_indices = []
    if len(obj_points) > 0:
        obj_points = np.array(obj_points)
        # Project world points to image
        obj2d = project_2d_kinect(np.array(obj_points),
                                  cam_calib_rgb).reshape((-1, 2))
        # Remove points outside the image, should not really happen
        obj2d = np.array([p for p in obj2d if p[0] < 896 and p[1] < 504
                          and p[0] > 0 and p[1] > 0])

        # Randomly sample points s.t. there are as many points on the hand(s)
        # as there are on the object.
        obj2d = obj2d[np.random.choice(len(obj2d),
                                       min(40, len(obj2d)), replace=False)]

        # print(f"{obj2d=}")
        if len(obj2d) > 0:
            obj_mask = segmentation.get_prediction(obj2d)

            # Assign color to object
            for i in range(points2d.shape[0]):
                dx = int(points2d[i, 0, 0])
                dy = int(points2d[i, 0, 1])

                if dx < width and dx > 0 and dy < height and dy > 0 \
                        and obj_mask[dy, dx]:
                            colors[i, :] = rgb[dy, dx, ::-1]/255.
                            obj_indices.append(i)

    if filename is not None:
        maskDest = Path("Masks")
        if not os.path.exists(maskDest):
            os.makedirs(maskDest)
        #if obj_mask is not None:
        combined_mask = hand_mask
        segmentation.save_image(maskDest.joinpath("hand_points_" + filename)
                                .as_posix(),
                                rgb, hand_mask, joints2d)
        # segmentation.save_image(maskDest.joinpath("obj_points_" + filename)
        #                         .as_posix(),
        #                         rgb, obj_mask, obj2d)
        segmentation.save_image(maskDest.joinpath("hand_" + filename)
                                .as_posix(),
                                rgb, hand_mask)
        # segmentation.save_image(maskDest.joinpath("obj_" + filename)
        #                         .as_posix(),
        #                         rgb, obj_mask)
        # segmentation.save_image(maskDest.joinpath("combined_" + filename)
        #                         .as_posix(),
        #                         rgb, combined_mask)
        # else:
        #     print("No object found!")
        #     combined_mask = hand_mask

    pcd_colored = o3d.geometry.PointCloud(pcd)
    # Set to False if no colours
    use_colors = False
    if use_colors:
        pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    return pcd_colored.select_down_sample(indices)


def map_depth_to_rgb(pcd, rgb, cam_calib_rgb,
                     cam_calib_depth, reference='depth', interpolate=True):
    M_depth = cam_calib_depth['M_dist']
    M_color = np.array(cam_calib_rgb['M_color'])
    pcd_points_depth = np.array(pcd.points)

    pcd_points_depth = np.c_[pcd_points_depth,
                             np.ones(pcd_points_depth.shape[0])]

    pcd_points = np.dot(np.dot(M_color, np.linalg.inv(M_depth)),
                        pcd_points_depth.transpose())
    pcd_points = pcd_points.transpose()[:, :3]

    pcd.transform(np.linalg.inv(M_depth))
    pcd.transform(M_color)

    pcd_points = np.array(pcd.points)
    assert False
    points2d = project_2d(pcd_points, np.array(cam_calib_rgb['K_color']))

    height, width = rgb.shape[:2]

    depth = np.zeros((height, width))

    x = []
    y = []
    z = []
    for i in range(points2d.shape[0]):
        dx = int(points2d[i, 0, 0])
        dy = int(points2d[i, 0, 1])

        if reference == 'depth':
            d_i = pcd_points_depth[i, 2]
        elif reference == 'rgb':
            d_i = pcd_points[i, 2]
        else:
            assert False, 'unknown reference'

        if dx < width and dx > 0 and dy < height and dy > 0:
            if d_i < 0:
                continue
            # Handle 3D occlusions
            if depth[dy, dx] == 0 or d_i < depth[dy, dx]:
                depth[dy, dx] = d_i

                x.append(dx*1./width)
                y.append(dy*1./height)
                z.append(d_i)

    if interpolate:
        X = np.array(range(width))*1./width
        Y = np.array(range(height))*1./height
        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        interp = NearestNDInterpolator(np.array(list(zip(x, y)))
                                       .reshape(-1, 2), z)
        Z = interp(X, Y)

        mask = np.zeros_like(depth).astype('uint8')
        indices = np.where(depth > 0)
        mask[indices] = 255
        mask = cv2.blur(mask, (5, 5))
        indices = np.where(mask > 0)
        depth[indices] = Z[indices]

        depth = cv2.medianBlur(depth.astype('float32'), 5)

    return depth
