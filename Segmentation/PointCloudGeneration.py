#!/usr/bin/env python
import open3d as o3d
import numpy as np
import cv2
import os
from pathlib import Path
from typing import List, Tuple
from collections.abc import Iterable
import utils
import time
import shutil
from tqdm import tqdm

# Distortion parameters
dparam = np.array([-0.22356547153865305, 0.03186705146154515, 0, 0, 0])


def project_2d(points3d, K, R=np.eye(3), t=np.zeros(3),
               dist_coeffs=np.zeros(5,)):
    pts2d, _ = cv2.projectPoints(points3d, R, t, K, dist_coeffs)
    return pts2d


def project_2d_kinect(points3d, M_color, K_color):
    R = np.array(M_color)[:3, :3]
    t = np.array(M_color)[:3, 3]
    K = np.array(K_color)
    pts2d = project_2d(points3d, K, R, t)

    return pts2d


# Assign color to depth cloud
# def get_colored_pcd(pcd, rgb, M_color, K_color, M_depth):
#     pcd = o3d.cpu.pybind.geometry.PointCloud(pcd)
#     # pcd.transform(np.linalg.inv(M_depth))
#     points2d = project_2d_kinect(np.array(pcd.points),
#                                  M_color, K_color[:3, :3])
#
#     height, width = rgb.shape[:2]
#     indices = []
#
#     colors = np.zeros_like(np.array(pcd.points), dtype='float32')
#     for i in range(points2d.shape[0]):
#         dx = int(points2d[i, 0, 0])
#         dy = int(points2d[i, 0, 1])
#
#         if dx < width and dx > 0 and dy < height and dy > 0:
#             colors[i, :] = rgb[dy, dx, ::-1]/255.
#             indices.append(i)
#
#     # pcd.transform(M_depth)
#     pcd_colored = o3d.geometry.PointCloud(pcd)
#     pcd_colored.colors = o3d.utility.Vector3dVector(colors)
#
#     return pcd_colored.select_by_index(indices)


class Frame:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.timeStamp = -1
        self.M = np.zeros((4, 4))
        self.K = np.zeros((4, 4))
        self.path = path

    def readTimeStamp(self, f):
        self.timeStamp = int(f.readline())

    def readM(self, f):
        # Read extrinsic matrix
        for i in range(4):
            self.M[i, :] = np.array(list(map(lambda x: float(x),
                                             f.readline()
                                             .strip().split(", "))))

    def writeM(self):
        with open(self.path.joinpath(f"M_{self.name}.txt"), "w") as f:
            for i in range(np.shape(self.M)[0]):
                for j in range(np.shape(self.M)[1]):
                    f.write(f"{self.M[i,j]}")
                    if j < len(self.M[0]) - 1:
                        f.write(" ")
                f.write("\n")

    def writeK(self):
        with open(self.path.joinpath(f"K_{self.name}.txt"), "w") as f:
            for i in range(np.shape(self.K)[0]):
                for j in range(np.shape(self.K)[1]):
                    f.write(f"{self.K[i,j]}")
                    if j < len(self.K[0]) - 1:
                        f.write(" ")
                f.write("\n")

    def __lt__(self, other):
        return self.timeStamp < other.timeStamp

    def __le__(self, other):
        return self.timeStamp <= other.timeStamp

    def __gt__(self, other):
        return self.timeStamp > other.timeStamp

    def __ge__(self, other):
        return self.timeStamp >= other.timeStamp


class ColorFrame(Frame):
    def __init__(self, name: str):
        super().__init__(name, Path("/cluster/scratch/bergar","ImageCloud", "rgb_ply"))
        self.img = cv2.imread(self.path.joinpath(f"{name}.png").as_posix())
        self.readMetadata()

    def readMetadata(self):
        self.joints = np.loadtxt(self.path.joinpath(f"joints_{self.name}.txt"))
        with open(self.path.joinpath(f"meta_{self.name}.txt")) as f:
            super().readTimeStamp(f)
            super().readM(f)
            # TODO: fix
            self.M = self.M.T

            # Transform extrinsic matrix to right coordinate system
            self.M[:3, 3] *= 1000
            new_axis_transform = np.eye(4)
            new_axis_transform[0, 0] = -1
            new_axis_transform[2, 2] = -1
            self.M = new_axis_transform @ np.linalg.inv(self.M)

            new_axis_transform = np.eye(4)
            new_axis_transform[0, 0] = -1
            new_axis_transform[1, 1] = -1
            self.M = new_axis_transform @ self.M

            # read intrinsics K
            for i in range(4):
                line = f.readline()
                self.K[i, :] = np.array(list(map(lambda x: float(x),
                                                 line.strip().split(", "))))
            self.K = self.K[:3, :3]
            return self


class DepthCloud(Frame):
    def __init__(self, name: str):
        super().__init__(name, Path("/cluster/scratch/bergar","ImageCloud", "depth_ply"))
        self.readMetadata()
        self.loadCloud()

    def loadCloud(self):
        # Create load point cloud from ply
        self.pcd = o3d.io.read_point_cloud(
                self.path.joinpath(f"{self.name}.ply").as_posix())

        # Reflect
        points = np.array(self.pcd.points) * 1000
        # TODO: REMOVE OR ADD FOR RECORDED DATA !!!! TODO
        points[:, 2] *= -1
        self.pcd.points = o3d.utility.Vector3dVector(points)

        # Transform using extrinsics, point cloud is wrt coordinate system
        # centered at origin
        self.pcd.transform(self.M)

    def readMetadata(self):
        with open(depthPath.joinpath(f"meta_{self.name}.txt")) as f:
            super().readTimeStamp(f)
            super().readM(f)
            self.M = self.M.T
            self.M[:3, 3] *= 1000
            self.M = np.linalg.inv(self.M)
            return self

    def writeToPLY(self):
        path = Path("/cluster/scratch/bergar","ImageCloud", "colored_ply")

        points = np.array(self.pcd.points) / 1000
        points[:, 2] *= -1
        self.pcd.points = o3d.utility.Vector3dVector(points)

        if not os.path.exists(path):
            os.makedirs(path)
        o3d.io.write_point_cloud(path.joinpath(f"{self.name}.ply")
                                 .as_posix(),
                                 self.pcd,
                                 write_ascii=True)

    def color_cloud(self, color_frame: ColorFrame):
        color_cam_calibration = {
                "M_color": color_frame.M,
                "K_color": color_frame.K
                }
        self.pcd = utils.get_colored_pcd(self.pcd, color_frame.img,
                                         color_cam_calibration, self.M,
                                         color_frame.joints,
                                         f"{self.name}.png")

    def transform(self, T: np.ndarray):
        self.pcd.transform(T)


def pair_closest_frames(depth: Iterable, rgb: List[ColorFrame]) \
            -> List[Tuple[DepthCloud, ColorFrame]]:
    def get_closest_color_frame(df: DepthCloud) -> ColorFrame:
        idx = np.searchsorted(rgb, df)
        if idx == len(rgb):
            return rgb[-1]
        if idx == 0:
            return rgb[0]
        if df.timeStamp - rgb[idx-1].timeStamp \
                < rgb[idx].timeStamp - df.timeStamp:
            return rgb[idx-1]

        return rgb[idx]

    return [(df, get_closest_color_frame(df)) for df in depth]


if __name__ == "__main__":
    start = time.time()
    # Change Paths accordingly:
    source_path = "TODO"
    depthPath = Path(source_path ,"ImageCloud", "depth_ply")
    rgbPath = Path(source_path,"ImageCloud", "rgb_ply")
    print(f"Paths:\n{depthPath}\n{rgbPath}")
    

    # Get a list of all image files
    depth_images = filter(lambda x: x.endswith(".ply"),
                          os.listdir(depthPath.as_posix()))
    rgb_images = filter(lambda x: x.endswith(".png"),
                        os.listdir(rgbPath.as_posix()))

    # Remove file extension from names
    def drop_extension(x: str) -> str:
        return x.split(".")[0]

    depth_names = (drop_extension(img) for img in depth_images)
    rgb_names = (drop_extension(img) for img in rgb_images)

    depth_clouds = (DepthCloud(name) for name in depth_names)
    rgb_frames = (ColorFrame(name) for name in rgb_names)

    print("Creating list of rgb frames")
    rgb_list = list(rgb_frames)
    rgb_list.sort()

    print("Pairing matching frames")
    paired = pair_closest_frames(depth_clouds, rgb_list)
    paired.sort(key=lambda x: x[0])

    print("Pairings:\n" + "-" * 50)
    for df, cf in paired:
        print(f"({df.name}, {cf.name})")

    for idx, (df, cf) in enumerate(tqdm(paired)):
        frame_start = time.time()
        print(f"Coloring cloud {df.name}, {idx=}")
        df.color_cloud(cf)
        if df.pcd is None:
            print(f"{(df.name,  cf.name)} does not contain hands")
            continue

        # Transform point cloud into object coordinate system
        df.writeToPLY()
        print(f"Processed ({df.name}, {cf.name}) -- ({idx+1} / {len(paired)})")
        print(f"Processing single frame took {time.time() - frame_start}")

    shutil.copy(depthPath.joinpath("object_pose.txt").as_posix(),
                Path(source_path,"ImageCloud", "colored_ply",
                     "object_pose.txt").as_posix())
    
    for file in depthPath.glob("meta*"):
        shutil.copy(file.as_posix(),
                    Path(source_path,"ImageCloud", "colored_ply",
                         file.name).as_posix())

    print(f"Post-processing took {time.time() - start}s")
