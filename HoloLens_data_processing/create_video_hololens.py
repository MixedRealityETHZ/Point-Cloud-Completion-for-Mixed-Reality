import ffmpeg
import os
import shutil
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import open3d as o3d
import cv2
import matplotlib; matplotlib.use('agg')
from tqdm import tqdm

def read_meta_data(pcd_path, partial_path):
    id = pcd_path.split('/')[-2]
    meta_data_path = os.path.join(partial_path, 'depth', 'meta_' + id + '.txt')
    M = np.zeros((4, 4))
    with open(meta_data_path) as f:
        timeStamp = int(f.readline())
        for i in range(4):
            M[i, :] = np.array(list(map(lambda x: float(x),f.readline().strip().split(", "))))
    M = M.T
    # M[:3, 3] *= 1000
    M = np.linalg.inv(M)
    return M

def get_ptcloud_img(ptcloud, view):
    # ptcloud = ptcloud[np.random.permutation(2048)]
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    try:
        ax = fig.gca(projection=Axes3D.name, adjustable='box')
    except:
        ax = fig.add_subplot(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(view[0], view[1])
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet', s=2)

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close()
    return img

def get_png(pcd_path, partial_path):
    # Step 1: Load Point Cloud
    points = np.load(pcd_path)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    id = pcd_path.split('/')[-2]
    save_path = partial_path + '_video'
    for perspective in ['top', 'bottom', 'left', 'back']:
        if perspective == 'top':
            view = (30, -45)
        elif perspective == 'bottom':
            view = (-30, 45)
        elif perspective == 'left':
            view = (30, 45)
        elif perspective == 'back':
            view = (-30, -45)
        
        split = pcd_path.split('/')[-1].split('.')[0]
        img = get_ptcloud_img(points, view)
        os.makedirs(os.path.join(save_path, perspective+f'_{split}'), exist_ok=True)
        cv2.imwrite(os.path.join(save_path, perspective+f'_{split}', id + '.jpg'), img)

def create_images_from_pc(partial_path, prediction_path):
    for frame in tqdm(os.listdir(prediction_path)):
        if frame in '.DS_Store':
            continue
        point_cloud_path = os.path.join(prediction_path, frame, 'fine.npy')
        get_png(point_cloud_path, partial_path)
        point_cloud_path = os.path.join(prediction_path, frame, 'input.npy')
        get_png(point_cloud_path, partial_path)


def create_video(partial_path, prediction_path):
    video_path = partial_path + '_video'
    os.makedirs(video_path, exist_ok=True)
    # partial rgb video
    input_pattern = '*.png'
    file_path = os.path.join(partial_path, 'rgb', input_pattern)
    output_file = os.path.join(video_path, 'rgb.mov')
    ffmpeg.input(file_path, framerate=18, pattern_type='glob').output(output_file, vcodec='libx264', vf='scale=640:-2,format=yuv420p').run()

    # prediction video
    create_images_from_pc(partial_path, prediction_path)
    for perspective in ['top_fine', 'bottom_fine', 'left_fine', 'back_fine', 'top_input', 'bottom_input', 'left_input', 'back_input']:
        input_pattern = '*.jpg'
        file_path = os.path.join(video_path, perspective, input_pattern)
        output_file = os.path.join(video_path, f'{perspective}.mov')
        ffmpeg.input(file_path, framerate=20, pattern_type='glob').output(output_file, vcodec='libx264', vf='scale=640:-2,format=yuv420p').run()



if __name__ == "__main__":
    partial_path = '/Users/simonschlapfer/Documents/ETH/Master/MixedReality/HoloLens/PRINTER_TUTORIAL5'
    prediction_path = partial_path + '_prediction'
    create_video(partial_path,   prediction_path)

