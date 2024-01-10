# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 14:38:36
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-03 09:23:07
# @Email:  cshzxie@gmail.com

import numpy as np
import torch
import transforms3d

class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            rnd_axis = np.random.uniform(-1, 1, 3)
            rnd_axis /= np.linalg.norm(rnd_axis)

            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                            RandomMirrorPoints, RandomScalePoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        elif transform.__class__ in [
                            RandomRotatePoints, RandomShiftCloud
                        ]:
                            data[k] = transform(v, rnd_value, rnd_axis)
                        else:
                            data[k] = transform(v)

        return data

class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud

class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud

class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_y = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.1:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.1 and rnd_value <= 0.2: 
            trfm_mat = np.dot(trfm_mat_y, trfm_mat)
        elif rnd_value > 0.2 and rnd_value <= 0.3: 
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.3 and rnd_value <= 0.4:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_y, trfm_mat)
        elif rnd_value > 0.4 and rnd_value <= 0.5: 
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.6: 
            trfm_mat = np.dot(trfm_mat_y, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.6 and rnd_value <= 0.7:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_y, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud

class RandomScalePoints(object):
    def __init__(self, parameters):
        self.scale_low = parameters['scale_low']
        self.scale_high = parameters['scale_high']

    def __call__(self, ptcloud, rnd_value):
        #rnd_value is from 0 to 1. Transform it to scale_low to scale_high
        scale = (self.scale_high - self.scale_low) * rnd_value + self.scale_low
        ptcloud[:, :3] *= scale
        return ptcloud

class RandomRotatePoints(object):
    def __init__(self, parameters):
        self.rotate_range = parameters['rotate_range']

    def __call__(self, ptcloud, rnd_value, rnd_axis):
        # transform rnd_value which is from 0 to 1 to angle which is from -rotate_range to rotate_range
        angle = (rnd_value - 0.5) * 2 * self.rotate_range
        trfm_mat = transforms3d.axangles.axangle2mat(rnd_axis, angle)
        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud

class RandomShiftPoints(object):
    # shift each point individually with some gaussian noise
    def __init__(self, parameters):
        self.shift_range = parameters['shift_range']
    
    def __call__(self, ptcloud):
        shift = np.random.normal(0, self.shift_range, ptcloud.shape)
        ptcloud += shift
        return ptcloud

class RandomShiftCloud(object):
    def __init__(self, parameters) -> None:
        self.shift_range = parameters['shift_range']

    def __call__(self, ptcloud, rnd_value, rnd_axis):
        shift = self.shift_range * rnd_axis
        ptcloud += shift
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data
