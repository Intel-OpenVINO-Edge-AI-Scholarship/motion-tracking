import torch
from torch import nn
from . import scenenet_pb2 as sn
import cv2
from .utils import *
import numpy as np

class Pose():

    def __init__(self, image_width=320, image_height=240, factor=0.001):
        super(Pose, self).__init__()
        self.width = image_width
        self.height = image_height
        self.factor = factor
    
    def pixel_to_ray(self, pixel, vfov=45, hfov=60):
        x, y = pixel
        data = np.zeros((self.height, self.width, 3))
        x_vect = np.tan(np.radians(hfov/2.0)) * ((2.0 * ((x+0.5)/self.width)) - 1.0)
        y_vect = np.tan(np.radians(vfov/2.0)) * ((2.0 * ((y+0.5)/self.height)) - 1.0)
        data[:,:,0] = x_vect
        data[:,:,1] = y_vect
        data[:,:,2] = 1.0
        return data

    def normalised_pixel_to_ray_array(self):
        pixel_to_ray_array = torch.zeros((self.height,self.width,3))
        pixel = np.meshgrid(np.arange(0,self.width), np.arange(0,self.height))
        pixel_to_ray_array[:,:,:] = normalize(
            torch.from_numpy(self.pixel_to_ray(pixel))
        )
        return pixel_to_ray_array
    
    def position_to_tensor(self, position):
        return np.array([ position['x'], position['y'], position['z'] ])
    
    def interpolate_poses(self, start_pose, end_pose, alpha):
        assert alpha >= 0.0
        assert alpha <= 1.0
        camera_pose = alpha * self.position_to_tensor(end_pose['camera'])
        camera_pose += (1.0 - alpha) * self.position_to_tensor(start_pose['camera'])
        lookat_pose = alpha * self.position_to_tensor(end_pose['lookat'])
        lookat_pose += (1.0 - alpha) * self.position_to_tensor(start_pose['lookat'])
        timestamp = alpha * end_pose['timestamp'] + (1.0 - alpha) * start_pose['timestamp']
        pose = sn.Pose()
        pose.camera.x = camera_pose[0]
        pose.camera.y = camera_pose[1]
        pose.camera.z = camera_pose[2]
        pose.lookat.x = lookat_pose[0]
        pose.lookat.y = lookat_pose[1]
        pose.lookat.z = lookat_pose[2]
        pose.timestamp = timestamp
        return pose

