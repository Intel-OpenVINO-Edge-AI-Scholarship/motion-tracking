from torch import nn
from .utils import *
import math
import numpy as np

class World():

    def __init__(self, pose):
        super(World, self).__init__()
        self.pose = pose
    
    def world_to_camera_with_pose(self, view_pose):
        lookat_pose = self.pose.position_to_tensor({'x': view_pose["lookat"]["x"], 'y': view_pose["lookat"]["y"], 'z': view_pose["lookat"]["z"]})
        camera_pose = self.pose.position_to_tensor({'x': view_pose["camera"]["x"], 'y': view_pose["camera"]["y"], 'z': view_pose["camera"]["z"]})
        up = np.array([0,1,0])
        R = np.diag(np.ones(4))
        R[2,:3] = normalize_norm_np(lookat_pose - camera_pose)
        R[0,:3] = normalize_norm_np(np.cross(R[2,:3],up))
        R[1,:3] = -normalize_norm_np(np.cross(R[0,:3],R[2,:3]))
        T = np.diag(np.ones(4))
        T[:3,3] = -camera_pose
        return R.dot(T)

    def camera_point_to_uv_pixel_location(self, point, vfov=45, hfov=60):
        point = point / point[2]
        u = ((self.pose.width/2.0) * ((point[0]/math.tan(math.radians(hfov/2.0))) + 1))
        v = ((self.pose.height/2.0) * ((point[1]/math.tan(math.radians(vfov/2.0))) + 1))
        return (u,v)

    def camera_to_world_with_pose(self, view_pose):
        return torch.from_numpy(np.linalg.inv(self.world_to_camera_with_pose(view_pose)))

    def points_in_camera_coords(self, depth_map, pixel_to_ray_array):
        camera_relative_xyz = depth_map[0].permute(1,2,0) * pixel_to_ray_array
        return torch.cat([camera_relative_xyz.double(),torch.ones(240,320,1).double()], dim=2)