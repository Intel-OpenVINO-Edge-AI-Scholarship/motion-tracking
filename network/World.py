from torch import nn
from .utils import *
import math
import numpy as np

class World():

    def __init__(self, pose):
        super(World, self).__init__()
        self.pose = pose
    
    def world_to_camera_with_pose(self, view_pose):
        lookat_pose = torch.from_numpy(self.pose.position_to_tensor({'x': view_pose.lookat.x, 'y': view_pose.lookat.y, 'z': view_pose.lookat.z}))
        camera_pose = torch.from_numpy(self.pose.position_to_tensor({'x': view_pose.camera.x, 'y': view_pose.camera.y, 'z': view_pose.camera.z}))
        up = torch.DoubleTensor([0,1,0])
        R = torch.eye(4).double()
        R[2,:3] = normalize_norm(lookat_pose - camera_pose)
        cross_P = torch.zeros(3).double()
        cross_P = crossProduct(R[2,:3], up, cross_P)
        R[0,:3] = normalize_norm(cross_P)
        cross_P = torch.zeros(3).double()
        cross_P = crossProduct(R[0,:3], R[2,:3], cross_P)
        R[1,:3] = -normalize_norm(cross_P)
        T = torch.eye(4).double()
        T[:3,3] = -camera_pose
        return torch.matmul(R,T)

    def camera_point_to_uv_pixel_location(self, point, vfov=45, hfov=60):
        point = point / point[2]
        u = ((self.pose.width/2.0) * ((point[0]/math.tan(math.radians(hfov/2.0))) + 1))
        v = ((self.pose.height/2.0) * ((point[1]/math.tan(math.radians(vfov/2.0))) + 1))
        return (u,v)

    def camera_to_world_with_pose(self, view_pose):
        return torch.from_numpy(np.linalg.inv(self.world_to_camera_with_pose(view_pose).detach().numpy().astype(np.float64)))

    def points_in_camera_coords(self, depth_map, pixel_to_ray_array):
        assert pixel_to_ray_array.shape[2] == 3
        camera_relative_xyz = torch.ones((depth_map.shape[2],depth_map.shape[3],4))
        for i in range(3):
            camera_relative_xyz[:,:,i] = depth_map[0,0,:,:] * pixel_to_ray_array[:,:,i]
        return camera_relative_xyz