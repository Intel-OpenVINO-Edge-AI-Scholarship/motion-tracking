import torch
from torch import nn
import numpy as np
from .utils import *
from itertools import product
import math
import matplotlib
import json
from pandas.io.json import json_normalize

class Flow(nn.Module):

    def __init__(self, world):
        super(Flow, self).__init__()
        self.world = world

    def view_from_detection(self):
        view = json.load(open("plugin_config/camera.json", "r"))
        view_df = json_normalize(view)
        return view_df.values

    def optical_flow(self, points, shutter_open, shutter_close, alpha=0.5, shutter_time=(1.0/60),
            hfov=60, pixel_width=320, vfov=45, pixel_height=240):
        # Alpha is the linear interpolation coefficient, 0.5 takes the derivative in the midpoint
        # which is where the ground truth renders are taken.  The photo render integrates via sampling
        # over the whole shutter open-close trajectory
        view_pose = self.world.pose.interpolate_poses(shutter_open,shutter_close,alpha)
        wTc = self.world.world_to_camera_with_pose(view_pose)
        camera_pose = torch.from_numpy(self.world.pose.position_to_tensor({'x': view_pose.camera.x, 'y': view_pose.camera.y, 'z': view_pose.camera.z}))
        lookat_pose = torch.from_numpy(self.world.pose.position_to_tensor({'x': view_pose.lookat.x, 'y': view_pose.lookat.y, 'z': view_pose.lookat.z}))

        # Get camera pixel scale constants
        uk = self.world.pose.width/2.0 * 1.0/math.tan(math.radians(hfov/2.0))
        vk = self.world.pose.height/2.0 * 1.0/math.tan(math.radians(vfov/2.0))

        # Get basis vectors
        ub1 = lookat_pose - camera_pose
        b1 = normalize_norm(ub1)
        cross_P = torch.zeros(3).double()
        cross_P = crossProduct(b1, torch.DoubleTensor([0,1,0]), cross_P)
        ub2 = normalize_norm(cross_P)
        b2 = normalize_norm(ub2)
        cross_P = torch.zeros(3).double()
        cross_P = crossProduct(b2, b1, cross_P)
        ub3 = normalize_norm(cross_P)
        b3 = -normalize_norm(ub3)

        # Get camera pose alpha derivative
        camera_end = torch.from_numpy(self.world.pose.position_to_tensor(shutter_close['camera']))
        camera_start = torch.from_numpy(self.world.pose.position_to_tensor(shutter_open['camera']))
        lookat_end = torch.from_numpy(self.world.pose.position_to_tensor(shutter_close['lookat']))
        lookat_start = torch.from_numpy(self.world.pose.position_to_tensor(shutter_open['lookat']))
        dc_dalpha = camera_end - camera_start

        # Get basis vector derivatives
        # dub1 means d unnormalised b1
        db1_dub1 = (torch.eye(3) - \
            torch.from_numpy(np.prod(np.array(list(product(b1,b1))), axis=1)).reshape(3,3))/torch.norm(ub1)
        dub1_dalpha = lookat_end - lookat_start - camera_end + camera_start
        db1_dalpha = torch.matmul(db1_dub1, dub1_dalpha)
        db2_dub2 = (torch.eye(3) - \
            torch.from_numpy(np.prod(np.array(list(product(b2,b2))), axis=1)).reshape(3,3))/torch.norm(ub2)
        dub2_dalpha = torch.DoubleTensor([-db1_dalpha[2],0,db1_dalpha[0]])
        db2_dalpha = torch.matmul(db2_dub2, dub2_dalpha)
        db3_dub3 = (torch.eye(3) - \
            torch.from_numpy(np.prod(np.array(list(product(b3,b3))), axis=1)).reshape(3,3))/np.linalg.norm(ub3)
        dub3_dalpha = torch.DoubleTensor([
                -(db2_dalpha[2]*b1[1]+db1_dalpha[1]*b2[2]),
                -(db2_dalpha[0]*b1[2] + db1_dalpha[2]*b2[0])+(db2_dalpha[2]*b1[0]+db1_dalpha[0]*b2[2]),
                (db1_dalpha[1]*b2[0]+db2_dalpha[0]*b1[1])
            ])
        db3_dalpha = -torch.matmul(db3_dub3, dub3_dalpha)

        # derivative of the rotated translation offset
        dt3_dalpha = torch.DoubleTensor([
            -torch.matmul(db2_dalpha.t(), camera_pose).item()-torch.matmul(dc_dalpha.t(), b2).item(),
            -torch.matmul(db3_dalpha.t(), camera_pose).item()-torch.matmul(dc_dalpha.t(), b3).item(),
            -torch.matmul(db1_dalpha.t(), camera_pose).item()-torch.matmul(dc_dalpha.t(), b1).item(),
        ])

        # camera transform derivative
        dT_dalpha = torch.empty((4,4))
        dT_dalpha[0,:3] = db2_dalpha.flatten()
        dT_dalpha[1,:3] = db3_dalpha.flatten()
        dT_dalpha[2,:3] = db1_dalpha.flatten()
        dT_dalpha[:3,3] = dt3_dalpha.flatten()

        # Calculate 3D point derivative alpha derivative
        dpoint_dalpha = torch.matmul(dT_dalpha.double(), points.t().double())
        point_in_camera_coords = torch.matmul(wTc.double(), points.t().double())

        # Calculate pixel location alpha derivative
        du_dalpha = uk * (dpoint_dalpha[0] * point_in_camera_coords[2] - dpoint_dalpha[2] * point_in_camera_coords[0])
        dv_dalpha = vk * (dpoint_dalpha[1] * point_in_camera_coords[2] - dpoint_dalpha[2] * point_in_camera_coords[1])
        du_dalpha = du_dalpha/(point_in_camera_coords[2]*point_in_camera_coords[2])
        dv_dalpha = dv_dalpha/(point_in_camera_coords[2]*point_in_camera_coords[2])

        # Calculate pixel location time derivative
        du_dt = du_dalpha / shutter_time
        dv_dt = dv_dalpha / shutter_time
        return torch.cat((du_dt.view(-1,1),dv_dt.view(-1,1)),dim=1)

    def flow_to_hsv_image(self, flow, magnitude_scale=1.0/100.0):

        height = self.world.pose.height
        width = self.world.pose.width
        pixel = np.meshgrid(np.arange(0,height), np.arange(0,width))
        hsv = torch.zeros((height,width,3)).double()
        v = torch.norm(flow, dim=2).double()
        idxs = np.where(v.detach().numpy().astype(np.float64) < 1e-8)
        hsv[idxs[0],idxs[1],0:3] = 0.0
        idxs = np.where(v.detach().numpy().astype(np.float64) >= 1e-8)
        direction = flow[idxs[0],idxs[1],:] / np.expand_dims(v.detach().numpy().astype(np.float64)[idxs],1)
        theta = np.arctan2(direction[:,1].detach().numpy().astype(np.float64),direction[:,0].detach().numpy().astype(np.float64))
        theta[theta<=0] = theta[theta<=0] + 2*np.pi
        if np.sum((theta < 0) | (theta > 2*np.pi)) > 0:
            raise Exception("Invalid value for theta")

        values = v[idxs].flatten().unsqueeze(1).double() * magnitude_scale
        values[values > 1] = 1.0
        # hsv[idxs[0],idxs[1],0] = torch.from_numpy(theta) / (2*np.pi)
        # hsv[idxs[0],idxs[1],1] = 1.0
        # hsv[idxs[0],idxs[1],2] = values
        return hsv

    def hsv_to_rgb(self, hsv):
        """
        Convert hsv values to rgb.

        Parameters
        ----------
        hsv : (..., 3) array-like
           All values assumed to be in range [0, 1]

        Returns
        -------
        rgb : (..., 3) ndarray
           Colors converted to RGB values in range [0, 1]
        """

        # check length of the last dimension, should be _some_ sort of rgb
        if hsv.shape[-1] != 3:
            raise ValueError("Last dimension of input array must be 3; "
                             "shape {shp} was found.".format(shp=hsv.shape))

        in_shape = hsv.shape

        h = hsv[:,:, 0]
        s = hsv[:,:, 1]
        v = hsv[:,:, 2]

        r = torch.empty(h.shape).double()
        g = torch.empty(h.shape).double()
        b = torch.empty(h.shape).double()

        i = (h * 6.0).int()
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        idx = i % 6 == 0
        r[idx] = v[idx]
        g[idx] = t[idx]
        b[idx] = p[idx]

        idx = i == 1
        r[idx] = q[idx]
        g[idx] = v[idx]
        b[idx] = p[idx]

        idx = i == 2
        r[idx] = p[idx]
        g[idx] = v[idx]
        b[idx] = t[idx]

        idx = i == 3
        r[idx] = p[idx]
        g[idx] = q[idx]
        b[idx] = v[idx]

        idx = i == 4
        r[idx] = t[idx]
        g[idx] = p[idx]
        b[idx] = v[idx]

        idx = i == 5
        r[idx] = v[idx]
        g[idx] = p[idx]
        b[idx] = q[idx]

        idx = s == 0
        r[idx] = v[idx]
        g[idx] = v[idx]
        b[idx] = v[idx]

        rgb = torch.cat([r.unsqueeze(2), g.unsqueeze(2), b.unsqueeze(2)],dim=2)
        # rgb = np.concatenate([np.expand_dims(r,2), np.expand_dims(g,2), np.expand_dims(b,2)],axis=2)

        return rgb.reshape(in_shape)
    
    def compute_flow(self, depth_map, view):

        cached_pixel_to_ray_array = self.world.pose.normalised_pixel_to_ray_array()
        depth_map[depth_map == 0.0] = 1000.0

        # This is a 320x240x3 array, with each 'pixel' containing the 3D point in camera coords
        points_in_camera = self.world.points_in_camera_coords(depth_map, cached_pixel_to_ray_array)

        # Transform point from camera coordinates into world coordinates
        ground_truth_pose = self.world.pose.interpolate_poses(build_shutter_open_view(view),build_shutter_close_view(view),0.5)
        camera_to_world_matrix = self.world.camera_to_world_with_pose(ground_truth_pose)
        points_in_world = transform_points(camera_to_world_matrix.double(),points_in_camera.double())

        # Calculate optical flow
        points_in_world = flatten_points(points_in_world)

        optical_flow_derivatives = self.optical_flow(points_in_world,build_shutter_open_view(view),build_shutter_close_view(view))
        optical_flow_derivatives = reshape_points(self.world.pose.height,self.world.pose.width,optical_flow_derivatives)

        # Write out hsv optical flow image.  We use the matplotlib hsv colour wheel
        hsv = self.flow_to_hsv_image(optical_flow_derivatives)
        # rgb = self.hsv_to_rgb(hsv)

        return hsv, optical_flow_derivatives

    def forward(self, depth_map, detection):

        view = self.view_from_detection()
        view = convert_view_to_df(view)
        return self.compute_flow(depth_map, view)

